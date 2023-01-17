#Libraries

import pandas as pd

import numpy as np

import scipy.stats as st

import matplotlib.pyplot as plt

import seaborn as sns



#Ignoring warnings

import warnings

warnings.filterwarnings("ignore")
#Reading CSV Dataset

df_NBA = pd.read_csv(r'../input/NBA_season1718_salary.csv')

df_NBA.head()

#DF Key information

df_NBA.info()

#Renaming and deleting columns

df_NBA.columns = ['cod', 'player', 'team', 'salary']

del df_NBA['cod']

df_NBA.head()

team_index = df_NBA['team'].value_counts()

sns.catplot(data=df_NBA,

            x='team',

            order=team_index.index,

            kind='count',

            aspect=2.5,

            palette='GnBu_d')
#Histogram and KDE

plt.figure(figsize=(8, 4))

sns.distplot(df_NBA['salary'], bins=40)

#Probability Density Function (PDF) Chart

x = df_NBA['salary']



plt.figure(figsize=(8, 4))

plt.plot(x, st.norm.pdf(x, x.mean(), x.std()))

plt.show()

#Creating a column with the salary log to normalize the distribution

df_NBA['salary_log'] = np.log1p(df_NBA['salary'])

sns.distplot(df_NBA['salary_log'], bins=25)

#Dividing by the mean and standard deviation to standardize the serie in a new column

df_NBA['norm_log_salary'] = ((df_NBA['salary_log'] - df_NBA['salary_log'].mean()) / df_NBA['salary_log'].std())

sns.distplot(df_NBA['norm_log_salary'], bins=25)

print(f"""Mean: {df_NBA.norm_log_salary.mean():.4f}

Standard: {df_NBA.norm_log_salary.std():.4f}""")

norm_mean = df_NBA.norm_log_salary.mean()

norm_std = df_NBA.norm_log_salary.std()



p_value = st.norm(norm_mean, norm_std).sf(2*norm_std) * 2 #to sides

p_value

z_score_inf = st.norm.interval(alpha=0.95, loc=norm_mean, scale=norm_std)[0]

z_score_sup = st.norm.interval(alpha=0.95, loc=norm_mean, scale=norm_std)[1]



print(f'{z_score_inf:.4f} <--------> {z_score_sup:.4f}')

#Players

df_NBA_lower = df_NBA[df_NBA['norm_log_salary'] < z_score_inf]

df_NBA_lower

#Players by team

team_index = df_NBA_lower['team'].value_counts()

team_index

#Plot players by team

plt.figure(figsize=(12, 5))

sns.countplot(df_NBA_lower['team'],

              order=team_index.index,

              palette='Blues_r')

print(f"""Players with a lower salary than the average: 

Total - {df_NBA_lower.shape[0]}

Rate - {df_NBA_lower.shape[0] / df_NBA.shape[0] * 100:.2f}%""")

#Players

df_NBA_higher = df_NBA[df_NBA['norm_log_salary'] > z_score_sup]

df_NBA_higher

print(f"""Players with a higher salary than the average: 

Total - {df_NBA_higher.shape[0]}

Rate - {df_NBA_higher.shape[0] / df_NBA.shape[0] * 100:.2f}%""")

#p-value and alpha max to the highest salary

p_value = st.norm(norm_mean, norm_std).sf(df_NBA['norm_log_salary'].max())

alpha = 1 - p_value

print(f'P-value: {p_value:.3f}\nAlpha Max: {alpha:.3f}\nWe can confirm that the highest salary is on the distribution!')
