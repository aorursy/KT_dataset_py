import numpy as np

import pandas as pd

pd.set_option('display.max_columns', None)

import matplotlib.pyplot as plt

import seaborn as sns



import scipy.stats

from scipy.stats import norm

from sklearn.preprocessing import StandardScaler

from scipy import stats



import statsmodels.api as sm

from statsmodels.formula.api import ols



import warnings

warnings.filterwarnings("ignore", message = 'numpy.dtype size changed')
df_b = pd.read_csv("../input/Batting.csv")

df_p = pd.read_csv("../input/People.csv")

df_s = pd.read_csv("../input/Salaries.csv")
df_b.columns
df_b.head()
df_b.describe()
df_b.isnull().sum()
df_p.columns
df_p.head()
df_p.describe()
df_p.isnull().sum()
df_s.columns
df_s.head()
df_s.describe()
df_s.isnull().sum()
df_b = df_b[df_b.yearID >= 2008]

df_b = df_b[df_b.AB >= 250]
df_names = df_p[['playerID', 'nameFirst', 'nameLast','bats', 'birthYear']]

df_salaries = df_s[['playerID', 'salary']]

df_merge = pd.merge(df_b, df_names, on = 'playerID')

df = pd.merge(df_merge, df_salaries, on = 'playerID')

df.head()
df.isnull().sum()
#first must create a feature for singles ('1B')

df['1B'] = df['H'] - (df['2B'] + df['3B'] + df['HR']) 

#creating wOBA

df['num'] = (0.69 * df['BB']) + (0.72 * df['HBP']) + (0.89 * df['1B']) + (1.27 * df['2B']) + (1.62 * df['3B']) + (2.10 * df['HR'])

df['dem'] = (df['AB'] + df['BB'] - df['IBB'] + df['SF'] + df['HBP'])

df['wOBA'] = df['num']/df['dem']



df['wOBA'].describe()
df['age'] = df['yearID'] - df['birthYear']



df['age'].describe()
corrmat = df.corr()

f, ax = plt.subplots(figsize = (12,9))

sns.heatmap(corrmat, vmax = 0.8, square = True);
sns.distplot(df['salary']);
sns.distplot(df['wOBA']);
sns.distplot(df['age']);
sns.distplot(df['BB']);
plt.figure(figsize = (20,10))

sns.boxplot(x = "teamID", y = "wOBA", data = df)

plt.xticks(rotation = 90)



plt.figure(figsize = (20,10))

sns.boxplot(x = "yearID", y = "wOBA", hue = "bats", data = df)

plt.xticks(rotation = 90)



plt.figure(figsize = (5,10))

sns.boxplot(x = "bats", y = "wOBA", data = df)

plt.xticks(rotation = 90)



plt.figure(figsize = (5,10))

sns.boxplot(x = "lgID", y = "wOBA", data = df)

plt.xticks(rotation = 90)
sns.jointplot(x = "wOBA", y = "BB", data = df, height = 10)

sns.jointplot(x = "wOBA", y = "SO", data = df, height = 10)

sns.jointplot(x = "wOBA", y = "age", data = df, height = 10)

sns.jointplot(x = "wOBA", y = "salary", data = df, height = 20)
sns.set()

cols = ['1B', 'HR', 'salary', 'wOBA', 'age']

sns.pairplot(df[cols], height = 2.5)

plt.show();
sns.distplot(df['wOBA'], fit = norm);

fig = plt.figure()

res = stats.probplot(df['wOBA'], plot = plt)
df['wOBA_log'] = np.log(df['wOBA'])



sns.distplot(df['wOBA_log'], fit = norm);

fig = plt.figure()

res = stats.probplot(df['wOBA_log'], plot = plt)
#one way anova

scipy.stats.f_oneway(df['wOBA_log'][df['bats'] == 'L'],

               df['wOBA_log'][df['bats'] == 'R'],

               df['wOBA_log'][df['bats'] == 'B'])
anova = ols('wOBA_log ~ C(bats)', data = df).fit()

anova.summary()
df_lefty = df[df['bats'] == 'L']

df_righty = df[df['bats'] == 'R']



stats.ttest_ind(df_lefty.wOBA_log, df_righty.wOBA_log)