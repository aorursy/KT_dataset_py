#data manipulation

import numpy as np

import pandas as pd

pd.set_option('display.max_columns', 500)



#plotting libraries

import matplotlib.pyplot as plt

import seaborn as sns



#statistical software

import scipy

from scipy.stats import norm

from scipy import stats
data = pd.read_csv('../input/Batting.csv')

#retrieved data from Sean Lahman Baseball Database
#total plate appearances, denominator in OBP

data['TPA'] = (data['AB'] + data['BB'] + data['IBB'] + data['SF'] + data['HBP'])

#OBS, one of the factors in OPS

data['OBP'] = (data['H'] + data['BB'] + data['IBB'] + data['HBP']) / data['TPA']

#total bases, numerator in SLG. first must find out how many singles were hit. data does not capture singles

data['XB'] = data['2B'] + data['3B'] + data['HR']

data['1B'] = data['H'] - data['XB']

data['TB'] = (1 * data['1B']) + (2 * data['2B']) + (3 * data['3B']) + (4 * data['HR'])

#creating SLG

data['SLG'] = data['TB'] / data['AB'] 

#finally creating OPS

data['OPS'] = data['OBP'] + data['SLG'] 
#after 1955 there are zero nulls. i think we can sacrifice the old data

df = data[(data.yearID >= 1955) & (data.AB >= 225)]
df.head()
df.columns
df.info()
df.describe()
sns.distplot(df['OPS'], fit = norm);

fig = plt.figure()

res = stats.probplot(df['OPS'], plot = plt)
#OPS is a bit skewed, positive skewness/right skewed

#apply log trans

df['logOPS'] = np.log(df['OPS'])
#testing log normality

sns.distplot(df['logOPS'], fit = norm);

fig = plt.figure()

res = stats.probplot(df['logOPS'], plot = plt)

#thats nice.
df_STL = df[(data.teamID == 'SLN')]

df_NYY = df[(data.teamID == 'NYA')]
df_STL.OPS.describe()

df_NYY.OPS.describe()
#plot data, with trendline

sns.regplot(x = 'yearID', y = 'OPS', data = df_STL, fit_reg = True) 

sns.regplot(x = 'yearID', y = 'OPS', data = df_NYY, fit_reg = True) 
#St. Louis Cardinals

f, ax = plt.subplots(figsize=(18, 18))

sns.heatmap(df_STL.corr(), annot=True, linewidths=.5, fmt= '.1f', ax=ax)

plt.show()

#New York Yankees

f, ax = plt.subplots(figsize=(18, 18))

sns.heatmap(df_NYY.corr(), annot=True, linewidths=.5, fmt= '.1f', ax=ax)

plt.show()
STLops = df_STL.OPS

NYYops = df_NYY.OPS



t, p = stats.ttest_ind(STLops, NYYops)

print("t = " + str(t))

print("p = " + str(p))