import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from scipy.stats import norm #Analysis 

from sklearn.preprocessing import StandardScaler #Analysis 

from scipy import stats



plt.style.use('seaborn')







import missingno as msno

import warnings

warnings.filterwarnings('ignore')

import datetime

from datetime import datetime



%matplotlib inline
df_train= pd.read_csv('../input/train.csv')

df_test = pd.read_csv('../input/test.csv')
df_train.dtypes
df_train.describe()
msno.matrix(df= df_train.iloc[: , :],figsize=(8,8), color = (0.8,0.5,0.2))
msno.matrix(df= df_test.iloc[: , :],figsize=(8,8), color = (0.8,0.5,0.2))
df_train[['season','count']].groupby(['season'], as_index=True).mean()
df_train[['season','count']].groupby(['season'], as_index=True).mean().plot.bar()
data = pd.concat([df_train['count'], df_train['season']], axis=1)

f, ax = plt.subplots(figsize=(8, 6))

fig = sns.boxplot(x='season', y="count", data=data)
data = pd.concat([df_train['count'], df_train['atemp']], axis=1)

f, ax = plt.subplots(figsize=(8, 6))

fig = sns.regplot(x='atemp', y="count", data=data)
df_train[['season','count']].groupby(['season'], as_index=True).mean().plot()
ch=df_train
ch['date']  = ch.datetime.apply(lambda x: x.split()[0])

ch['hour'] = ch.datetime.apply(lambda x: x.split()[1].split(':')[0])
ch['weekday'] = ch.date.apply(lambda dateString : datetime.strptime(dateString, '%Y-%m-%d').weekday())

ch['month'] = ch.date.apply(lambda dateString : datetime.strptime(dateString, '%Y-%m-%d').month)
#df_train.drop(['datetime'], axis=1, inplace=True)
ch['year']=[d.split('-')[0] for d in ch.date]

ch['day']=[d.split('-')[2] for d in ch.date]
ch['hour']=pd.to_numeric(ch['hour'])

ch['day']=pd.to_numeric(ch['day'])

ch['year']=pd.to_numeric(ch['year'])

ch.dtypes
ch.head(10)
y=[2011,2012]

m=list(range(1,13))

d=list(range(1,20))

h=list(range(0,24))

#for row_num in list(range(0,len(ch))):

#    print(ch['day'].loc[row_num])

#
y=[2011,2012]

m=list(range(1,13))

d=list(range(1,20))

h=list(range(0,24))

#i=0

#for row_num in list(range(0,len(ch))):

#    h=list(range(i,i+24))

#        if ch['hour'][row_num]%24==h[ch['hour'][row_num]%24]:

#            continue

        
df_train.head()
#plt.boxplot(out['windspeed'])
f, ax = plt.subplots(figsize=(8, 6))

sns.distplot(df_train['temp'])

print("Skewness: %f" % df_train['temp'].skew())

print("Kurtosis: %f" % df_train['temp'].kurt())
f, ax = plt.subplots(figsize=(8, 6))

sns.distplot(df_train['atemp'])

print("Skewness: %f" % df_train['atemp'].skew())

print("Kurtosis: %f" % df_train['atemp'].kurt())
f, ax = plt.subplots(figsize=(8, 6))

sns.distplot(df_train['humidity'])

print("Skewness: %f" % df_train['humidity'].skew())

print("Kurtosis: %f" % df_train['humidity'].kurt())
f, ax = plt.subplots(figsize=(8, 6))

sns.distplot(df_train['windspeed'])

print("Skewness: %f" % df_train['windspeed'].skew())

print("Kurtosis: %f" % df_train['windspeed'].kurt())
fig = plt.figure(figsize = (15,10))



fig.add_subplot(1,2,1)

res = stats.probplot(df_train['windspeed'], plot=plt)



fig.add_subplot(1,2,2)

res = stats.probplot(np.log1p(df_train['windspeed']), plot=plt)
df_train['windspeed'] = np.log1p(df_train['windspeed'])

#histogram

f, ax = plt.subplots(figsize=(8, 6))

sns.distplot(df_train['windspeed'])
f, ax = plt.subplots(figsize=(8, 6))

sns.distplot(df_train['count'])

print("Skewness: %f" % df_train['count'].skew())

print("Kurtosis: %f" % df_train['count'].kurt())
df_train['count'] = np.log1p(df_train['count'])

#histogram

f, ax = plt.subplots(figsize=(8, 6))

sns.distplot(df_train['count'])
import scipy as sp



cor_abs = abs(df_train.corr(method='spearman')) 

cor_cols = cor_abs.nlargest(n=10, columns='count').index # count과 correlation이 높은 column 10개 뽑기(내림차순)

# spearman coefficient matrix

cor = np.array(sp.stats.spearmanr(df_train[cor_cols].values))[0] # 10 x 10

print(cor_cols.values)

plt.figure(figsize=(10,10))

sns.set(font_scale=1.25)

sns.heatmap(cor, fmt='.2f', annot=True, square=True , annot_kws={'size' : 8} ,xticklabels=cor_cols.values, yticklabels=cor_cols.values)