import pandas as pd

import numpy as np

import seaborn as sns

import warnings

warnings.simplefilter(action='ignore')

import matplotlib.pyplot as plt

from tqdm import tqdm

from datetime import datetime

import json

from sklearn.preprocessing import LabelEncoder
train = pd.read_csv('../input/tmdb-box-office-prediction/train.csv')

train.info()
test = pd.read_csv('../input/tmdb-box-office-prediction/test.csv')

test.info()
train.head()
test.head()
train.describe(include='all')
test.describe(include='all')
train.isna().sum()
test.isna().sum()
sns.jointplot(x='budget', y='revenue', data=train, height=11, ratio=4, color='g')

plt.show()
sns.jointplot(x='popularity', y='revenue', data=train, height=11, ratio=4, color='g')

plt.show()
sns.jointplot(x='runtime', y='revenue', data=train, height=11, ratio=4, color='g')

plt.show()
sns.distplot(train.revenue)
train.revenue.describe()
train['logRevenue'] = np.log1p(train['revenue'])

sns.distplot(train['logRevenue'])
# 년도 자료는 2자리 수로 제공되지만, 몇몇 레코드에서 4자리로 표현하므로 둘을 나눠서 해야 함



train[['release_month','release_day','release_year']] = train['release_date'].str.split('/', expand=True).replace(np.nan, -1).astype(int)



# 년도 자료가 4 자리인 것 때문에 추가 된 라인 

train.loc[(train['release_year'] <= 19) & (train['release_year'] <100), 'release_year' ] += 2000

train.loc[(train['release_year'] > 19) & (train['release_year'] <100), 'release_year' ] += 1900



releaseDate = pd.to_datetime(train['release_date'])

train['release_dayofweek'] = releaseDate.dt.dayofweek

train['release_quarter'] = releaseDate.dt.quarter
plt.figure(figsize = (20,12))

sns.countplot(train['release_year'].sort_values())

plt.title('Movie Release count by Year', fontsize = 20)

loc, labels = plt.xticks()

plt.xticks(fontsize = 12, rotation = 90)

plt.show()
plt.figure(figsize=(20,12))

sns.countplot(train['release_month'].sort_values())

plt.title('Release Month Count', fontsize = 20)

loc, labels = plt.xticks()

loc, labels = loc, ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

plt.xticks(loc, labels, fontsize=20)

plt.show()
plt.figure(figsize=(20,12))

sns.countplot(train['release_day'].sort_values())

plt.title('Release Day count',fontsize=20)

plt.xticks(fontsize=20)

plt.show()
plt.figure(figsize=(20,12))

sns.countplot(train['release_dayofweek'].sort_values())

plt.title('Release Dayof count',fontsize=20)

loc, labels = plt.xticks()

loc, labels = loc, ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]

plt.xticks(loc, labels, fontsize=20)

plt.show()
plt.figure(figsize=(20,12))

sns.countplot(train['release_quarter'].sort_values())

plt.title('Release Quater count',fontsize=20)

plt.xticks(fontsize=20)

plt.show()
train['meanRevenueByYear'] = train.groupby('release_year')['revenue'].aggregate('mean')

train['meanRevenueByYear'].plot(figsize=(15,10),color='g')

plt.xticks(np.arange(1920,2018,4)) # 1920부터 2018까지 4 단위로 x축 나오도록 함

plt.xlabel('ReleaseYear')

plt.ylabel('Revenue')

plt.title('Movie Mean Revenue By Year', fontsize=20)

plt.show()
train['meanRevenueByMonth'] = train.groupby('release_month')['revenue'].aggregate('mean')

train['meanRevenueByMonth'].plot(figsize=(15,10),color='g')

plt.xlabel('ReleaseMonth')

plt.ylabel('Revenue')

plt.title('Movie Mean Revenue By Month', fontsize=20)

plt.show()