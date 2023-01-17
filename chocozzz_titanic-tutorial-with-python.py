# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df_train = pd.read_csv("../input/train.csv")
df_train.head()
df_train.tail()
df_train['Fare'].describe()
df_train[df_train['Fare'] > 500]
df_train[df_train['Fare'] == 0].head()
df_train[df_train['Age'] < 5].head()
df_train[df_train['Fare'] == 0]
df_train[df_train['Name'] == '']
import numpy as np
from sklearn.metrics import accuracy_score
import pandas as pd
import re
import sklearn
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
data = pd.concat([df_train['Fare'], df_train['Embarked']], axis=1)
f, ax = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(x='Embarked', y="Fare", data=data)
df_train[df_train['Fare']>500] 
#histogram
#missing_data = missing_data.head(20)

#missing data
total = df_train.isnull().sum().sort_values(ascending=False)
percent = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
percent_data = percent.head(20)
percent_data.plot(kind="bar", figsize = (8,6), fontsize = 10)
plt.xlabel("", fontsize = 20)
plt.ylabel("", fontsize = 20)
plt.title("Total Missing Value (%)", fontsize = 20)
(df_train.isnull().sum()/df_train.isnull().count())
adsf = df_train.isnull()
import missingno as msno
missingdata_df = df_train.columns[df_train.isnull().any()].tolist()
msno.heatmap(df_train[missingdata_df], figsize=(8,6))
plt.title("Correlation with Missing Values", fontsize = 20)
df_train['hasCabin'] = df_train['Cabin'].isnull().apply(lambda x: 0 if x == True else 1)
df_train['hasAge'] = df_train['Age'].isnull().apply(lambda x: 0 if x == True else 1)
df_train.head()
df_train.corr() 
data = pd.concat([df_train['Fare'], df_train['hasCabin']], axis=1)
f, ax = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(x='hasCabin', y="Fare", data=data)
from statsmodels.graphics.mosaicplot import mosaic
mosaic(df_train, ['hasCabin', 'Pclass'],gap=0.02)
plt.show()
df_train[df_train['Embarked'].isnull()] 
df_train.shape[0]
# 완전 제거법
# 결측치가 들어가 있는 행 제거 
a = df_train.dropna(axis = 0)

# 결측치가 들어가 있는 열 제거 
b = df_train.dropna(axis = 1)

# 특정 열을 대상으로 제거 
c = df_train[df_train['Cabin'].notnull()]
print(a.shape, b.shape, c.shape)

data = pd.concat([df_train['Fare'], df_train['Embarked']], axis=1)
f, ax = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(x='Embarked', y="Fare", data=data)
f, ax = plt.subplots(figsize=(8, 6))
sns.boxplot(x="Embarked", y="Fare", hue="Pclass",
               data=df_train, palette="Set3")
df_train[df_train['Embarked'].isnull()]
# 특정 행의 열을 채우는 방법 
df_train.loc[61, 'Embarked'] = 'S'
df_train.loc[829, 'Embarked'] = 'S'

# 특정 열의 결측치를 채우는 방법 
df_train['Embarked'] = df_train['Embarked'].fillna('S')
df_train['Fare'] = df_train['Fare'].fillna(df_train.groupby(['Embarked', 'Pclass'])['Fare'].agg({'median'}))
df_train.groupby(['Embarked', 'Pclass'])['Fare'].agg({'median'})
