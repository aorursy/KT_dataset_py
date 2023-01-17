# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import time

import math

import seaborn as sns

%matplotlib inline
train = pd.read_csv('../input/home-data-for-ml-course/train.csv')

test = pd.read_csv('../input/home-data-for-ml-course/test.csv')
X = pd.concat([train.drop("SalePrice", axis=1),test], axis=0)

y = train[['SalePrice']]
X.info()
len(X.columns)
numeric_col = X.select_dtypes(exclude=['object']).drop(['MSSubClass','Id'], axis=1).copy()

numeric_col.columns
len(numeric_col.columns)
fig = plt.figure(figsize=(18,18))

for index,col in enumerate(numeric_col):

    plt.subplot(7,5,index+1)

    sns.distplot(numeric_col.loc[:,col].dropna(), kde=False)

fig.tight_layout(pad=1.0)
X.drop(['BsmtFinSF2','LowQualFinSF','EnclosedPorch',

        '3SsnPorch','ScreenPorch','PoolArea','MiscVal'], axis=1, inplace=True)
cat_train = X.select_dtypes(exclude=['int64','float64']).copy()

cat_train.columns
len(cat_train.columns)
fig = plt.figure(figsize=(20,20))

for index,col in enumerate(cat_train.columns):

    plt.subplot(9,5,index+1)

    sns.countplot(x=cat_train.iloc[:,index], data=cat_train)

fig.tight_layout(pad=1.0)
X.drop(['Utilities','Street','Condition1',

        'Condition2','LandSlope','RoofStyle','Heating','Functional',

       'GarageCond','GarageQual','Electrical','BsmtCond','ExterCond',

        'PavedDrive','SaleCondition','SaleType','MiscFeature'], axis=1, inplace=True)
len(X.columns)
plt.figure(figsize=(25,8))

plt.title('Number of missing rows')

missing_count = pd.DataFrame(X.isnull().sum(), columns=['sum']).sort_values(by=['sum'],ascending=False).head(20).reset_index()

missing_count.columns = ['features','sum']

sns.barplot(x='features',y='sum', data = missing_count)
X.drop(['PoolQC','Alley','Fence'], axis=1, inplace=True)
X.info()
len(X.columns)
plt.figure(figsize=(14,12))

correlation = numeric_.corr()

sns.heatmap(correlation, mask = correlation <0.8, linewidth=0.5, cmap='Blues')