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
import seaborn as sb

from matplotlib import pyplot as plt
## importing datasets

train = pd.read_csv('/kaggle/input/restaurant-revenue-prediction/train.csv.zip')

test = pd.read_csv('/kaggle/input/restaurant-revenue-prediction/test.csv.zip')
## checking null values

print("Training Dataset",train.columns[train.isnull().any()].tolist())

print("Test DataSet",test.columns[test.isnull().any()].tolist())
## train Analysis

train.info()
train['Open Date'] = pd.to_datetime(train['Open Date'])

test['Open Date'] = pd.to_datetime(test['Open Date'])

## checking numerical and categorical features



numerical_features = train.select_dtypes(include=np.number).columns.tolist()

categorical_features = train.select_dtypes(exclude=[np.number,np.datetime64]).columns.tolist()
numerical_features
categorical_features
train[numerical_features].head()
train[categorical_features].head()
sb.distplot(train.revenue,bins=50)
train[train.revenue > 10000000]
## dropping them

train = train[train.revenue < 10000000]

train.reset_index(drop=True).head()
cities = train.City.value_counts().head().keys().tolist()

cities
train['City Group'].value_counts()
train.Type.value_counts()
sb.barplot(y=train.revenue, x=train[train.City.isin(cities)].City, data=train,estimator=sum, hue=train['City Group'])

plt.xlabel('Cities')
train.groupby(by=['City']).revenue.sum().sort_values(ascending=False).head()
## Time series Graph

sb.lineplot(y=train.revenue, x=train[train.City == 'İstanbul']['Open Date'])