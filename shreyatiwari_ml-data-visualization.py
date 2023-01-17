# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

from sklearn import linear_model

from sklearn.model_selection import train_test_split

import statsmodels.api as sm



from matplotlib import rcParams#Size of plots 

import plotly as py

import cufflinks

from tqdm import tqdm_notebook as tqdm







# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
path = "../input/novel-corona-virus-2019-dataset/covid_19_data.csv"

data = pd.read_csv(path)
data.head()
data.isnull().sum()
data.describe(include = "all")
data.info()
data.hist(figsize=(20,30))
sns.boxplot(x="Confirmed" , y="Deaths" , data=data)
sns.boxplot(x="Confirmed" , y="Recovered" , data=data)
sns.boxplot(x="Deaths" , y="Recovered" , data=data)
pd.crosstab(data['Confirmed'] , data['Deaths'])
pd.crosstab(data['Recovered'],data['Deaths'])
pd.crosstab(data['Confirmed'],data['Recovered'])
sns.pairplot(data)
data.mean()
data['Confirmed'].mean()
data['Confirmed'].std()
data['Deaths'].mean()
data['Recovered'].mean()
data['Recovered'].std()
data['Recovered'].var()
data['Confirmed'].median()
data['Confirmed'].var()
data['Deaths'].median()
data['Deaths'].var()
data['Deaths'].std()
data.cov()
data['Deaths'].std()
data['Recovered'].std()
data['Recovered'].median()
sns.distplot(data['Confirmed'],bins=1)
sns.distplot(data['Deaths'],bins=5)
sns.distplot(data['Recovered'],bins=1)
sns.countplot(x='Confirmed' , data=data)
sns.countplot(x='Deaths' , data=data)
sns.countplot(x='Recovered' , data=data)
sns.boxplot(x=data['Confirmed'])
sns.boxplot(x=data['Confirmed'] , y=data['Deaths'])
sns.boxplot(x=data['Deaths'])
sns.boxplot(x=data['Recovered'])
sns.boxplot(x=data['Deaths'] , y=data['Recovered'])
sns.boxplot(x=data['Confirmed'] , y=data['Recovered'])
corr = data.corr()

corr
sns.heatmap(corr , annot=True)
data.dtypes
data.drop(['ObservationDate','Province/State' , 'Country/Region' , 'Last Update'], axis='columns', inplace=True)



# Examine the shape of the DataFrame (again)

print(data.shape)
# split the dataset into train and test

# --------------------------------------

train, test = train_test_split(data, test_size = 0.3)

print(train.shape)

print(test.shape)

# split the train and test into X and Y variables

# ------------------------------------------------

train_x = train.iloc[:,0:1]; train_y = train.iloc[:,1]

test_x  = test.iloc[:,0:1];  test_y = test.iloc[:,1]

print(train_x)

print(test_x)



print(train_x.shape)

print(train_y.shape)

print(test_x.shape)

print(test_y.shape)

train_y.head()
train_x.head()
train.head()
train.tail()
train.dtypes
lm1 = sm.OLS(train_y, train_x).fit()

pdct1 = lm1.predict(test_x)

print(pdct1)
actual = list(test_y.head(5))

type(actual)

predicted = np.round(np.array(list(pdct1.head(5))),2)

print(predicted)

type(predicted)

data_results = pd.DataFrame({'actual':actual, 'predicted':predicted})

print(data_results)

from sklearn import metrics  

print('Mean Absolute Error:', metrics.mean_absolute_error(test_y, pdct1))  

print('Mean Squared Error:', metrics.mean_squared_error(test_y, pdct1))  

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(test_y, pdct1)))  