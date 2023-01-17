# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import statsmodels.api as sm

from sklearn.model_selection import train_test_split

import seaborn as sns

import matplotlib.pyplot as plt





# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
path = "../input/covid19-in-india/AgeGroupDetails.csv"

Age = pd.read_csv(path)
Age.head()
Age.columns
Age.ndim
Age.dtypes
Age.tail()
Age.isnull().sum()
dummies = pd.get_dummies(Age.AgeGroup)

print(dummies)
Age_dummies = pd.concat([Age , dummies], axis='columns')

print(Age_dummies)
Age_dummies.drop(['AgeGroup','Percentage', '0-9', '10-19'],axis='columns',inplace=True)
Age_dummies
train, test = train_test_split(Age_dummies, test_size=0.3)
train.shape
test.shape
# split the train and test into X and Y variables

# ------------------------------------------------

train_x = train.iloc[:,0:3]; train_y = train.iloc[:,3]

test_x  = test.iloc[:,0:3];  test_y = test.iloc[:,3]

train_x
test_x
print(train_x.shape)

print(train_y.shape)

print(test_x.shape)

print(test_y.shape)

train_x.head()
train_y.head()
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
df_results = pd.DataFrame({'actual':actual, 'predicted':predicted})

print(df_results)

from sklearn import metrics  

print('Mean Absolute Error:', metrics.mean_absolute_error(test_y, pdct1))  

print('Mean Squared Error:', metrics.mean_squared_error(test_y, pdct1))  

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(test_y, pdct1)))  

path = "../input/covid19-in-india/covid_19_india.csv"

covid_india = pd.read_csv(path)
covid_india.head()
dummies = pd.get_dummies(covid_india.Date)

print(dummies)

covid_india_dummies = pd.concat([covid_india , dummies], axis='columns')

print(covid_india_dummies)

dummies1 = pd.get_dummies(covid_india.ConfirmedIndianNational)

print(dummies1)

covid_india_dummies1 = pd.concat([covid_india , dummies1], axis='columns')

print(covid_india_dummies1)
dummies2 = pd.get_dummies(covid_india.ConfirmedForeignNational)

print(dummies2)

covid_india_dummies2 = pd.concat([covid_india , dummies2], axis='columns')

print(covid_india_dummies2)
covid_india_dummies.drop(['Sno','ConfirmedIndianNational','ConfirmedForeignNational','State/UnionTerritory','Time', 'Date','29/03/20','29/04/20', '30/01/20', '30/03/20', '30/04/20', '31/01/20','31/03/20'],axis='columns',inplace=True)

print(covid_india_dummies)
train, test = train_test_split(covid_india_dummies, test_size=0.3)
train.shape
test.shape
train_x = train.iloc[:,0:3]; train_y = train.iloc[:,3]

test_x  = test.iloc[:,0:3];  test_y = test.iloc[:,3]
train_x
test_x
train_x.shape
train_y.shape
test_x.shape
test_y.shape
train_x.head()
train_y.head()
train.head()
train.tail()
train.dtypes
pred = sm.OLS(train_y, train_x).fit()

pdct1 = pred.predict(test_x)

print(pdct1)
actual = list(test_y.head(5))

type(actual)

predicted = np.round(np.array(list(pdct1.head(5))),2)

print(predicted)

type(predicted)
df_results = pd.DataFrame({'actual':actual, 'predicted':predicted})

print(df_results)
from sklearn import metrics  

print('Mean Absolute Error:', metrics.mean_absolute_error(test_y, pdct1))  

print('Mean Squared Error:', metrics.mean_squared_error(test_y, pdct1))  

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(test_y, pdct1)))  
