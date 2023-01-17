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
df = pd.read_csv('/kaggle/input/insurance/insurance.csv')
df.head(2)
df.info()
df.describe()
df.isnull().sum()
import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
sns.distplot(df.age, kde = False, bins = 40);
df.age.value_counts().head()
df.columns
sns.countplot('sex', data = df)
df.columns


sns.distplot(df.bmi, kde = False, bins = 40);
sns.countplot('children', data = df);
sns.countplot('smoker', data = df);
df.columns
sns.countplot('region', data = df);
sns.distplot(df.charges, kde = False, bins = 40);
# Create Dummy Variables
sex = pd.get_dummies(df.sex, drop_first= True)
children = pd.get_dummies(df.children, drop_first= True)
smoker = pd.get_dummies(df.smoker, drop_first= True)
region = pd.get_dummies(df.region, drop_first= True)
train = pd.concat([df,sex, children, smoker, region], axis = 1)
train.head(2)
train.drop(['sex', 'children', 'smoker', 'region'], axis = 1, inplace = True)
train
from sklearn.model_selection import train_test_split
X = train.drop('charges', axis = 1)
y = train.charges
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
X_train
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train, y_train)
predictions = lm.predict(X_test)
predictions
from sklearn import metrics 

print('MAE:', metrics.mean_absolute_error(y_test, predictions))

print('MSE:', metrics.mean_squared_error(y_test, predictions))

print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))

print(lm.score(X_test,y_test))
from sklearn.ensemble import RandomForestRegressor
RFR = RandomForestRegressor()
RFR.fit(X_train, y_train)
print(RFR.score(X_test,y_test))
Rpredictions = RFR.predict(X_test)
from sklearn import metrics 

print('MAE:', metrics.mean_absolute_error(y_test, Rpredictions))

print('MSE:', metrics.mean_squared_error(y_test, Rpredictions))

print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, Rpredictions)))
from sklearn.neighbors import KNeighborsRegressor
KNN = KNeighborsRegressor()
KNN.fit(X_train, y_train)
print(KNN.score(X_test,y_test))
Kpredictions = KNN.predict(X_test)
Kpredictions
print('MAE:', metrics.mean_absolute_error(y_test, Kpredictions))

print('MSE:', metrics.mean_squared_error(y_test, Kpredictions))

print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, Kpredictions)))