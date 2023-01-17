# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import statsmodels.api as sm



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
path = "../input/1000-cameras-dataset/camera_dataset.csv"

df = pd.read_csv(path)
df.head()
df.dtypes
df.describe
df.ndim
df.columns
df.count
df.isnull().sum()
df.shape
df.isnull().sum()
df.drop(['Model'],axis=1,inplace=True)
df.drop(['Release date'],axis=1,inplace=True)
df.shape
df['Macro focus range'].fillna(df['Macro focus range'].mean())
df['Storage included'].fillna(df['Storage included'].mean())
df['Weight (inc. batteries)'].fillna(df['Weight (inc. batteries)'].mean())

df['Dimensions'].fillna(df['Dimensions'].mean())
df['Price'].fillna(df['Price'].mean())
train , test = train_test_split(df,test_size=0.3)
print(train.shape)
print(test.shape)
train_x = train.iloc[:,0:3]; train_y = train.iloc[:,3]

test_x  = test.iloc[:,0:3];  test_y = test.iloc[:,3]
print(train_x)
print(test_x)
print(train_x.shape)
train_y.shape
test_x.shape
test_y.shape
train_x.head()
train_y.head()
train.head()
train.tail()
train.dtypes
fit = sm.OLS(train_y,train_x).fit()
#Predction

pred = fit.predict(test_x)
pred
#Actual

actual = list(test_y.head(103))
actual
predicted = np.round(np.array(list(pred.head(103))),2)
predicted
type(predicted)
#Actual vs Predicted

df_results = pd.DataFrame({'actual':actual, 'predicted':predicted})

df_results
from sklearn import metrics  

print('Mean Absolute Error:', metrics.mean_absolute_error(test_y, pred))  

print('Mean Squared Error:', metrics.mean_squared_error(test_y,pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(test_y, pred)))  
