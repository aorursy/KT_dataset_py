# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.model_selection import train_test_split



# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import pandas as pd
df = pd.read_csv('../input/creditcardfraud/creditcard.csv')
df.head()
df.isnull().sum()
df.dtypes
df.count
df.describe
len(df[df['Class']==0])
len(df[df['Class']==1])
df.columns
total_cols = len(df.columns)
total_cols
x = df.values[:,:total_cols-1]
y = df.values[:,total_cols-1]

#split the data into training set and testing set
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25, random_state = 0)
print(x_train)
print(y_train)
print(x_test)
print(y_test)
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(x_train, y_train)


y_pred = regressor.predict(x_test)
print(y_pred)
df.corr()
df.dtypes
df.count
df.corr()
cor = df.iloc[:,0:30].corr()
print(cor)
train,test = train_test_split(df,test_size=0.3)
print(train.shape)
print(test.shape)
train_x = train.iloc[:,0:3]; train_y = train.iloc[:,3]
test_x  = test.iloc[:,0:3];  test_y = test.iloc[:,3]
print(train_x)
print(test_x)

print(train_x.shape)
print(test_x.shape)
print(train_y.shape)
print(test_y.shape)
train_x.head()
train_y.head()
train_x.tail()
train_y.tail()
train.head()
train.dtypes
#for predict the value
value = sm.OLS(train_y , train_x).fit()
pred = value.predict(test_x)
print(pred)
#Store the Actual and predicted values in a dataframe for comprison
actual = list(test_y.head(50))
type(actual)
predicated = np.round(np.array(list(pred.head(50))),2)
print(predicated)
type(predicated)
df_results = pd.DataFrame({'actual':actual, 'predicated':predicated})
print(df_results)

#To check Accuracy:
from sklearn import metrics  
print('Mean Absolute Error:', metrics.mean_absolute_error(test_y, pred))  
print('Mean Squared Error:', metrics.mean_squared_error(test_y, pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(test_y, pred)))  
