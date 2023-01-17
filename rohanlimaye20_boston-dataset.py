# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

%matplotlib inline

import matplotlib.pyplot as plt

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

sns.set()



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



        

        

# Any results you write to the current directory are saved as output.
df=pd.read_csv('../input/bostonhoustingmlnd/housing.csv')

df.head()
df.info()
df.describe()
#Going w/o mean normalisation



x=df.iloc[:,1:4].values

print(type(x))

print(x.shape)

print(x)
y=df.iloc[:,3].values

print(type(y))

print(y.shape)

print(y)
plt.figure(figsize=(10,5))

sns.distplot(y)

plt.show()
from sklearn.model_selection import train_test_split 

from sklearn.linear_model import LinearRegression

from sklearn import metrics
x_train,x_test,y_train,y_test= train_test_split(x,y,train_size=0.8,random_state=0)
print(x_train.shape)

print(type(x_train))

print(x_test.shape)

print(type(x_test))

print(y_train.shape)

print(type(y_train))

print(y_test.shape)

print(type(y_test))

regressor=LinearRegression()

regressor.fit(x_train,y_train)
regressor.coef_
y_pred=regressor.predict(x_test)
print(type(y_pred))

print(y_pred.shape)

print(y_pred)
df1 = pd.DataFrame({'Actual':y_test,'Predicted':y_pred})

df1.head()
df_plot=df1.head(10)
df_plot.plot(kind='bar')