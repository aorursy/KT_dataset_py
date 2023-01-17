# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsRegressor



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output. 
df=pd.read_csv('../input//USD_PKR Historical Data (1999-2019) - USD_PKR Historical Data (1).csv')

df['Date']=pd.to_datetime(df['Date'], errors='coerce')#some pandas miracle here

df['Price']=pd.to_numeric(df['Price'], errors='coerce')# The Şimdi column look as object so i turned it to numeric

print(df.head(2), df.tail(2))

print(df.info())
print(df.describe()) 
X=df['Price']

Y=df['Date']

X=np.array(X).reshape((len(X), 1))#i got some issues with shapes but i found this solution on stackoverflow

Y=np.array(Y).reshape((len(Y), 1))

fig=plt.figure()

ax=fig.add_subplot(111)

ax.plot_date(Y, X, '.')

plt.show()



sns.distplot(df['Price'])
Y=Y.astype('float64')

x_train, x_test, y_train, y_test=train_test_split(X, Y, test_size=0.2)



#firstly let's try linear regression

lin_reg=LinearRegression()

lin_reg.fit(x_train, y_train)

co=lin_reg.score(X, Y)



#now we will try KNeighborsRegressor

knr=KNeighborsRegressor()

knr.fit(X, Y)

co2=knr.score(X, Y)

X2=knr.predict(X)

X2.astype('float64')



print("linear regsession: ", co)

print("KNeighbors Regressor: ", co2)#we have 89 precent accuracy with knr

print("prediction: ", X2)



plt.scatter(X, Y, color='red')#let's plot the normal values and predicted values

plt.plot(X, X2, color='blue')

plt.show