import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
data = pd.read_csv('../input/bitcoin.csv')
data.head()
#Here we don't need columns close_ratio, spread, and market. So, i am gonna drop that columns.
data.drop(columns=['close_ratio', 'market', 'spread'], inplace = True)
data
#As we can see that we have dropped the columns that are not required.
data.shape
#We have 2039 rows and 5 columns in the data.
#Let's find min, count, max etc now to know more about the data.
data.describe()
data.dtypes
#As we can see that date is not in the date format. So, i will convert it into date datatype.
data['date'] = pd.to_datetime(data['date'])
data.dtypes
#Hureee...it's coverted
data.head()
#Also we don't need the columns high and low. So, i am gonna drop them too.
data.drop(columns=['high', 'low'], inplace = True)
data.head()
#Now we are ready to apply Linear Regression.
#Let's first import required libraries for the same.
from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split
#Setting date as index
data.set_index('date', inplace = True)
#Let's first draw a graph between opening and closing rate of the bitcoin.
x = data['open']

y = data['close']

plt.figure(figsize=(15,12))

plt.plot(x, color='red')

plt.plot(y, color = 'blue')

plt.show()

plt.xlabel('Open')

plt.ylabel('Close')
#Sorting index in case there are some rows are not in their place.

data.sort_index(inplace=True)
X = data['open']

Y = data['close']
X_train= X[:1600]

X_test = X[1600:]

Y_train= Y[:1600]

Y_test = Y[1600:]
x_train = np.array(X_train).reshape(-1,1)

x_test = np.array(X_test).reshape(-1,1)
LR = LinearRegression()
LR.fit(x_train, Y_train)
LR.score(x_test, Y_test)
plt.figure(figsize=(15,12))

plt.plot(Y_train)

plt.plot(Y_test)
pred = LR.predict(np.array(X_test).reshape(-1,1))
pred[:4]