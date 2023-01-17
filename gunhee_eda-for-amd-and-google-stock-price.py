import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline
# historical daily Stock price Data downloaded from yahoo.com/finance

# choose 2 big companies randomly.

# read .csv files into data frame



amd = pd.read_csv('../input/AMD.csv')

google = pd.read_csv('../input/GOOGL.csv')
# 2009-05-23 ~ 2017-05-03

# have 2001 rows and 7 columns



amd.head()

#amd.info()

#amd.shape

#amd.describe
# drop high, low, close columns, and change name of Adj Close to Close.



amd.drop(['High','Low','Close'], axis=1, inplace=True)

amd.rename(columns={'Adj Close' : 'Close'}, inplace=True)



google.drop(['High', 'Low', 'Close'], axis=1, inplace=True)

google.rename(columns={'Adj Close' : 'Close'}, inplace=True)
# reverse row order because as you can see above data, it starts from recent date. 

amd = amd.iloc[::-1]



# use Date column as index and delete index name.

amd.set_index(['Date'], inplace=True)

amd.index.name=None



# plot

amd['Close'].plot()

plt.show()
# reverse row order 

google = google.iloc[::-1]



google.set_index(['Date'], inplace=True)

google.index.name=None



google['Close'].plot()

plt.show()

# make new columns; returns and log_returns



import math



amd['returns'] = amd['Close'] / amd['Open']

# amd['log_returns'] = amd['returns'].apply(lambda x: math.log10(x))





google['returns'] = google['Close'] / google['Open']

# google['log_returns'] = google['returns'].apply(lambda x: math.log10(x))





amd = amd[['Open', 'Close', 'Volume', 'returns',]]

google = google[['Open', 'Close', 'Volume', 'returns']]
# check what has changed; row order, columns

amd.head()
# to compare daily change of price with daily volume change use zip func.

# change = today volume / yesterday volume

# to calculate change, we need to transfer data type of Volume into float.

amd.Volume = amd.Volume.astype(float)

change = []

for a, b in zip(amd.Volume, amd.Volume[1:]):

    x = b/a

    change.append(x)
google.Volume = google.Volume.astype(float)

change_gl = []

for a, b in zip(google.Volume, google.Volume[1:]):

    x = b/a

    change_gl.append(x)
# changed

len(change)
# we will use 2000 data, so delete oldest data row 

# because we can't calculate volume change in 2009-05-22. we don't volume data on 2009-05-21.

amd.drop(amd.index[0], inplace=True)

#amd.head()



google.drop(google.index[0], inplace=True)

google.head()
# merge change into dataframe.

amd['vol_change'] = np.array(change)

google['vol_change'] = np.array(change_gl)
amd.describe()
# find if there is infinite value in dataframe

amd.loc[amd['vol_change']==np.inf]
amd.loc['2015-01-05']
amd.loc['2014-12-28':'2015-01-06']
# to get vol_change value at 2015-01-05, use 2014-12-31 volume cause 2015-01-02 has zero volume.

amd['vol_change']['2015-01-05'] = amd['Volume']['2015-01-05'] / amd['Volume']['2014-12-31']

amd.loc['2015-01-05']
amd.loc['2014-12-28':'2015-01-06']
amd.drop(['2015-01-02'], axis=0, inplace=True)

amd.shape
# amd.loc['2014-12-28':'2015-01-06']

amd.describe()
google.describe()
# divide data for cross validation.

from sklearn.model_selection import train_test_split



# select vol_change column as X, returns as y.

amd_X, amd_y = np.array(amd.iloc[:, 4]), np.array(amd.iloc[:,3])

# 80% of data for train, 20% for test

amd_X_train, amd_X_test, amd_y_train, amd_y_test = train_test_split(amd_X, amd_y,  test_size=0.2, random_state=0)

amd_X_train, amd_X_test = amd_X_train.reshape(-1, 1), amd_X_test.reshape(-1, 1)

amd_y_train, amd_y_test = amd_y_train.reshape(-1, 1), amd_y_test.reshape(-1, 1)



gl_X, gl_y = np.array(google.iloc[:, 4]), np.array(google.iloc[:,3])

gl_X_train, gl_X_test, gl_y_train, gl_y_test = train_test_split(gl_X, gl_y, test_size=0.2, random_state=0)

gl_X_train, gl_X_test = gl_X_train.reshape(-1, 1), gl_X_test.reshape(-1, 1)    

gl_y_train, gl_y_test = gl_y_train.reshape(-1, 1), gl_y_test.reshape(-1, 1)

from sklearn import linear_model



# fit linear model

regr1 = linear_model.LinearRegression()

amd_regr = regr1.fit(amd_X_train, amd_y_train)
regr2 = linear_model.LinearRegression()

gl_regr = regr2.fit(gl_X_train, gl_y_train)
print ("amd, coefficient: %.3f" %np.float(amd_regr.coef_))

print ("google, coefficient: %.3f" %np.float(gl_regr.coef_))
print ("Mean squared error of amd: %.3f" %np.mean((amd_regr.predict(amd_X_test) - amd_y_test)**2))

print ("Mean squared error of google: %.3f" %np.mean((gl_regr.predict(gl_X_test) - gl_y_test)**2))
# Explained variance score: 1 is perfect prediction

print('Variance score: %.3f' % amd_regr.score(amd_X_test, amd_y_test))

print('Variance score: %.3f' % gl_regr.score(gl_X_test, gl_y_test))

plt.scatter(amd_X_test, amd_y_test, color='black')

plt.plot(amd_X_test, amd_regr.predict(amd_X_test), color='blue', linewidth=3)

plt.xlabel('Volume')

plt.ylabel('Return')

plt.grid(True)



plt.show()

plt.scatter(gl_X_test, gl_y_test, color='black')

plt.plot(gl_X_test, gl_regr.predict(gl_X_test), color='blue', linewidth=3)

plt.xlabel('Volume')

plt.ylabel('Return')

plt.grid(True)



plt.show()
