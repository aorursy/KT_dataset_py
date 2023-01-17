# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

from sklearn.model_selection import train_test_split

%matplotlib inline
train = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-2/train.csv')

test = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-2/test.csv')

train = train.drop(['Province_State'], axis=1)

test = test.drop(['Province_State'], axis=1)

df = train

df['ConfirmedCases'] = np.log(df['ConfirmedCases'])

df['Fatalities'] = np.log(df['Fatalities'])

df.replace([np.inf, -np.inf], 0, inplace=True)

def d(x):

    return x[8:10]

def m(x):

    return x[5:7]

def y(x):

    return x[:4]



df['day'] = df['Date'].apply(d)

df['month'] = df['Date'].apply(m)

df['year'] = df['Date'].apply(y)

df.drop(['Date'], axis=1)

df['day'] = df['day'].astype(int)

df['month'] = df['month'].astype(int)

df['year'] = df['year'].astype(int)

cnt = df[df['Country_Region']=='Italy']
X_train = cnt[['Id', 'day', 'month', 'year']][:60]

Y_train = cnt['ConfirmedCases'][:60]

X_test = cnt[['Id', 'day', 'month', 'year']][60:]

Y_test = cnt['ConfirmedCases'][60:]

from sklearn.linear_model import LinearRegression

lr = LinearRegression()

lr.fit(X_train, Y_train)

Y_pred = lr.predict(X_test)

plt.plot(X_test['Id'], np.exp(Y_test), color = 'blue')

plt.plot(X_test['Id'], np.exp(Y_pred), color = 'red')

plt.show()
train = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-2/train.csv')

test = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-2/test.csv')

train = train.drop(['Province_State'], axis=1)

test = test.drop(['Province_State'], axis=1)

df = train

df['ConfirmedCases'] = np.sqrt(df['ConfirmedCases'])

df['Fatalities'] = np.sqrt(df['Fatalities'])

def d(x):

    return x[8:10]

def m(x):

    return x[5:7]

def y(x):

    return x[:4]



df['day'] = df['Date'].apply(d)

df['month'] = df['Date'].apply(m)

df['year'] = df['Date'].apply(y)

df.drop(['Date'], axis=1)

df['day'] = df['day'].astype(int)

df['month'] = df['month'].astype(int)

df['year'] = df['year'].astype(int)

cnt = df[df['Country_Region']=='Italy']
X_train = cnt[['Id', 'day', 'month', 'year']][:60]

Y_train = cnt['ConfirmedCases'][:60]

X_test = cnt[['Id', 'day', 'month', 'year']][60:]

Y_test = cnt['ConfirmedCases'][60:]

from sklearn.linear_model import LinearRegression

lr = LinearRegression()

lr.fit(X_train, Y_train)

Y_pred = lr.predict(X_test)

plt.plot(X_test['Id'], Y_test**2, color = 'blue')

plt.plot(X_test['Id'], Y_pred**2, color = 'red')

plt.show()
# gunakan transformasi log
train['Country_Region'].unique()
train = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-2/train.csv')

test = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-2/test.csv')

submission = test

train = train.drop(['Province_State'], axis=1)

test = test.drop(['Province_State'], axis=1)

df = train

df['ConfirmedCases'] = np.sqrt(df['ConfirmedCases'])

df['Fatalities'] = np.sqrt(df['Fatalities'])

def d(x):

    return x[8:10]

def m(x):

    return x[5:7]

def y(x):

    return x[:4]



df['day'] = df['Date'].apply(d)

df['month'] = df['Date'].apply(m)

df['year'] = df['Date'].apply(y)

df = df.drop(['Date'], axis=1)

df['day'] = df['day'].astype(int)

df['month'] = df['month'].astype(int)

df['year'] = df['year'].astype(int)



test['day'] = test['Date'].apply(d)

test['month'] = test['Date'].apply(m)

test['year'] = test['Date'].apply(y)

test = test.drop(['Date'], axis=1)

test['day'] = test['day'].astype(int)

test['month'] = test['month'].astype(int)

test['year'] = test['year'].astype(int)



test = test.drop(['ForecastId'], axis=1)

df = df.drop(['Id'], axis=1)
hasil = []

for country in train['Country_Region'].unique():

    train_country = df[df['Country_Region']==str(country)]

    test_country = test[test['Country_Region']==str(country)]

    test_country = test_country.drop(['Country_Region'], axis=1)

    train_country = train_country.drop(['Country_Region'], axis=1)

    X_train = train_country[['day', 'month', 'year']]

    Y_train = train_country['ConfirmedCases']

    lr = LinearRegression()

    lr.fit(X_train, Y_train)

    Y_pred = lr.predict(test_country)

    hasil.append(np.exp(Y_pred))
changes = []

for array in hasil:

    for i in array:

        changes.append(i)

submission['ConfirmedCases'] = changes
hasil = []

for country in train['Country_Region'].unique():

    train_country = df[df['Country_Region']==str(country)]

    test_country = test[test['Country_Region']==str(country)]

    test_country = test_country.drop(['Country_Region'], axis=1)

    train_country = train_country.drop(['Country_Region'], axis=1)

    X_train = train_country[['day', 'month', 'year']]

    Y_train = train_country['Fatalities']

    lr = LinearRegression()

    lr.fit(X_train, Y_train)

    Y_pred = lr.predict(test_country)

    hasil.append(np.exp(Y_pred))
changes = []

for array in hasil:

    for i in array:

        changes.append(i)

submission['Fatalities'] = changes
submission = submission.drop(['Province_State', 'Country_Region', 'Date'], axis=1)
submission
sample