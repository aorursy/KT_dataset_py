1# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from matplotlib import style

import datetime as dt1

from datetime import datetime as dt







# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df=pd.read_csv('../input/prices.csv')

df.tail()

df=df.loc[df['symbol'] == 'CSCO']

print(df.tail())

from sklearn import preprocessing, cross_validation, svm

from sklearn.linear_model import LinearRegression

import math
forecast_col = 'close'

df.fillna(value=-99999, inplace=True)

forecast_out = int(math.ceil(0.01 * len(df)))

print(forecast_out)

df['label'] = df[forecast_col].shift(-forecast_out)

print(df.head())
#X = np.array(df.drop(['label'], 1))



X=np.array(df.drop(['label','symbol','date'], axis=1))

#print(X)

X = preprocessing.scale(X)

#print(X)

X_lately = X[-forecast_out:]

#print(X_lately)

X = X[:-forecast_out]

df.dropna(inplace=True)

y = np.array(df['label'])

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)
clf = LinearRegression(n_jobs=-1)

clf.fit(X_train, y_train)

confidence = clf.score(X_test, y_test)

print(confidence)
forecast_set = clf.predict(X_lately)

print(forecast_set)

df['Forecast'] = np.nan

last_date = df.iloc[-1].date

print(last_date)

last_date=dt.strptime(last_date, '%Y-%m-%d').timestamp()

last_unix = last_date

one_day = 86400

next_unix = last_unix + one_day



for i in forecast_set:

    next_date = dt.fromtimestamp(next_unix)

    next_unix += 86400

    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)]+[i]

print(df.tail())

df['close'].plot()

df['Forecast'].plot()

#plt.legend(loc=4)

plt.xlabel('Date')

plt.ylabel('Price')

plt.show()