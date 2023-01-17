!pip install nsepy


import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from nsepy import get_history # to download historical data

from datetime import date 

import matplotlib.pyplot as plt

%matplotlib inline

from sklearn.model_selection import train_test_split # to perform train_test split

from sklearn import linear_model # to perform Linear Regression

nifty = get_history(symbol='NIFTY 50',

                   start=date(2003,1,1),

                   end=date(2019,11,28),

                   index=True)
nifty.head(3)
nifty.shape
series = nifty["Close"]

series.plot()
maindata = pd.DataFrame()

maindata.shape
maindata[ "prev_high"] = nifty [ "High"].shift(1)

maindata[ "prev_low"] = nifty [ "Low" ].shift(1)

maindata[ "prev_Volume"] = nifty [ "Volume"].shift(1)

maindata ["diff_series1"] = series.shift(1)

maindata ["diff_series2"] = series.shift(2)

maindata ["diff_series3"] = series.shift(3)

maindata ["diff_series4"] = series.shift(4)

maindata ["diff_series5"] = series.shift(5)

maindata["diff"] = maindata["diff_series1"]- maindata["diff_series2"]

maindata["average"] = (maindata["prev_high"] + maindata["prev_low"])/2

maindata ["Close"] = series

maindata.head(5)
maindata = maindata.dropna()

maindata.head()
maindata.shape
X_close = maindata.drop("Close",1)

y_close = maindata["Close"]



X_close.head(2)
X_train, X_test, y_train, y_test = train_test_split(X_close, y_close, test_size=0.30)
regr = linear_model.LinearRegression()

regr.fit(X_train, y_train)
print(regr.coef_)
regr.score(X_test, y_test)
regr.predict([[12137.15	, 12023.70, 720945335,12048.20,12056.05, 12151.15, 12100.70, 12037.70,-7.80, 12089.40 ]])