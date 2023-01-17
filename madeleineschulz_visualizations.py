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


original_cars = pd.read_csv("../input/original-cars/original cars.csv")
import pandas_datareader.data as web

import matplotlib.pyplot as plt

import datetime as dt
tickers = ["AAPL"]

start_date = "2017-01-01"

end_date = dt.datetime.now()

df = web.DataReader(tickers,data_source="yahoo", start=start_date, end=end_date)

df.head
df.plot()
df[["High", "Low"]].plot()
plt.style.available
plt.style.use('fivethirtyeight')

df[["High","Low"]].plot()
google = web.DataReader("GOOG", data_source="yahoo", start=start_date, end=end_date)

google["Close"].mean()
def rank_performance(stock_price):

    if stock_price <900:

        return "Poor"

    elif stock_price >900 and stock_price <= 1200:

        return "Average"

    elif stock_price >1200:

        return "Stellar"
google["Close"].apply(rank_performance).value_counts().plot(kind="bar")
original_cars.head()


original_cars.plot(kind="scatter", x="MPG", y="Horsepower")
size = original_cars[["Displacement"]]

original_cars[["Acceleration", "Horsepower"]].plot(kind="scatter", x="Acceleration",y="Horsepower",s=size,alpha=.3)
hist=original_cars.hist(column="MPG")
hist=original_cars.hist(figsize=(12,8))