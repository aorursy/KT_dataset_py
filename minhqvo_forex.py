import datetime

import pandas as pd

%matplotlib inline

import matplotlib.pyplot as plt

forex = pd.read_csv("../input/data-euro-usd.csv")

forex.info()
forex.head(5)
forex['Date'] = pd.to_datetime(forex['Date'])

forex.index = forex['Date']

del forex['Date']

forex.head(5)

plt.figure(figsize=(12, 6))

forex['Price'].plot()

plt.show()

plt.figure(figsize=(12, 6))

forex['Change %'].plot()

plt.show()

forex['Moving Average 50 Days'] = forex.Price.rolling(window=50).mean()







data = forex[['Price', 'Moving Average 50 Days']][-365:]
plots = data.plot(subplots=False,figsize=(12, 5))

plt.show()
forex['Moving Average 100 Days'] = forex.Price.rolling(window=100).mean()

data = forex[['Price', 'Moving Average 100 Days']][-365:]

plots = data.plot(subplots=True,figsize=(12, 5))

plt.show()
