!pip install yfinance # for getting stock data
import pandas as pd # load google trends data

import matplotlib.pyplot as plt # plot data
google_search_history = pd.read_csv('/kaggle/input/google_search_history.csv')

google_search_history['date'] = pd.DatetimeIndex(google_search_history['date'])

google_search_history = google_search_history.set_index('date')

google_search_history = google_search_history.astype(float)
import yfinance as yf # for downloading stock prices
aapl_prices = yf.download('AAPL') # get AAPL stock prices

aapl_search_history = google_search_history['AAPL']
aapl_prices.index = pd.DatetimeIndex(aapl_prices.index)
aapl_prices = aapl_prices[aapl_prices.index >= aapl_search_history.index.min()]
aapl_prices['Close'].plot()

plt.show()

plt.plot(aapl_search_history)

plt.show()
plt.plot(aapl_search_history)