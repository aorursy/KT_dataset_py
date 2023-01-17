import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.offline as py

from fbprophet import Prophet

import datetime as dt

import warnings

warnings.filterwarnings("ignore")



plt.style.use("seaborn-whitegrid")



df = pd.read_csv('../input/all_stocks_5yr.csv')

df = df.rename(columns={'Name': 'Ticks'})

googl = df.loc[df['Ticks'] == 'GOOGL']

googl_df = googl.copy()

googl_df.loc[:, 'date'] = pd.to_datetime(googl.loc[:,'date'], format="%Y/%m/%d")



f, (ax1, ax2) = plt.subplots(1, 2, figsize=(40,5))

ax1.plot(googl_df["date"], googl_df["close"])

ax1.set_xlabel("Date", fontsize=12)

ax1.set_ylabel("Stock Price")

ax1.set_title("Google Stock Price History")



f.delaxes(ax2)

plt.show()
ph_df = googl_df.drop(['open', 'high', 'low','volume', 'Ticks'], axis=1)

ph_df.rename(columns={'close': 'y', 'date': 'ds'}, inplace=True)



m = Prophet()

m.fit(ph_df)



future_prices = m.make_future_dataframe(periods=365)



forecast = m.predict(future_prices)



import matplotlib.dates as mdates



starting_date = dt.datetime(2018, 4, 7)

starting_date1 = mdates.date2num(starting_date)

trend_date = dt.datetime(2018, 6, 7)

trend_date1 = mdates.date2num(trend_date)



pointing_arrow = dt.datetime(2018, 2, 18)

pointing_arrow1 = mdates.date2num(pointing_arrow)



fig = m.plot(forecast)

ax1 = fig.add_subplot(111)

ax1.set_title("Google Stock Price Forecast", fontsize=16)

ax1.set_xlabel("Date", fontsize=12)

ax1.set_ylabel("Stock Price", fontsize=12)



ax1.annotate('Forecast Initialization\n(Data Available Till This Point)', xy=(pointing_arrow1, 1100), xytext=(starting_date1,1700),

            arrowprops=dict(facecolor='#ff7f50', shrink=0.1),

            )



ax1.annotate('Upward Trend\n(Prediction)', xy=(trend_date1, 1225), xytext=(trend_date1,950),

            arrowprops=dict(facecolor='#6cff6c', shrink=0.1),

            )



ax1.axhline(y=1110, color='c', linestyle='-')

plt.show()