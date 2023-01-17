%matplotlib inline

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
df = pd.read_csv("../input/tesla-stock-price/Tesla.csv - Tesla.csv.csv", index_col=0)
df.shape
df.columns
df.head()
# plt.figure(figsize=(12,6))

plt.plot(df.index, df['Close'])

plt.title("tesla stock price")

plt.ylabel("Price($)")

plt.show()
df['date'] = df.index
df.head()
import fbprophet
df = df.reset_index();
df.head()
df = df.loc[:, ["Date","Close"]]
df.head()
df = df.rename(columns={'Date':'ds', "Close":"y"})
df.head()
df.shape
train = df.iloc[0:1500, :]

test = df.iloc[1500:, :]
train.head()
test.head()
test.shape
prophet = fbprophet.Prophet(changepoint_prior_scale=0.05, n_changepoints=10)

prophet.fit(train)

pred_model = prophet.make_future_dataframe(periods=192, freq='D')

pred = prophet.predict(pred_model)

prophet.plot(pred)
prophet.plot_components(pred)