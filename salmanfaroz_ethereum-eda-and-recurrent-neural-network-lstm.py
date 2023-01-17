from pandas import read_csv, Grouper, DataFrame, concat

import matplotlib.pyplot as plt

from statsmodels.tsa.stattools import adfuller

import statsmodels.tsa.api as smt

import numpy as np

from sklearn.metrics import mean_squared_error

import seaborn as sns

from datetime import datetime

import pandas as pd

from pandas_profiling import ProfileReport

from sklearn.preprocessing import MinMaxScaler

from keras.preprocessing import sequence 

from keras.models import Sequential 

from keras.layers import Dense, Embedding ,Dropout

from keras.layers import LSTM 

from plotly.offline import plot, iplot, init_notebook_mode

import plotly.graph_objs as go

init_notebook_mode(connected=True)
df=pd.read_csv("../input/ethereum-historical-data/Ethereum Historical Data.csv")
df["Vol."]=df["Vol."].str.replace("-","0")

df["Vol."] = (df["Vol."].replace(r'[KMB]+$', '', regex=True).astype(float) * df["Vol."].str.extract(r'[\d\.]+([KMB]+)', expand=False).fillna(1).replace(['K','M', 'B'], [10**3, 10**6, 10**9]).astype(int))

df["Price"]=df["Price"].str.replace(",","")

df["Price"]=df["Price"].astype("float")

df["Open"]=df["Open"].str.replace(",","")

df["Open"]=df["Open"].astype("float")

df["High"]=df["High"].str.replace(",","")

df["High"]=df["High"].astype("float")

df["Low"]=df["Low"].str.replace(",","")

df["Low"]=df["Low"].astype("float")

df["Change %"]=df["Change %"].str.replace("%","")

df["Change %"]=df["Change %"].astype("float")
df['Date']= pd.to_datetime(df['Date'], format="%b %d, %Y")

df = df.sort_values(by='Date', ascending=True)

df.index = df.Date

del df["Date"]

sns.set(rc={'figure.figsize':(7,5)})

sns.barplot(x=df.index.year, y=df["Price"], data=df)

plt.xlabel("Year", labelpad=17)

plt.ylabel("Price", labelpad=14)



plt.title("Price by year ", y=1.01);
cols_plot = ["Price","Open","High","Low"]

axes = df[cols_plot].plot(marker='.', alpha=0.5, linestyle='None', figsize=(11, 9), subplots=True)

for ax in axes:

    ax.set_ylabel('Monthly Totals')
sns.boxplot(data=df, x=df.index.year, y=df["Change %"])

plt.title("Year wise report on Change %")
sns.set(rc={'figure.figsize':(15,5)})

start, end = '2016', '2020'

plt.plot(df.loc[start:end, 'Vol.'],

marker='.', linestyle='-', linewidth=0.5, label='Daily')

plt.title("Volume by year ", y=1.01);
data=df["Price"]



feature_ts_train_diff = data.diff(periods=1)

feature_ts_train_diff.dropna(inplace=True)



feature_ts_train_diff



fig, axes = plt.subplots(1, 2)

fig.set_figwidth(12)

fig.set_figheight(4)

plt.xticks(range(0,30,1), rotation = 90)

smt.graphics.plot_acf(feature_ts_train_diff, lags=30, ax=axes[0])

smt.graphics.plot_pacf(feature_ts_train_diff, lags=30, ax=axes[1])

plt.tight_layout()
dftest = adfuller(data)

print("Statistics",dftest[0])

fig = go.Figure(go.Indicator(

    mode = "gauge+number",

    value = dftest[1],

    title = {'text': "P value"},

    domain = {'x': [0, 1], 'y': [0, 1]}

))





fig.show()
ethe_train = data.iloc[0:1260].values

ethe_test = data.iloc[1260:].values



ethe_train=ethe_train.reshape(-1, 1)

scaler = MinMaxScaler(feature_range = (0, 1))



ethe_training_scaled = scaler.fit_transform(ethe_train)



ethe_training_scaled



features_set = []

labels = []

for i in range(60, 1260):

    features_set.append(ethe_training_scaled[i-60:i, 0])

    labels.append(ethe_training_scaled[i, 0])



features_set, labels = np.array(features_set), np.array(labels)





features_set = np.reshape(features_set, (features_set.shape[0], features_set.shape[1], 1))
model = Sequential()





model.add(LSTM(units=50, return_sequences=True, input_shape=(features_set.shape[1], 1)))

model.add(Dropout(0.2))

model.add(LSTM(units=50, return_sequences=True))

model.add(Dropout(0.2))



model.add(LSTM(units=50, return_sequences=True))

model.add(Dropout(0.2))



model.add(LSTM(units=50))

model.add(Dropout(0.2))

model.add(Dense(units = 1))

model.compile(optimizer = 'adam', loss = 'mean_squared_error')

model.fit(features_set, labels, epochs = 50, batch_size = 32)

test_inputs = data[len(data) - len(ethe_test) - 60:].values





test_inputs = test_inputs.reshape(-1,1)

test_inputs = scaler.transform(test_inputs)



test_features = []

for i in range(60, 400):

    test_features.append(test_inputs[i-60:i, 0])



test_features = np.array(test_features)

test_features = np.reshape(test_features, (test_features.shape[0], test_features.shape[1], 1))



predictions = model.predict(test_features)

predictions = scaler.inverse_transform(predictions)





plt.figure(figsize=(10,6))

plt.plot(ethe_test, color='blue', label='Actual Ethereum Stock Price')

plt.plot(predictions , color='red', label='Predicted Ethereum Stock Price')

plt.title('Ethereum Stock Price Prediction')

plt.xlabel('Date')

plt.ylabel('Ethereum Stock Price')

plt.ylim(0,500)

plt.legend()

plt.show()