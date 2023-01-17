import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

from matplotlib.pylab import rcParams

rcParams["figure.figsize"] = 20,10

from sklearn import neighbors

from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0, 1))

from sklearn.tree import DecisionTreeRegressor

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

import seaborn as sns

import yfinance as yf

msft = yf.Ticker("MSFT")





tickers = ["NZD"]

data = yf.download(tickers, start="2020-08-21", end="2020-09-03", interval="1D")

data.to_csv('aapl.csv')

df = pd.read_csv('aapl.csv')



future_days = 4



df = df[["Close"]]

df["Prediction"] = df[['Close']].shift(-future_days)

X = np.array(df.drop(["Prediction"], 1))[:-future_days]

y = np.array(df["Prediction"])[:-future_days]

x_train, x_test, y_train, t_test = train_test_split(X, y, test_size = 0.25)

tree = DecisionTreeRegressor().fit(x_train, y_train)

lr = LinearRegression().fit(x_train, y_train)

x_future = df.drop(['Prediction'], 1)[:-future_days]

#Get the last 'x' rows

x_future = x_future.tail(future_days) 

#Convert the data set into a numpy array

x_future = np.array(x_future)

tree_prediction = tree.predict(x_future)

print( tree_prediction )

print()

#Show the model linear regression prediction

lr_prediction = lr.predict(x_future)

#Visualize the data

predictions = tree_prediction

#Plot the data

sns.set()

valid =  df[X.shape[0]:]

valid['Predictions'] = predictions #Create a new column called 'Predictions' that will hold the predicted prices

plt.figure(figsize=(18,10))

plt.title('Model')

plt.xlabel('Days',fontsize=18)

plt.ylabel('Close Price USD ($)',fontsize=18)

plt.plot(df['Close'])

plt.plot(valid[['Close','Predictions']])

plt.legend(['Train', 'Val', 'Prediction' ])

plt.show()
!pip install yfinance