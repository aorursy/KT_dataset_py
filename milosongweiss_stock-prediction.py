!pip install yfinance



import yfinance as yf # import stock data from yahoo finance

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # give me nice plots

import datetime as dt

from sklearn.linear_model import LinearRegression

from sklearn.tree import DecisionTreeRegressor

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_absolute_error
################################################################



# enter the abbreviation of the stock

stock_name = "AAPL"



# how many days into the future should the model predict?

pred_length = 60



################################################################



# get the data from the stock into a dataframe called stock_data

ticker = yf.Ticker(stock_name)

stock_data = ticker.history(period="max")



# plot the stock data

stock_data.plot(y="Close", figsize=(20,10))

plt.title(stock_name + " stock price")

plt.ylabel("Price (USD)")

plt.grid(True)

plt.xlim("2004-1-1", "2024-1-1")



# create new dataframe with only the close price as well as shifted close price

close_data = stock_data[["Close"]]

close_data["Prediction"] = close_data["Close"].shift(-pred_length)



close_data.head()
# create the feature and target arrays

X = np.array(close_data.drop(["Prediction"], 1)[:-pred_length])

y = np.array(close_data["Prediction"])[:-pred_length]



# split feature/target arrays into train and test data

train_X, test_X, train_y, test_y = train_test_split(X, y, random_state = 0)
# create and train the linear regressor model

model = LinearRegression()

model.fit(train_X, train_y)

print("Model score: ", model.score(test_X, test_y))

print()



predict_from = np.array(stock_data.drop(["Open", "High", "Low", "Volume", "Dividends", "Stock Splits"], 1)[-pred_length:])



print("Close data to make predictions from")

print(predict_from)

print()



predictions = model.predict(predict_from)

print("Linear Regression Predictions")

print(predictions)
current = stock_data.index[-1] # the last day we have stock data on

current += dt.timedelta(days=1) # our predictions start on the next day



# get an offset to add to each prediction so it lines up with the graph

last_price = stock_data.last("1D")["Close"]

first_price = predictions[0]

offset = last_price - first_price

print("The offset is: ", float(offset))



# add the predictions to the end of the stock_data dataframe

for prediction in predictions:

    # if the day is a weekend, skip over because trading doesn't happen on the weekends

    if current.weekday() == 5:

        current += dt.timedelta(days=2)

    if current.weekday() == 6:

        current += dt.timedelta(days=1)

    

    # add the prediction, with the offset

    stock_data.loc[current, "Prediction"] = float(prediction) + float(offset)

    

    # iterate the date

    current += dt.timedelta(days=1)
# graph the normal stock data and the prediction data

stock_data["Close"].plot(y="Close", figsize=(20,10))

stock_data["Prediction"].plot()

plt.title(stock_name + " stock price")

plt.ylabel("Price (USD)")

plt.grid(True)



view_start = dt.datetime.now() - dt.timedelta(days=pred_length*3)

view_end = dt.datetime.now() + dt.timedelta(days=pred_length*2)



# make the frame a good size

plt.xlim(view_start, view_end)



# make a vertical line indicating the current day

plt.axvline(x=dt.datetime.now(), linewidth=4, color="black", label="Today")