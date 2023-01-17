# Kaggle's setup code

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#Machine learning packages sklearn
from sklearn.model_selection import train_test_split
from sklearn import linear_model
#loads Google's historical data
goog_data = pd.read_csv("../input/nasdaq-and-nyse-stocks-histories/full_history/GOOG.csv")

#loads Apple's historical data
index_symbol = "AAPL"
aapl_data = pd.read_csv("../input/nasdaq-and-nyse-stocks-histories/full_history/" + index_symbol + ".csv")

# Reverse order so most recent data must be predicted
aapl_data = aapl_data.sort_index(axis=0 ,ascending=False)

# One way to turn a string into the datetime datatype
# aapl_data['date'] = aapl_data["date"].astype('datetime64')

# This way is better because it deals with errors and allows you to specify the format
aapl_data["date"] = pd.to_datetime(aapl_data["date"], format = "%Y-%m-%d", errors = "coerce")

print(aapl_data.info()) #everything is numerical except for the date, which is now a datetime instead of a string
aapl_data.head(100)
# Good to check, if there are Nan values I need to deal with them before using a ml model
aapl_data.isnull().sum()
aapl_data["date"].describe(datetime_is_numeric=True)

# Orginally I used pd.to_datetime(), but because the data type was changed above its already a datetime
dates = aapl_data["date"]

# Changes size of graph: (width, height) | Must be called before plt.plot()
plt.figure(figsize=(20,7))

# Simple plot of apple's close price over its entire history
graph = plt.plot(dates, aapl_data["close"], label = "Apple Stock Close Price")
#graph = plt.plot(dates, aapl_data["open"], label = "Apple Stock Open Price")

graph = plt.plot(pd.to_datetime(goog_data["date"]), goog_data["close"], label = "Google Stock Close Price")

plt.legend(loc = 'upper left')
plt.show()
# Date values are  datetime data, needs to be numerical for the ML Model
# First I tried switching to ordinal data, but that doesn't allow the model to compare days of week, months of the year, etc.
# Pandas datetime actually has a lot of functionaliy that will allow me to easily seperate the data into year, month, week, day, etc.

from datetime import datetime as dt

# The year as an int
aapl_data["date_year"] = aapl_data["date"].dt.year

# Month values (1-12)
aapl_data["date_month"] = aapl_data["date"].dt.month

# Regular Series.dt.week is deprecated, thats why this one is slightly different.  (1-53)
aapl_data["date_week"] = aapl_data["date"].dt.isocalendar().week

# Day values for each month (1-31)
aapl_data["date_day"] = aapl_data["date"].dt.day

# Day values from 1 to 366 (if it's a leap year)
aapl_data["date_day_of_year"] = aapl_data["date"].dt.dayofyear

# Assigns days values of 0-6 for Monday-Sunday
aapl_data["date_day_of_week"] = aapl_data["date"].dt.dayofweek

# we now have pure numerical data to feed into the ML algorithim!
print(aapl_data.info())
aapl_data.describe()
# This is the label we are trying to predict
aapl_y = aapl_data["close"]


# For now I will only use the opening price, year, and day of the year (1-366)
aapl_X = aapl_data.drop(columns=["close", "date", "high", "low", "adjclose", "volume", "date_month", "date_week", "date_day", "date_day_of_week"])
# Split the data
X_train, X_test, y_train, y_test = train_test_split(aapl_X, aapl_y, test_size = .3, shuffle = False)

# The features in the training data
X_train.head()
# Simple stats on the label in the training data (close price)
y_train.describe()
#Intializes the model and fits it to the training data
model = linear_model.Lasso(alpha=.1)
model.fit(X_train, y_train)
# Predicts using the test data
y_pred = model.predict(X_test)
# Defines an evaluation function to display metrics on the model

import math
from sklearn.metrics import mean_squared_error, r2_score
def eval_metrics(y_test, y_pred):
    print("Mean Squared Error: %.3f" % mean_squared_error(y_test, y_pred))
    print("Root Mean Squared Error: %.3f" % math.sqrt(mean_squared_error(y_test, y_pred)))
    print("R Score: %.4f" % math.sqrt(abs(r2_score(y_test, y_pred))))
    print("R^2 Score: %.4f" % r2_score(y_test, y_pred))
#Cool way to turn the year and day of year back into a datetime value for graphing
aapl_X_test_graph_date = pd.to_datetime(X_test['date_year'] * 1000 + X_test['date_day_of_year'], format='%Y%j')
#Plot of the results

plt.figure(figsize=(20,12))


graph = plt.plot(aapl_X_test_graph_date, y_test, label = "Acutal Values")
graph = plt.scatter(aapl_X_test_graph_date, y_pred, s=1, c= "orange", label = "Predicted Values")
plt.legend()
#plt.margins(0,0)
plt.show()
eval_metrics(y_test,y_pred)
# determine correlation between each variable
# 'pearson' is the standard correlation coefficient method
goog_data.corr(method='pearson')

# the correlation between close and open is 0.999902, which is insane.  Acutally all the stock price data has extremely high correlations

plt.figure(figsize=(15,5))

#Calculate r values for open and close stock data
aapl_r = math.sqrt(abs(r2_score(aapl_data["open"], aapl_data["close"])))
goog_r = math.sqrt(abs(r2_score(goog_data["open"], goog_data["close"])))

print("\t\tApple R Score: %.4f \t\t\t\t\t\tGoogle R Score: %.4f" % (aapl_r, goog_r))

#Plots for Apple and Google, they show the very strong correlation

plt.subplot(1, 2, 1)
plt.scatter(aapl_data["open"], aapl_data["close"], label="Apple open price vs close price", s=1)
plt.title("Apple Open vs. Close Price")
plt.xlabel('Open Price')
plt.ylabel('Close Price')

plt.subplot(1, 2,2)
plt.scatter(goog_data["open"], goog_data["close"], color="orange", label="Google open price vs close price", s=1)
plt.title("Google Open vs. Close Price")
plt.xlabel('Open Price')
plt.ylabel('Close Price')
plt.show()

# I checked about 5 other stocks, they all have a really high corrleation between open price and close price
plt.figure(figsize=(20,7))

plt.plot(dates, aapl_data['close'], linewidth=1)
plt.scatter(dates, aapl_data['open'], s=1, c="orange")
plt.title("Open and Close Price plotted by date")
plt.show()
# Runs the model again on about 250 days so we can see a closer view of the predictions

# Split the data

X_train, X_test, y_train, y_test = train_test_split(aapl_X[9300:9556], aapl_y[9300:9556], test_size = .3, shuffle = False)

model = linear_model.Lasso(alpha=.1)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

#Plot of the results


aapl_X_test_graph_date = pd.to_datetime(X_test['date_year'] * 1000 + X_test['date_day_of_year'], format='%Y%j')

plt.figure(figsize=(20,8))



graph = plt.plot(aapl_X_test_graph_date, y_test, label = "Actual Values")
graph = plt.scatter(aapl_X_test_graph_date, y_pred, s=5, c= "orange", label = "Predicted Values")
plt.legend()
#plt.margins(0,0)
plt.show()
eval_metrics(y_test,y_pred)

# Runs the model again without any stock price as a feature

aapl_X = aapl_data.drop(columns=["close", "date", "high", "low", "adjclose", "open"])
aapl_y = aapl_data["close"]


# Split the data
X_train, X_test, y_train, y_test = train_test_split(aapl_X, aapl_y, test_size = .3, shuffle = True)


# Train the model and generate predictions
model = linear_model.Lasso(alpha=.1)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)


#Plot of the results
aapl_X_test_graph_date = pd.to_datetime(X_test['date_year'] * 1000 + X_test['date_day_of_year'], format='%Y%j')

plt.figure(figsize=(15,8))



graph = plt.scatter(aapl_X_test_graph_date, y_test, label = "Actual Values", s=2)
graph = plt.scatter(aapl_X_test_graph_date, y_pred, s=2, c= "orange", label = "Predicted Values")
plt.legend()
#plt.margins(0,0)
plt.show()
eval_metrics(y_test,y_pred)


