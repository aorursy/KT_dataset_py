# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
aapl_df = pd.read_csv("../input/nasdaq-and-nyse-stocks-histories/full_history/AAPL.csv")

aapl_df = aapl_df.sort_index(axis=0, ascending=False)

aapl_df["date"] = pd.to_datetime(aapl_df["date"])

aapl_df.head()
# previously used way to transfer strings into ints

from datetime import datetime as dt

dates=aapl_df["date"]



# The year as an int

aapl_df["date_year"] = dates.dt.year



# Month values (1-12)

aapl_df["date_month"] = dates.dt.month



# Regular Series.dt.week is deprecated, thats why this one is slightly different.  (1-53)

aapl_df["date_week"] = dates.dt.isocalendar().week



# Day values for each month (1-31)

aapl_df["date_day"] = dates.dt.day



# Day values from 1 to 366 (if it's a leap year)

aapl_df["date_day_of_year"] = dates.dt.dayofyear



# Assigns days values of 0-6 for Monday-Sunday

aapl_df["date_day_of_week"] = dates.dt.dayofweek



aapl_df.describe()
# I will skip normalization for now, I haven't gotten it to work with a model yet.

aapl_data = aapl_df.copy()

#aapl_data["volume"] /= 100000000

#aapl_data["open"] /= 100

#aapl_data["close"] /= 100





# delete columns we don't need (high,low is look ahead bias, adjclose is redundant, date is already represented with various int variables)

del(aapl_data["high"])

del(aapl_data["low"])

del(aapl_data["adjclose"])

del(aapl_data["date"])

aapl_data.describe()
# moving average of volume, and open price at 5 and 20 days

# iloc[: is whole column, int is column number]



#.rolling takes the last 'int' values as a range to calculate a stat

aapl_data['volume_5MA'] = aapl_data.iloc[:,0].rolling(window=5).mean()

aapl_data['volume_20MA'] = aapl_data.iloc[:,0].rolling(window=20).mean()



aapl_data['open_5MA'] = aapl_data.iloc[:,1].rolling(window=5).mean()

aapl_data['open_20MA'] = aapl_data.iloc[:,1].rolling(window=20).mean()

aapl_data.head()
# Now we have NaN values at the beginning.  Luckily its only the first 19 rows, so we can safely drop them

print(aapl_data.isna().sum())

aapl_data.dropna(inplace=True)

print(aapl_data.isna().sum())

aapl_data.head()
#split the data at 70%, 20%, and 10%

n = len(aapl_data)

# try model with less data

aapl_data = aapl_data[int(n*.4):]

n = len(aapl_data)



train_df = aapl_data[0:int(n*0.7)]

val_df = aapl_data[int(n*0.7):int(n*0.9)]

test_df = aapl_data[int(n*0.9):]

val_df.head()
#get labels for each data split

train_y = train_df["close"]

val_y = val_df["close"]

test_y = test_df["close"]





#get features for each data split, everything except for open and close price for now

train_X = train_df.drop(columns=['close', 'open'])

val_X = val_df.drop(columns=['close', 'open'])

test_X = test_df.drop(columns=['close', 'open'])





'''

train_X = train_df.drop(columns=['close'])

val_X = val_df.drop(columns=['close'])

test_X = test_df.drop(columns=['close'])'''

val_X.head()
from sklearn import linear_model



#Intializes the model and fits it to the training data

model = linear_model.Lasso(alpha=.1)

model.fit(train_X, train_y)

y_val_pred = model.predict(val_X)

y_test_pred = model.predict(test_X)



#function for evaluation metrics, I no longer use r or r^2 after last time

import math

from sklearn.metrics import mean_squared_error, r2_score

def eval_metrics(y_test, y_pred, dataset):

    print("\nEvaluation metrics for " + dataset + ":\n")

    print("Mean Squared Error: %.3f" % mean_squared_error(y_test, y_pred))

    print("Root Mean Squared Error: %.3f" % math.sqrt(mean_squared_error(y_test, y_pred)))

    print("R Score: %.4f" % math.sqrt(abs(r2_score(y_test, y_pred))))

    print("-----------------------------------------")



eval_metrics(val_y,y_val_pred, "Validation Data")

eval_metrics(test_y,y_test_pred, "Test Data")

import matplotlib.pyplot as plt



#converts back into datetime value for graphing

aapl_X_val_graph_date = pd.to_datetime(val_X['date_year'] * 1000 + val_X['date_day_of_year'], format='%Y%j')

aapl_X_test_graph_date = pd.to_datetime(test_X['date_year'] * 1000 + test_X['date_day_of_year'], format='%Y%j')



#Plot of the results

plt.figure(figsize=(20,12))





graph = plt.plot(aapl_X_val_graph_date, val_y, label = "Acutal Validation Values")

graph = plt.plot(aapl_X_val_graph_date, y_val_pred, label = "Predicted Validation Values")



graph = plt.plot(aapl_X_test_graph_date, test_y, label = "Acutal Test Values")

graph = plt.plot(aapl_X_test_graph_date, y_test_pred, label = "Predicted Test Values")





plt.legend()

#plt.margins(0,0)

plt.show()
# To better understand the MSE and RMSE, I calculated MSE by hand (RMSE is just the square root of MSE)

# Essentially RMSE is the mean of the absolute value of the differences between the predicted and actual stock price



df = pd.DataFrame(aapl_X_val_graph_date, columns=['date'])

df["acutal_close"] = val_y

df["pred_close"] = y_val_pred

df["residual"] = val_y-y_val_pred

df["residual square"] = df["residual"] * df["residual"]

print("This is the mean squared error for validation data: %.3f" % df["residual square"].mean())
df = pd.DataFrame(aapl_X_test_graph_date, columns=['date'])

df["acutal_close"] = test_y

df["pred_close"] = y_test_pred

df["residual"] = test_y-y_test_pred

df["residual square"] = df["residual"] * df["residual"]

print("This is the mean squared error for test data: %.3f" % df["residual square"].mean())


df = pd.DataFrame(val_df, columns=['open', 'close'])

df['residual'] = df['close'] - df['open']

df["residual square"] = df["residual"] * df["residual"]

print("This is the MSE for open vs. close stock price in val data: %.3f" % df["residual square"].mean())

print("This is the RMSE for open vs. close stock price in val data: %.3f" % math.sqrt(df["residual square"].mean()))



df = pd.DataFrame(test_df, columns=['open', 'close'])

df['residual'] = df['close'] - df['open']

df["residual square"] = df["residual"] * df["residual"]

print("\nThis is the MSE for open vs. close stock price in test data: %.3f" % df["residual square"].mean())

print("This is the RMSE for open vs. close stock price in test data: %.3f" % math.sqrt(df["residual square"].mean()))
