# Importing Packages & Dependancies
import pandas as pd
import pandas_datareader.data as web
from pandas import Series, DataFrame
import numpy as np
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
import matplotlib as mpl
import datetime

# Uploading & Importing the MSFT Stock data
df = pd.read_csv("/kaggle/input/Microsoft-Stock/MSFT-stock.csv")
df.head (7)
#Creates two lists called dates and prices
dates = []
prices = []
#Prints the last row and column within the dataset. This is the price we will be predicting (167.770004)
df.tail(1)
#Creates a new dataset that includes all but the last row
df = df.head(len(df)-1)

#Counts and prints the number of rows and columns within the dataset
df.shape
#Gets all of the roads from the dates & open cloumn
df_dates = df.loc[:, 'Date']
df_open = df.loc[:, 'Open']
#Creates an dataset for dates and open prices
for date in df_dates:
  dates.append( [int(date.split('-')[2])])

for open_price in df_open:
  prices.append(float(open_price))
#Prints the prices that was recorded
print(prices)
#Prints the dates that was recorded
print(dates)
def predict_prices(dates, prices, x):

  #Creates 3 SVR models
  svr_lin = SVR(kernel='linear', C=1e3)
  svr_poly = SVR(kernel='poly', C=1e3)
  svr_rbf = SVR(kernel='rbf', C=1e3)

  #Trains the SVR models
  svr_lin.fit(dates, prices)
  svr_poly.fit(dates, prices)
  svr_rbf.fit(dates, prices)

  #Creates & Trains the Linear Regression Model
  lin_reg = LinearRegression()
  lin_reg.fit(dates, prices)

  #Plots all models and data onto a graph
  plt.scatter(dates, prices, color='black', label='Data')
  plt.plot(dates, svr_rbf.predict(dates), color='red', label='SVR RBF')
  plt.plot(dates, svr_poly.predict(dates), color='blue', label='SVR POLY')
  plt.plot(dates, svr_lin.predict(dates), color='green', label='SVR LINEAR')
  plt.plot(dates, lin_reg.predict(dates), color='orange', label='LINEAR REG')
  plt.xlabel('Days')
  plt.ylabel('Price')
  plt.title('Regression')
  plt.legend()
  plt.show()

  return svr_rbf.predict(x)[0], svr_lin.predict(x)[0], svr_poly.predict(x)[0], lin_reg.predict(x)[0]
predicted_price = predict_prices(dates, prices, [[24]])
print(predicted_price)
start = datetime.datetime(2010, 1, 1)
end = datetime.datetime(2020, 1, 1)
df = web.DataReader("MSFT", 'yahoo', start, end)
df.tail()
close_px = df['Adj Close']
mavg = close_px.rolling(window=100).mean()
# Adjusting the size of matplotlib
mpl.rc('figure', figsize=(8, 7))
mpl.__version__

# Adjusting the style of matplotlib
style.use('ggplot')

close_px.plot(label='MSFT')
mavg.plot(label='Moving Average')
plt.legend()