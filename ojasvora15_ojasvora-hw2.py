import pandas as pd

import numpy as np

import scipy

import matplotlib.pyplot as plt

from sklearn import linear_model as ln

import os

print(os.listdir("../input"))
df = pd.read_csv("../input/tesla-stock-data-from-2010-to-2020/TSLA.csv")

df.head()
df.columns
df.tail()
#Checking for Null/Nan Values

df = df.dropna()

print("Check for NaN/Null values:\n", df.isnull().values.any())

print("Number of NaN/Null values:\n", df.isnull().sum())
# Simple Linear Regression

inputDF = df[["Open"]]

outcomeDF = df[["Close"]]

model = ln.LinearRegression()

results = model.fit(inputDF, outcomeDF)



print(model.intercept_, model.coef_)
# Result: Scikit - Learn

yp2 = model.predict(inputDF)

print(yp2)
# Compare the outcomes with the actual values of the data set.

print("Value in dataset:\n",df['Open'])

print("-------------------------------------------------------")

print ("Predicted Values:\n",yp2)

print("-------------------------------------------------------")

print ("Value in dataset:\n",df['Close'])

# Calculate the sum of squares of residuals for your model

df['Predicted 1'] = yp2

diff = df['Close'] - df['Predicted 1']

rss = np.sum(np.square(diff))

print("Sum of Squares of Residuals:\n", rss)
# Linear Regression representation using scatter plot

plt.title("Opening Prices vs Closing Prices daily",fontsize=16)

plt.scatter(inputDF,outcomeDF)

plt.plot(inputDF,yp2,color="red")

plt.show()
# Simple Linear Regression

inputDF = df[["High"]]

outcomeDF = df[["Close"]]

model = ln.LinearRegression()

results = model.fit(inputDF, outcomeDF)



print(model.intercept_, model.coef_)
# Result: Scikit - Learn

yp3 = model.predict(inputDF)

print(yp3)
# Compare the outcomes with the actual values of the data set.

print("Value in dataset:\n",df['High'])

print("-------------------------------------------------------")

print ("Predicted Values:\n",yp3)

print("-------------------------------------------------------")

print ("Value in dataset:\n",df['Close'])
# Calculate the sum of squares of residuals for your model

df['Predicted 2'] = yp3

diff = df['Close'] - df['Predicted 2']

rss = np.sum(np.square(diff))

print("Sum of Squares of Residuals:\n", rss)
# Linear Regression representation using scatter plot

plt.title("Highest Prices vs Closing Prices daily",fontsize=16)

plt.scatter(inputDF,outcomeDF)

plt.plot(inputDF,yp3,color="red")

plt.show()
# Simple Linear Regression

inputDF = df[["Low"]]

outcomeDF = df[["Close"]]

model = ln.LinearRegression()

results = model.fit(inputDF, outcomeDF)



print(model.intercept_, model.coef_)
# Result: Scikit - Learn

yp4 = model.predict(inputDF)

print(yp4)
# Compare the outcomes with the actual values of the data set.

print("Value in dataset:\n",df['Low'])

print("-------------------------------------------------------")

print ("Predicted Values:\n",yp4)

print("-------------------------------------------------------")

print ("Value in dataset:\n",df['Close'])

# Calculate the sum of squares of residuals for your model

df['Predicted 3'] = yp4

diff = df['Close'] - df['Predicted 3']

rss = np.sum(np.square(diff))

print("Sum of Squares of Residuals:\n", rss)
# Linear Regression representation using scatter plot

plt.title("Lowest Prices vs Closing Prices daily",fontsize=16)

plt.scatter(inputDF,outcomeDF)

plt.plot(inputDF,yp4,color="red")

plt.show()
import seaborn as sns
df.boxplot(column=['Open'])
df.boxplot(column=['Close'])
df.boxplot(column=['High'])
df.boxplot(column=['Low'])
df.boxplot(column=['Volume'])
x = df["Open"]

y = df["Close"]

plt.title("Opening Price v/s Closing Price",fontsize=16)

plt.scatter(x,y)

plt.show()
x = df["High"]

y = df["Low"]

plt.title("Highest Price That Day v/s Lowest Price That Day",fontsize=16)

plt.scatter(x,y)

plt.show()
x = df["High"]

y = df["Volume"]

plt.title("Highest Price That Day v/s Trading Volume",fontsize=16)

plt.scatter(x,y)

plt.show()
x = df["Low"]

y = df["Volume"]

plt.title("Lowest Price That Day v/s Trading Volume",fontsize=16)

plt.scatter(x,y)

plt.show()