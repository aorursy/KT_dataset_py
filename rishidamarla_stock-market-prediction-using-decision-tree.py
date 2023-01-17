# Importing all necessary libraries.

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt
# Using data from Apple's stock.

df = pd.read_csv('../input/sandp500/individual_stocks_5yr/individual_stocks_5yr/AAPL_data.csv') 
df.head()
df.info()
df.describe()
df.shape
# Visualizing the opening prices of the data.

plt.figure(figsize=(16,8))

plt.title('Apple')

plt.xlabel('Days')

plt.ylabel('Opening Price USD ($)')

plt.plot(df['open'])

plt.show()
# Visualizing the high prices of the data.

plt.figure(figsize=(16,8))

plt.title('Apple')

plt.xlabel('Days')

plt.ylabel('High Price USD ($)')

plt.plot(df['high'])

plt.show()
# Visualizing the low prices of the data.

plt.figure(figsize=(16,8))

plt.title('Apple')

plt.xlabel('Days')

plt.ylabel('Low Price USD ($)')

plt.plot(df['low'])

plt.show()
# Visualizing the closing prices of the data.

plt.figure(figsize=(16,8))

plt.title('Apple')

plt.xlabel('Days')

plt.ylabel('Closing Price USD ($)')

plt.plot(df['close'])

plt.show()
df2 = df['close']
df2.tail()
df2 = pd.DataFrame(df2)     
df2.tail()
# Prediction 100 days into the future.

future_days = 100

df2['Prediction'] = df2['close'].shift(-future_days)
df2.tail()
X = np.array(df2.drop(['Prediction'], 1))[:-future_days]

print(X)
y = np.array(df2['Prediction'])[:-future_days]

print(y)
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
from sklearn.tree import DecisionTreeRegressor

from sklearn.linear_model import LinearRegression
# Implementing Linear and Decision Tree Regression Algorithms.

tree = DecisionTreeRegressor().fit(x_train, y_train)

lr = LinearRegression().fit(x_train, y_train)
x_future = df2.drop(['Prediction'], 1)[:-future_days]

x_future = x_future.tail(future_days)

x_future = np.array(x_future)

x_future
tree_prediction = tree.predict(x_future)

print(tree_prediction)
lr_prediction = lr.predict(x_future)

print(lr_prediction)
predictions = tree_prediction 

valid = df2[X.shape[0]:]

valid['Predictions'] = predictions
plt.figure(figsize=(16,8))

plt.title("Model")

plt.xlabel('Days')

plt.ylabel('Close Price USD ($)')

plt.plot(df2['close'])

plt.plot(valid[['close', 'Predictions']])

plt.legend(["Original", "Valid", 'Predicted'])

plt.show()