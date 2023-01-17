import numpy as np

import pandas as pd
ram_prices = pd.read_csv('../input/ram-prices/ram_price.csv')
ram_prices.head()
import matplotlib.pyplot as plt

%matplotlib inline

plt.figure(figsize=(15,10))

plt.semilogy(ram_prices.date,ram_prices.price)

plt.xlabel('Year')

plt.ylabel("Price in $/Mbyte")
from sklearn.tree import DecisionTreeRegressor

from sklearn.linear_model import LinearRegression

data_train = ram_prices[ram_prices.date < 2000]

data_test = ram_prices[ram_prices.date >= 2000]

X_train = data_train.date[:,np.newaxis]

y_train = np.log(data_train.price)

tree = DecisionTreeRegressor().fit(X_train,y_train)

linear_reg = LinearRegression().fit(X_train,y_train)

X_all = ram_prices.date[:,np.newaxis]

pred_tree = tree.predict(X_all)

pred_lr = linear_reg.predict(X_all)

price_tree = np.exp(pred_tree)

price_lr = np.exp(pred_lr)
plt.figure(figsize=(15,10))

plt.semilogy(data_train.date,data_train.price,label="Training Data",color='blue')

plt.semilogy(data_test.date,data_test.price,label="Test Data",color='red')

plt.semilogy(ram_prices.date,price_tree,label="Tree Prediction",color='green')

plt.semilogy(ram_prices.date,price_lr,label="Linear Prediction",color='yellow')

plt.legend()

plt.show()