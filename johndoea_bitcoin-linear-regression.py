import matplotlib.pyplot as plt

import pandas as pd

import numpy as np

from sklearn import datasets, linear_model

from sklearn.model_selection import cross_validate

from sklearn.model_selection import train_test_split
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
dados = pd.read_csv('/kaggle/input/bitcoin-historical-data/bitstampUSD_1-min_data_2012-01-01_to_2019-08-12.csv',',').dropna()

dados
# Attempt at using multiple columns (Scrapped as data doesn't correlate at all)

# x = np.array([dados['Open'], dados['High'], dados['Low'], dados['Close']])

# x = x[:, np.newaxis]

# x = x.reshape(x.shape[2], x.shape[1], x.shape[0])

# x = x[:,:,2]

# y = dados['Weighted_Price']

# [x.shape, y.shape]



x = dados[['Timestamp']]

y = dados['Weighted_Price']
x_train, x_test, y_train, y_test=train_test_split(x, y)

[x_train, x_test, y_train, y_test]
modelo=linear_model.LinearRegression()

modelo.fit(x_train, y_train)
print('Fórmula (a * x + b): {} * x + {}'.format(modelo.coef_[0], modelo.intercept_))

print('Qualidade: {}'.format(modelo.score(x_test,y_test)))

print('Desvio padrão: {}'.format(np.sqrt(np.mean((modelo.predict(x_test)-y_test)**2))))
plt.scatter(x_test, y_test, color='black')

plt.plot(x_test, modelo.predict(x_test), color='blue',linewidth=5)
w_test = np.array([[1325317920], [1325391360], [1385775000], [1393639980], [1407332460], [1412097660], [1439094960], [1511926260], [1523958420], [1565568000]])

#                   first input   1224th        1007618th     1138701th     1366909th     1446329th     1889812th     3103667th     3304203th     last input

y_pred = modelo.predict(w_test)

y_pred # ideal would be 4.390000, 4.580000, 1137.000000, 550.481835, 577.350000, 381.058418, 261.864553, 10196.695773, 8148.332643, 11540.450291