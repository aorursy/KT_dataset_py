from sklearn.linear_model import LinearRegression



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
train_data = pd.read_csv('/kaggle/input/random-linear-regression/train.csv')

test_data = pd.read_csv('/kaggle/input/random-linear-regression/test.csv')
train_data.head()
train_data.info()
train_data = train_data.dropna()
train_data.describe()
plt.title('Distribuição da variavel X')

plt.hist(train_data['x'])

plt.show()
plt.title('Distribuição da variavel Y')

plt.hist(train_data['y'])

plt.show()
plt.title('Relação entre a variavel X e Y')

plt.scatter(x=train_data['x'], y=train_data['y'])

plt.show()
reg_linear = LinearRegression().fit(train_data[['x']], train_data['y'])
plt.title('Distribuição dos residuos da regressão')

plt.hist(reg_linear.predict(train_data[['x']]) - train_data['y'])

plt.show()
round(reg_linear.score(test_data[['x']], test_data['y']), 5)
np.sqrt(np.sum(np.power(reg_linear.predict(test_data[['x']]) - test_data['y'], 2)) / len(test_data))