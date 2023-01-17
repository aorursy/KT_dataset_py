import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt
data = pd.read_csv('../input/BostonHousing.csv')
data.head()
data.info()
plt.figure(figsize=(15,8))

sns.heatmap(data.corr(),annot=True)
data.describe()
x = data.iloc[:, :-1].values

y = data.iloc[:, 13].values
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 0)
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

regressor.fit(x_train, y_train)
y_pred = regressor.predict(x_test)
y_pred
regressor.coef_


plt.scatter(y_test,y_pred)

plt.xlabel('Y Test')

plt.ylabel('Predictions')
sns.distplot((y_test-y_pred),bins=50)
np.sqrt(np.mean((y_pred-y_test)**2))