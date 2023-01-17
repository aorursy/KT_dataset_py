import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
dataset = pd.read_csv('../input/sp500-stock-market-index/SP500_train.csv')
df = pd.read_csv('../input/sp500-stock-market-index/SP500_test.csv')
dataset.drop(["date","adj_close"],axis=1,inplace=True)
df.drop(["date","adj_close"],axis=1,inplace=True)
X_train = dataset.iloc[:, :-1].values
y_train = dataset.iloc[:, -1].values
X_test = df.iloc[:, :-1].values
y_test = df.iloc[:, -1].values
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))
from sklearn.model_selection import cross_val_predict
from sklearn import linear_model

fig, ax = plt.subplots()
ax.scatter(y_test, y_pred)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()