import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split



from sklearn.linear_model import LinearRegression
data = pd.read_csv("../input/world-happiness/2019.csv")
data
data.drop(['Overall rank', 'Country or region'], axis=1, inplace=True)
plt.figure(figsize=(14, 12))

sns.heatmap(data.corr(), annot=True, vmin=-1.0, vmax=1.0)

plt.show()
data.isnull().sum()
data.dtypes
y = data['Score']

X = data.drop('Score', axis=1)
scaler = MinMaxScaler()



X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
X
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7)
model = LinearRegression()



model.fit(X_train, y_train)
print(f"Regression R2: {model.score(X_test, y_test)}")