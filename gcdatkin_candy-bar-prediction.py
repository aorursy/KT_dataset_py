import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split



from sklearn.linear_model import LogisticRegression
data = pd.read_csv("../input/the-ultimate-halloween-candy-power-ranking/candy-data.csv")
data
names = data['competitorname']

data.drop('competitorname', axis=1, inplace=True)
plt.figure(figsize=(12, 10))

sns.heatmap(data.corr(), annot=True, vmin=-1, vmax=1)

plt.show()
data.isnull().sum()
y = data['bar']

X = data.drop('bar', axis=1)
scaler = MinMaxScaler()



X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
X
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7)
model = LogisticRegression()
model.fit(X_train, y_train)
print(f"Model Accuracy: {model.score(X_test, y_test)}")