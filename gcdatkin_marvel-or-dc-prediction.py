import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split



from sklearn.linear_model import LogisticRegression
data = pd.read_csv('../input/marvel-vs-dc/db.csv', encoding='latin-1')
data
data.info()
data.drop([data.columns[0], 'Original Title'], axis=1, inplace=True)
data
y = data['Company']

X = data.drop('Company', axis=1)
scaler = StandardScaler()



X = scaler.fit_transform(X)
encoder = LabelEncoder()



y = encoder.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7)
model = LogisticRegression()

model.fit(X_train, y_train)

print("Model Accuracy:", model.score(X_test, y_test))
data['Company'] = encoder.fit_transform(data['Company'])
data
corr = data.corr()



plt.figure(figsize=(12, 10))

sns.heatmap(corr, annot=True, vmin=-1, vmax=1)

plt.show()