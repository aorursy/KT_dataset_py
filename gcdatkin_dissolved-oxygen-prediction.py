import numpy as np

import pandas as pd



from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split



from sklearn.linear_model import LinearRegression, LogisticRegression
data = pd.read_csv('../input/dissolved-oxygen-prediction-in-river-water/train.csv')
data
data.info()
data.drop('Id', axis=1, inplace=True)
data.isna().sum()
null_columns = list(data.columns[data.isna().sum() > 100])



data.drop(null_columns, axis=1, inplace=True)
data
data.isna().sum()
print("Columns with missing values:", (data.isna().sum(axis=0) != 0).sum())

print("Rows with missing values:", (data.isna().sum(axis=1) != 0).sum())
data.dropna(axis=0, inplace=True)
data.isna().sum().sum()
data
y = data['target']

X = data.drop('target', axis=1)
scaler = StandardScaler()



X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7)
model = LinearRegression()

model.fit(X_train, y_train)

model_R2 = model.score(X_test, y_test)
print("Model R^2 Score:", model_R2)
y
y.mean()
y_new = pd.qcut(y, q=2, labels=[0, 1])
X_train, X_test, y_train, y_test = train_test_split(X, y_new, train_size=0.7)
model = LogisticRegression()

model.fit(X_train, y_train)

model_acc = model.score(X_test, y_test)
print("Model Accuracy:", model_acc)