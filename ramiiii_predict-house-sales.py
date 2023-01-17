import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

import seaborn as sns
import os

print(os.listdir("../input"))
df = pd.read_csv('../input/kc_house_data.csv', index_col='date', parse_dates=True)
df.head()
df.info()
df.isnull().sum()
df.describe()
cols = []

for col in df.columns:

    if col == 'id' or col == 'price':

        continue

    cols.append(col)

print(cols)

len(cols)
fig, axis = plt.subplots(9, 2, figsize=(20, 40))



for i, ax in enumerate(axis.flat):

    df.pivot_table('price', index=cols[i]).plot(ax=ax)
X = df.drop(['id', 'price'], 1).values

y = df['price'].values
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor

from sklearn.linear_model import LinearRegression
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
rfr = RandomForestRegressor(n_estimators=100).fit(X_train, y_train)

lr = LinearRegression(normalize=True).fit(X_train, y_train)
rfr_pred = rfr.predict(X_test)

lr_pred = lr.predict(X_test)
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
print("Random Forest Regressor Model Accuracy Is: {0:.2f}".format(rfr.score(X_test, y_test)*100))

print("Linear Regression Model Accuracy Is: {0:.2f}".format(lr.score(X_test, y_test)*100))