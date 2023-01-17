import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.metrics import confusion_matrix
data = pd.read_csv("../input/carprices.csv")
data.shape
plt.scatter(data['Mileage'],data['Sell Price($)'])

plt.xlabel("Mileage")

plt.ylabel("Sell Price($)")
plt.scatter(data['Age(yrs)'],data['Sell Price($)'])

plt.xlabel("Age")

plt.ylabel("Sell Price($)")
X = data[['Mileage','Age(yrs)']]

y = data['Sell Price($)']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
model.score(X_test, y_test)