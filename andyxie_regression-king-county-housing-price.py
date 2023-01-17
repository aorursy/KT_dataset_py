import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt



house = pd.read_csv("../input/kc_house_data.csv")
house.head()
house.info()
y = house["price"]

X = house.loc[:,"bedrooms":"sqft_lot15"]
y.shape, X.shape
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

X_train.shape, y_train.shape
from sklearn.linear_model import LinearRegression
model = LinearRegression()

model.fit(X_train, y_train)

model.score(X_test, y_test)
from sklearn.model_selection import cross_val_score

model = LinearRegression()

scores = cross_val_score(model, X, y, cv=5)

scores
np.mean(scores)
from sklearn.linear_model import Ridge

model = Ridge(alpha=0.1, normalize=True)

model.fit(X_train, y_train)

model.score(X_test, y_test)
coef = model.coef_

coef
col_names = house.drop(["id", "date", "price"], axis=1).columns

plt.figure(figsize=(10,10))

_ = plt.plot(range(len(col_names)), coef)

_ = plt.xticks(range(len(col_names)), col_names, rotation=60)

_ = plt.ylabel("Coefficients")

plt.show()