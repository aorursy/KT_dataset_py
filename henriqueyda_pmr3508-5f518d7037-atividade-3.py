import numpy as np 
import pandas as pd 
import sklearn
import matplotlib.pyplot as plt
train = pd.read_csv("../input/train.csv")
train.head()
train.describe()
plt.scatter(train["longitude"],train["latitude"],c=train["median_house_value"],s=5)

Xtrain = train
Xtrain = Xtrain.drop(columns=["median_house_value"])
Xtrain.head()
Ytrain = train["median_house_value"]
from sklearn import linear_model
from sklearn.neighbors import KNeighborsRegressor
ridge = linear_model.Ridge(alpha = 0.5)
ridge.fit(Xtrain,Ytrain)
ridge.coef_
lasso = linear_model.Lasso(alpha = 0.1)
lasso.fit(Xtrain,Ytrain)
KNNRegression = KNeighborsRegressor(n_neighbors=52)
KNNRegression.fit(Xtrain, Ytrain) 
from sklearn.model_selection import cross_val_score
scoresRidge = cross_val_score(ridge, Xtrain, Ytrain, cv=10)
scoresRidge
scoresLasso = cross_val_score(lasso, Xtrain, Ytrain, cv=10)
scoresLasso
scoresKNN = cross_val_score(KNNRegression, Xtrain, Ytrain, cv=10)
scoresKNN
test = pd.read_csv("../input/test.csv")
test.head()
test_pred = ridge.predict(test)
for i in range(0,len(test_pred)):
    if test_pred[i] < 0:
        test_pred[i] = np.mean(test_pred)
pred = pd.DataFrame(test.Id)
pred["median_house_value"] = test_pred
pred.head()
pred.to_csv("prediction.csv", index=False)