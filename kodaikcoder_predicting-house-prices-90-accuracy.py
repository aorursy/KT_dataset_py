import pandas as pd

import numpy as np

import matplotlib as plt

%matplotlib inline

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

import xgboost

from sklearn.metrics import accuracy_score

from sklearn.metrics import explained_variance_score

df =pd.read_csv('../input/kc_house_data.csv')
df.head(5)
data = pd.concat([df['sqft_living'],df['price']],axis = 1)

plot = data.plot.scatter(x = 'sqft_living',y = 'price')

plot.set_xlabel("Sqft Living")

plot.set_ylabel("House Price")

plot.axes.set_title("Sq ft Living area and House Prices")

feature_cols = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',

       'waterfront', 'view', 'condition', 'grade', 'sqft_above',

       'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode', 'lat', 'long',

       'sqft_living15', 'sqft_lot15']
X = df[feature_cols]
y = df['price']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=13)
linearModel = LinearRegression()
linearModel.fit(X_train,y_train)
linearModel.coef_
accuracy = linearModel.score(X_train,y_train)

"Accuracy on Train Data: {}%".format(int(round(accuracy * 100)))
accuracy = linearModel.score(X_test,y_test)

"Accuracy on Test Data: {}%".format(int(round(accuracy * 100)))
#Setting up XGBoost Parameters

xgb = xgboost.XGBRegressor(n_estimators=100, learning_rate=0.25, gamma=0, subsample=0.75,colsample_bytree=1, max_depth=5)
xgb.fit(X_train,y_train)
predictions = xgb.predict(X_test)

print(explained_variance_score(predictions,y_test))