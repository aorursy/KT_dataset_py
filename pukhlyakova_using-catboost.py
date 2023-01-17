import pandas as pd

import numpy as np

from catboost import CatBoostRegressor
train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')
train_df["Embarked"] = train_df["Embarked"].fillna("S")

train_df["Age"].fillna(train_df["Age"].mean(), inplace=True)

train_df["Fare"].fillna(train_df["Fare"].mean(), inplace=True)
X_train = train_df.drop(["PassengerId", "Survived", "Name", "Ticket", "Cabin"], axis=1)

Y_train = train_df["Survived"]

X_test  = test_df.drop(["PassengerId", "Name", "Ticket", "Cabin"], axis=1).copy()

print (X_train.tail())

print (X_test.tail())
model = CatBoostRegressor(learning_rate=1, depth=10, loss_function='RMSE')

cat_features = [1, 6]

fit_model = model.fit(X_train, Y_train, cat_features)

Y_test = fit_model.predict(X_test)