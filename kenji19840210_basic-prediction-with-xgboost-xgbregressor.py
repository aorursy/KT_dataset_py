import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

from sklearn.model_selection import StratifiedKFold

import xgboost as xgb

from sklearn import cross_validation

from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import accuracy_score

from sklearn.model_selection import RandomizedSearchCV,GridSearchCV





train = pd.read_csv("../input/Uniqlo(FastRetailing) 2012-2016 Training - stocks2012-2016.csv")

test=pd.read_csv('../input/Uniqlo(FastRetailing) 2017 Test - stocks2017.csv')

train_num = train.shape[0]

data = pd.concat([train,test])



def dummy_date(df):

    df["year"] = df["Date"].apply(lambda x: x.split("-")[0])

    df["month"] = df["Date"].apply(lambda x: x.split("-")[1])

    df["day"] = df["Date"].apply(lambda x: x.split("-")[2])

    df.drop("Date",inplace=True,axis=1)

    return df



def LabelEncord_categorical(df):

    categorical_params = ["year","month","day"]

    for params in categorical_params:

        le = LabelEncoder()

        df[params] = le.fit_transform(df[params])

    return df



def dummies(df):

    categorical_params = ["year","month","day"]

    for params in categorical_params:

        dummies =  pd.get_dummies(df[params])

        df = pd.concat([df, dummies],axis=1)

    return df



def pre_processing(df):

    df = dummy_date(df)

    df = LabelEncord_categorical(df)

#    df = dummies(df)

    return df



data = pre_processing(data)



train = data[:train_num]

test = data[train_num:]
y_train = train["Close"].values

X_train = train.drop("Close",axis=1).values

y_test = test["Close"].values

X_test =test.drop("Close",axis=1).values
gbm = xgb.XGBRegressor()

reg_cv = GridSearchCV(gbm, {"colsample_bytree":[1.0],"min_child_weight":[1.0,1.2]

                            ,'max_depth': [3,4,6], 'n_estimators': [500,1000]}, verbose=1)

reg_cv.fit(X_train,y_train)
reg_cv.best_params_
gbm = xgb.XGBRegressor(**reg_cv.best_params_)

gbm.fit(X_train,y_train)
predictions = gbm.predict(X_test)

predictions
gbm.score(X_test,y_test)
gbm.score(X_train,y_train)