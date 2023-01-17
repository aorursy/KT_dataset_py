import numpy as np

import os

import pandas as pd

import pickle

import xgboost as xgb

import sklearn.metrics

from sklearn.model_selection import train_test_split
def preprocess(df):

    dropped = df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'])

    X = pd.get_dummies(dropped.replace({'male':0, 'female':1}))

    return X



def load_data(filename):

    df = pd.read_csv(filename)

    X = preprocess(df.drop(columns=['Survived']))

    y = df.Survived

    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, random_state=32)

    return train_test_split(X, y, shuffle=True, random_state=32)



X_train, X_test, y_train, y_test = load_data('../input/train.csv')

dtrain = xgb.DMatrix(X_train, label=y_train)

dtest = xgb.DMatrix(X_test, label=y_test)
xgb_params = {

    'objective': 'binary:logistic',

    'eval_metric': 'logloss',

}

model = xgb.train(xgb_params, dtrain)



pickle.dump(model, open("XGBoost_model.pickle", "wb"))
model = pickle.load(open("XGBoost_model.pickle", "rb"))



y_pred_proba = model.predict(dtest)

y_pred = np.where(y_pred_proba > 0.5, 1, 0)
print(sklearn.metrics.classification_report(y_test, y_pred))
print(sklearn.metrics.confusion_matrix(y_test, y_pred))
df = pd.read_csv('../input/test.csv')

ids = df.PassengerId

X_submit = preprocess(df)

d_submit = xgb.DMatrix(X_submit)



y_submit_proba = model.predict(d_submit)

y_submit = np.where(y_submit_proba > 0.5, 1, 0)



df_submit = pd.DataFrame({'PassengerId': ids, 'Survived': y_submit})

df_submit.to_csv('submit.csv', index=False)