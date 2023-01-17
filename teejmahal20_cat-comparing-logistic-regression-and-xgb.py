from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import LabelEncoder

from sklearn.feature_selection import SelectFromModel

from xgboost import XGBClassifier

from sklearn.metrics import roc_auc_score

import matplotlib.pyplot as plt

import seaborn as sns

import pandas as pd

import numpy as np 

import os



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# Pull data into dataframes

submission = pd.read_csv("/kaggle/input/cat-in-the-dat/sample_submission.csv")

train = pd.read_csv("/kaggle/input/cat-in-the-dat/train.csv")

test = pd.read_csv("/kaggle/input/cat-in-the-dat/test.csv")
# First look at training data

train.head()
# First look at testing data

test.head()
# Save 'id' for submission and drop it from test data

test_id = test["id"]

test.drop(['id'],inplace=True, axis = 1)

test.head()
# Prepare training data prior to encoding

y = train['target']

X = train.drop(['target', 'id'], axis = 1)

X.head()
# What are all of the catagorical columns?

cat_columns = [cols for cols in train.columns if train[cols].dtype == 'object']

print(cat_columns)
# Encode catagorical columns

encoder = LabelEncoder()

for col in cat_columns:

    X[col] = pd.DataFrame(encoder.fit_transform(X[col]))

    test[col] = pd.DataFrame(encoder.fit_transform(test[col]))   

# Check data to ensure correct encoding

X.head()
# Create train and test datasets from training data for model validation

Xtrain,Xtest,ytrain,ytest = train_test_split(X,y,random_state=0)
# Create, train, and test XGB model

model = XGBClassifier(n_estimators=500,scale_pos_weight=2,random_state=1,colsample_bytree=0.5)

model.fit(Xtrain,ytrain)

pred = model.predict_proba(Xtest)[:, 1]

score = roc_auc_score(ytest, pred)



print("score: ", score)
# Reset test and other variables to use for logistic regression

test = pd.read_csv("/kaggle/input/cat-in-the-dat/test.csv")

y2 = train['target']

X2 = train.drop(['target', 'id'], axis = 1)

test.drop(['id'],inplace=True, axis = 1)

test.head()
# Create dummies

traintest = pd.concat([X2, test])

dummies = pd.get_dummies(traintest, columns=traintest.columns, sparse=True)
# What do dummies look like?

X2 = dummies.iloc[:X2.shape[0], :]

test = dummies.iloc[X2.shape[0]:, :]



print(X2.shape)

print(test.shape)
# Speed up model

X2 = X2.sparse.to_coo().tocsr()

test = test.sparse.to_coo().tocsr()
# Create training and test data sets

Xtrain,Xtest,ytrain,ytest = train_test_split(X2,y,random_state=0)
# Create, train, and test Logistic Regression model

model = LogisticRegression(solver='lbfgs', C = 0.1, max_iter=1000)

model.fit(Xtrain,ytrain)

pred = model.predict_proba(Xtest)[:, 1]

score = roc_auc_score(ytest, pred)



print("score: ", score)
submission["id"] = test_id

submission["target"] = model.predict_proba(test)
submission.head()
submission.to_csv("submission.csv", index=False)