import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns

import os

print(os.listdir("../input"))
train_set = pd.read_csv("../input/train/train.csv")

test_set = pd.read_csv("../input/test/test.csv")
train_set.info()
def check_unique(dataframe):

    for column in dataframe:

        print("{}: {} / {}".format(column, dataframe[column].nunique(), dataframe[column].notnull().sum()))
def export(y_pred):

    submission = pd.read_csv("../input/test/sample_submission.csv")

    for i,val in enumerate(y_pred):

        submission.at[i,'AdoptionSpeed'] = val

    submission["AdoptionSpeed"] = submission["AdoptionSpeed"].astype(int)

    submission.to_csv('submission.csv', index=False)
check_unique(train_set)
# Drop "Name" and "Description" column

train_set = train_set.drop(["Name", "Description", "PetID", "RescuerID"], axis=1)

test_set = test_set.drop(["Name", "Description", "PetID", "RescuerID"], axis=1)
train_set['ispure'] = (train_set['Breed2'] == 0).astype('int')

test_set['ispure'] = (test_set['Breed2'] == 0).astype('int')
train_x = train_set.drop("AdoptionSpeed", axis=1)

train_y = train_set["AdoptionSpeed"]
from sklearn.linear_model import LogisticRegression
LogReg_model = LogisticRegression()

LogReg_model.fit(train_x, train_y)
y_pred = LogReg_model.predict(test_set)
export(y_pred)
from sklearn import svm
svm_model = svm.SVC(gamma='scale')

svm_model.fit(train_x, train_y)
y_pred = svm_model.predict(test_set)
export(y_pred)
from xgboost import XGBClassifier
xgb_model = XGBClassifier()

xgb_model.fit(train_x, train_y)
y_pred = xgb_model.predict(test_set)
export(y_pred)