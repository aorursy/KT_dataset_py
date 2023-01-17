# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import xgboost as xgb
train_df = pd.read_csv('../input/titanic/train.csv')

train_df.head()
test_df = pd.read_csv('../input/titanic/test.csv')

test_df.head()
train_df.info()
def process_df(df):

    # Delete unused columns.

    df = df.drop(["PassengerId", "Name", "Ticket", "Cabin", "Embarked"], axis=1)

    

    # Fill the missing values with the mean value.

    df["Age"] = df["Age"].fillna(df["Age"].mean())

    

    # Trainform Sex column string to number.

    df = df.replace("male", 0)

    df = df.replace("female", 1)

    return df



train_df = process_df(train_df)

test_df = process_df(test_df)

train_df.head()
train_X = train_df.drop(["Survived"], axis=1)

train_Y = train_df["Survived"]

print(train_X.head())

print()

print(train_Y.head())
# Model definition.

xgb_model = xgb.XGBClassifier(objective='binary:logistic')

# Model training.

# Dataframe.values --> numpy.ndarray

xgb_model.fit(train_X.values, train_Y.values)
test_X = test_df.values

test_Y = xgb_model.predict(test_X)
paId = pd.read_csv('../input/titanic/gender_submission.csv')['PassengerId']

sub = pd.DataFrame({'PassengerId':paId, 'Survived':test_Y})

print(sub)

sub.to_csv('./submission.csv', index=False)