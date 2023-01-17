# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_score

from sklearn.feature_selection import SelectKBest, chi2 # For feature selection



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
!pwd
print("Loading data set......")

train = pd.read_csv("../input/learn-together/train.csv")

test = pd.read_csv("../input/learn-together/test.csv")

print("Done...")
print("train data size:", train.shape)

train.head()
print("test data size:", test.shape)

test.head()
train.info()
train.describe().T
# Declare target and predictors

print("Selecting features and target columns for model")

target = train['Cover_Type']

train_df = train.drop(["Cover_Type", "Id", "Vertical_Distance_To_Hydrology"], axis=1)

test_df = test.drop(["Id", "Vertical_Distance_To_Hydrology"], axis=1)

train_df.shape, test_df.shape
# Feature  selection



best = SelectKBest(chi2, k=25).fit(train_df, target)

train_best = best.transform(train_df)

test_best = best.transform(test_df)
# Create Model

print("Creating model")

rf = RandomForestClassifier(n_estimators=100)

print("Model created")

print("Cross vaidation Score")

print(cross_val_score(rf,train_best, target, cv=3, scoring="accuracy" ))

print("Fitting Model on training data set.....")

# Fit Model to traing data

rf.fit(train_best, target)

print("Predict on test data set....")

test_pred = rf.predict(test_best)



# Save test predictions to file

print("Creating submission file")

output = pd.DataFrame({'Id': test.Id,'Cover_Type': test_pred})

output.to_csv('submission_rf_1.csv', index=False)
