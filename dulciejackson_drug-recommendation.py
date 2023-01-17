# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Choose the best model

!pip install pycaret

from pycaret.classification import *



#open the dataset

drugs = pd.read_csv('../input/drug-classification/drug200.csv')



#define target label and parameters

exp1 = setup(drugs, target = 'Drug')



compare_models(fold = 5, turbo = True)
from sklearn.model_selection import train_test_split



X_full = pd.read_csv('../input/drug-classification/drug200.csv')



# Remove rows with missing target

X_full.dropna(axis=0, subset=['Drug'], inplace=True)



# Set target col as y, and remove from X_full

y = X_full.Drug

X_full.drop(['Drug'], axis=1, inplace=True)

X = X_full



X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,random_state=0)



print(X_train.head())
# All categorical columns

object_cols = [col for col in X_train.columns if X_train[col].dtype == "object"]



# Columns that can be safely label encoded

good_label_cols = [col for col in object_cols if 

                   set(X_train[col]) == set(X_valid[col])]



# Problematic columns that will be dropped from the dataset

bad_label_cols = list(set(object_cols)-set(good_label_cols))



print('Columns to label encode: ', good_label_cols)

print('Columns to remove from dataset: ', bad_label_cols)
from sklearn.preprocessing import LabelEncoder



# Drop categorical columns that will not be encoded

label_X_train = X_train.drop(bad_label_cols, axis=1)

label_X_valid = X_valid.drop(bad_label_cols, axis=1)



# Apply label encoder 

label_encoder = LabelEncoder()

for col in good_label_cols:

    label_X_train[col] = label_encoder.fit_transform(X_train[col])

    label_X_valid[col] = label_encoder.transform(X_valid[col])

    

# Encode labels from drugX, drugY etc to numbers

target_label_encoder = LabelEncoder()

label_y_train = label_encoder.fit_transform(y_train)

label_y_valid = label_encoder.transform(y_valid)
from sklearn.multiclass import OneVsRestClassifier

from xgboost.sklearn import XGBClassifier



model = OneVsRestClassifier(estimator=XGBClassifier(importance_type='gain',

                                            n_estimators=100, n_jobs=-1,

                                            objective='binary:logistic',

                                            random_state=6656,

                                            verbosity=0),n_jobs=-1)

from sklearn.metrics import mean_absolute_error, r2_score



# Fit the model to the training data

model.fit(label_X_train, label_y_train)

preds = model.predict(label_X_valid)

print("Mean Absolute Error: " + str(mean_absolute_error(label_y_valid, preds)))

print("R2 Score: " + str(r2_score(label_y_valid, preds)))