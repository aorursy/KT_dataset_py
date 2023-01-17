# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
    
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# Import training data
household_data_train_path = "../input/train.csv"
df_house_train = pd.read_csv(household_data_train_path)

# Data exploration
df_house_train.head()
df_house_train.info()
df_house_train.columns
df_house_train.describe()
df_house_train.describe(include='O')
# Define target variable
y = df_house_train.Target
# Define features
# Only numeric features has been taken for the model
household_features = df_house_train.drop('Target', axis= 1).select_dtypes(exclude='O').columns
X = df_house_train[household_features]
X.head()
# Check missing values in the feature
missing_values = [col for col in X.columns if X[col].isnull().any]
missing_values
# Imputing missing values
imputer = SimpleImputer()
X_imputed = imputer.fit_transform(X)
# Split into training and validation set
X_train, X_val, y_train, y_val = train_test_split(X_imputed, y, test_size= 0.33, random_state= 1)
# Define model
model = RandomForestClassifier(random_state= 1)

# Fit training data into model
model.fit(X_train, y_train)
# Predict validation set
y_pred = model.predict(X_val)
print(f1_score(y_val, y_pred, average=None))
print(f1_score(y_val, y_pred, average="macro"))
print(f1_score(y_val, y_pred, average="micro"))
print(f1_score(y_val, y_pred, average="weighted"))
# Import test data
household_data_test_path = "../input/test.csv"
df_house_test = pd.read_csv(household_data_test_path)
# Update test data set as per the training set
X_test = df_house_test[household_features]
X_test_imputed = imputer.transform(X_test)
# Predict using test data
y_pred_test = model.predict(X_test_imputed)
# Create output file
output = pd.DataFrame({"Id": df_house_test.Id,
                      "Target": y_pred_test})
output.to_csv("submission.csv", index= False)