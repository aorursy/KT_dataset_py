#Import Libraries

import numpy as np

import pandas as pd

import xgboost as xgb

import os
# Importing the dataset

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
# Taking care of missing data



# Age

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan, strategy = 'mean')

imputer = imputer.fit(train.iloc[:, 5:6])

train.iloc[:, 5:6] = imputer.transform(train.iloc[:, 5:6])

imputer = imputer.fit(test.iloc[:, 4:5])

test.iloc[:, 4:5] = imputer.transform(test.iloc[:, 4:5])



# Embarked

from sklearn.impute import SimpleImputer

imputer_Emb = SimpleImputer(missing_values=np.nan, strategy='most_frequent')

imputer_Emb.fit(train.iloc[:, 11:12])

train.iloc[:, 11:12] = imputer_Emb.transform(train.iloc[:, 11:12])

imputer_Emb.fit(test.iloc[:, 10:11])

test.iloc[:, 10:11] = imputer_Emb.transform(test.iloc[:, 10:11])
# Tranforming Categorical in dummies

X_train = pd.get_dummies(train, prefix_sep="_", columns=["Pclass", "Sex", "Embarked"])

X_test = pd.get_dummies(test, prefix_sep="_", columns=["Pclass", "Sex", "Embarked"])
# Drop columns and prepare train, test datasets

y_train = np.array(X_train['Survived'])

X_train.drop(['Survived', 'Name', 'Ticket', 'Cabin'], inplace=True, axis=1)

X_test.drop(['Name', 'Ticket', 'Cabin'], inplace=True, axis=1)
# Estimate using XGBoost

model = xgb.XGBClassifier(n_estimators=2000, max_depth=5, learning_rate=0.1)

model.fit(X_train, y_train)

model_predictions = model.predict(X_test)
# Saving the CSV file 

csv = pd.read_csv('../input/test.csv')

csv.insert((csv.shape[1]),'Survived',model_predictions)

csv.drop(csv.columns.difference(['PassengerId','Survived']), 1, inplace=True)

csv.to_csv('../XGBoost_Model.csv', index = False)