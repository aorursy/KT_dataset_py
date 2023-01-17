import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib as mpl

import matplotlib.pyplot as plt

import matplotlib.gridspec as gridspec
from sklearn.model_selection import train_test_split

# Reading Data

X = pd.read_csv('../input/titanic/train.csv')

X_test_full = pd.read_csv('../input/titanic/test.csv')



# Putting target variable in 'y' and dropping that column from X_train

y = X['Survived']

X = X.drop(['Survived'], axis=1)



# Breaking off X in train and validation datasets

X_train_full, X_valid_full, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size = 0.2, random_state = 0)



X.head()
fig = plt.figure(figsize=(15,8))

gs = fig.add_gridspec(2,2)
sns.set(style='white')

corr = X.corr()

mask = np.triu(np.ones_like(corr, dtype=np.bool))

f, ax = plt.subplots(figsize=(11, 9))

                             

sns.heatmap(corr, mask=mask, vmax=.3, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5})
# Number of missing values in each column of X_train_full

X_train_full.isnull().sum()
# Number of missing values in each column of X (train + valid)

X.isnull().sum()
from sklearn.impute import SimpleImputer

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import OneHotEncoder



# Droping columns from train and validation sets

drop_columns = ['Cabin', 'Name', 'Ticket']

X_train = X_train_full.drop(drop_columns, axis=1)

X_valid = X_valid_full.drop(drop_columns, axis=1)



# Creating the transformers

numeric_transformer = SimpleImputer(strategy = 'mean')

categoric_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy = 'most_frequent')),

                                        ('onehot', OneHotEncoder())])

categoric_cols = ['Sex', 'Embarked']



# Creating the pipeline with ColumnTransformer

preprocessor = ColumnTransformer(transformers=[('imputer_numeric',

                                               numeric_transformer,

                                               ['Age']),

                                              ('imputer_categoric',

                                              categoric_transformer,

                                              categoric_cols)])
from sklearn.ensemble import RandomForestClassifier

#from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score



# Defining Model

model = RandomForestClassifier(n_estimators=300)

#model2 = LogisticRegression()



# Bundle preprocessing and modeling code in a Pipeline

my_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])

my_pipeline.fit(X_train, y_train)

preds_val = my_pipeline.predict(X_valid)



# Accuracy of the prediction

accuracy_score(y_valid,preds_val)
X_test = X_test_full.drop(['Name', 'Cabin', 'Ticket'], axis=1)

preds_test = my_pipeline.predict(X_test)
# Save test predictions to file

output = pd.DataFrame({'PassengerID': X_test['PassengerId'],

    'Survived': preds_test})

output.set_index(['PassengerID'])

output.to_csv('submission.csv', index=False)