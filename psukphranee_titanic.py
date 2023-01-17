# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_data = pd.read_csv('/kaggle/input/titanic/train.csv')



test_data = pd.read_csv('/kaggle/input/titanic/test.csv')
train_data.head()
test_data.head()
train_data.info()
import matplotlib.pyplot as plt

train_data.hist(figsize=(20,15), bins=50)

plt.show()
#set index to passenger's ID

train_data.set_index('PassengerId', inplace=True)

test_data.set_index('PassengerId', inplace=True)
#separate features and labels

X_train, y_train = train_data.drop('Survived', axis='columns'), train_data['Survived']
#define custom transformer to drop unwanted columns

from sklearn.base import BaseEstimator, TransformerMixin



class DropColumns(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):

        return self

    def transform(self, X):

        X_lite = X.drop(['Ticket', 'Cabin', 'Name'], axis='columns')

        return X_lite



col_dropper = DropColumns()
#create an imputer object with a median strategy

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy="median")



#Create custom Pipeline 

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import OneHotEncoder

cat_encoder = OneHotEncoder()
num_pipeline = Pipeline([

    ('imputer', SimpleImputer(strategy='median')),

    ('std_scaler', StandardScaler())

])
#1. Drop unwanted columns

X_lite = col_dropper.transform(X_train)

X_lite.head()
#There are null-values in 'Embarked'. Get rid of them with dropna(). But first find out which rows they are to drop the corresponding labels.

null_emb_passid = X_lite.index[X_lite['Embarked'].isna()]

X_lite.drop(index=null_emb_passid, inplace=True)

y_lite = y_train.drop(index=null_emb_passid)

display(null_emb_passid)

display(y_lite)
from sklearn.compose import ColumnTransformer



#All columns

all_cols = list(X_train)

#Make a list of numerical columns to transform by imputing and scaling.

num_cols = ['Age', 'Fare']

#make list of categrorical data to OneHotEncode

cat_cols = ['Sex', 'Embarked']



full_pipeline = ColumnTransformer([

    ("num", num_pipeline, num_cols),

    ("one-hot", OneHotEncoder(), cat_cols)

])
X_lite.info()
X_lite_prepared = full_pipeline.fit_transform(X_lite)
from sklearn.ensemble import RandomForestClassifier

forest_clf = RandomForestClassifier(n_estimators=100, random_state=42)



forest_clf.fit(X_lite_prepared, y_lite)
test_data.head()

test_data.info()
test_data.index
#drop test data columsn

test_data_lite = col_dropper.transform(test_data)

test_data_lite.head()
test_data_lite_prepared = full_pipeline.fit_transform(test_data_lite)

test_data_lite_prepared
y_pred = sgd_clf.predict(test_data_lite_prepared)

y_pred
y_pred_rand_forest = forest_clf.predict(test_data_lite_prepared)
sum(abs(y_pred_rand_forest - y_pred))


output = pd.DataFrame({'PassengerId': test_data.index, 'Survived': y_pred_rand_forest})

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")
