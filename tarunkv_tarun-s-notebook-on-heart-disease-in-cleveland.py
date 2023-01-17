# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
dataset = pd.read_csv("../input/cleveland-heart-disease-data-csv/Cleveland Heart Disease Data.csv")

dataset.head()
dataset.describe()
y = dataset['num']

X = dataset.drop('num', axis = 1)

X.shape
y.shape
from sklearn.model_selection import train_test_split



X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)



#Checking if it worked

print(X_train_full.shape)

print(X_test.shape)

print(y_train_full.shape)

print(y_test.shape)
# Creating some valid sets

X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full, test_size = 0.2, random_state = 0)



print(X_train.shape)

print(X_valid.shape)

print(y_train.shape)

print(y_valid.shape)
X_train.dtypes
numerical_columns = ['Age', 'trestbps','chol', 'thalach','oldpeak'] 

categorical_columns = ['Sex','CP','fbs','restecg','exang','slope','ca','thal']



cols_with_missing_train = [col for col in X_train.columns if X_train[col].isnull().any()]

print(cols_with_missing_train)

cols_with_missing_valid = [col for col in X_valid.columns if X_valid[col].isnull().any()]

print(cols_with_missing_valid)
for column in categorical_columns:

    print(X_train[column].value_counts())
X_train.ca.replace("?", np.NaN).value_counts()
X_train.thal.replace("?", np.NaN).value_counts()
from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder



#pipelines!

#creating a transformers for categories

numerical_transformer = SimpleImputer(strategy='constant')

categorical_transformer = Pipeline(steps=[

    ('imputer', SimpleImputer(strategy='most_frequent')),

    ('onehot', OneHotEncoder(handle_unknown='ignore'))

])



preprocessor = ColumnTransformer(

    transformers=[

        ('num', numerical_transformer, numerical_columns),

        ('cat', categorical_transformer, categorical_columns)

    ])
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error



def get_score(n_estimators):

    my_pipeline = Pipeline(steps=[

        ('preprocessor', preprocessor),

        ('model', RandomForestRegressor(n_estimators, random_state=0))

        ])

    my_pipeline.fit(X_train, y_train)

    preds_valid = my_pipeline.predict(X_valid)

    scores = mean_absolute_error(y_valid, preds_valid)

    return scores



MAE = {}



for num_leaves in range(100,151):

    MAE[num_leaves] = get_score(num_leaves)

    

MAE

my_pipeline = Pipeline(steps=[

        ('preprocessor', preprocessor),

        ('model', RandomForestRegressor(131, random_state=0))

        ])

my_pipeline.fit(X_train, y_train)



#Applying that to the test model. I actually have a y-value for test too.

preds_test = my_pipeline.predict(X_test)

score_test = mean_absolute_error(y_test, preds_test)

print("MAE:", score_test)



# Save test predictions to file

output = pd.DataFrame({'num': preds_test})

output.to_csv('submission.csv', index=False)