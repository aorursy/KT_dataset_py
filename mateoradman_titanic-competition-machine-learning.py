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



train_data_path = '../input/titanic/train.csv'

test_data_path = '../input/titanic/test.csv'

train_data = pd.read_csv(train_data_path)

test_data = pd.read_csv(test_data_path)



# Any results you write to the current directory are saved as output.
train_data.head()
test_data.head()
test_data.describe()
train_data.isnull().sum()
from sklearn.model_selection import train_test_split



y = train_data.Survived

X = train_data.drop(['Survived', 'Age', 'Cabin'], axis = 1)

# Divide data into training and validation subsets

X_train_full, X_valid_full, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,

                                                                random_state=0)



# "Cardinality" means the number of unique values in a column

# Select categorical columns with relatively low cardinality (convenient but arbitrary)

categorical_cols = [cname for cname in X_train_full.columns if X_train_full[cname].nunique() < 10 and 

                        X_train_full[cname].dtype == "object"]



# Select numerical columns

numerical_cols = [cname for cname in X_train_full.columns if X_train_full[cname].dtype in ['int64', 'float64']]



# Keep selected columns only

my_cols = categorical_cols + numerical_cols

X_train = X_train_full[my_cols].copy()

X_valid = X_valid_full[my_cols].copy()
from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder



# Preprocessing for numerical data

numerical_transformer = SimpleImputer(strategy='constant')



# Preprocessing for categorical data

categorical_transformer = Pipeline(steps=[

    ('imputer', SimpleImputer(strategy='most_frequent')),

    ('onehot', OneHotEncoder(handle_unknown='ignore'))

])



# Bundle preprocessing for numerical and categorical data

preprocessor = ColumnTransformer(

    transformers=[

        ('num', numerical_transformer, numerical_cols),

        ('cat', categorical_transformer, categorical_cols)

    ])
from sklearn.metrics import mean_absolute_error

from xgboost import XGBClassifier



model = XGBClassifier(learning_rate=0.05,n_estimators=1000, n_thread = 4, objective='binary:logistic', early_stopping_rounds=5, 

             eval_set=[(X_valid, y_valid)],

             verbose=False)

# Bundle preprocessing and modeling code in a pipeline

my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),

                              ('model', model)

                             ])



# Preprocessing of training data, fit model 

my_pipeline.fit(X_train, y_train)



# Preprocessing of validation data, get predictions

preds = my_pipeline.predict(X_valid)



# Evaluate the model

score = mean_absolute_error(y_valid, preds)

print('MAE:', score)


# create test_X which comes from test_data but includes only the columns you used for prediction.

# The list of columns is stored in a variable called features

test_X = test_data[my_cols]



# make predictions which we will submit. 

test_preds = my_pipeline.predict(test_X)



# The lines below shows how to save predictions in format used for competition scoring

# Just uncomment them.



output = pd.DataFrame({'PassengerId': test_data.PassengerId,

                       'Survived': test_preds})

output.to_csv('submission.csv', index=False)


