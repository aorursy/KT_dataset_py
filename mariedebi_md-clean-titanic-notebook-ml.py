# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline



# Import sklearn modules

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder

from sklearn.metrics import mean_absolute_error

from sklearn.ensemble import RandomForestRegressor



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train_data = pd.read_csv("/kaggle/input/titanic/train.csv", index_col='PassengerId')

train_data.head()
train_data.describe()
test_data = pd.read_csv("/kaggle/input/titanic/test.csv", index_col='PassengerId')

test_data.head()
test_data.describe()
cols_with_missing = [col for col in train_data.columns if train_data[col].isnull().any()]

cols_with_missing
cat = (train_data.dtypes == 'object')

object_cols = list(cat[cat].index)

object_cols
test_cols_with_missing = [col for col in test_data.columns if test_data[col].isnull().any()]

test_cols_with_missing
test_cat = (test_data.dtypes == 'object')

test_object_cols = list(test_cat[test_cat].index)

test_object_cols
# Get number of unique entries in each column with categorical data

object_nunique = list(map(lambda col: train_data[col].nunique(), object_cols))

d = dict(zip(object_cols, object_nunique))



object_nunique_test = list(map(lambda col: test_data[col].nunique(), test_object_cols))

d_test = dict(zip(test_object_cols, object_nunique_test))



# Print number of unique entries by column, in ascending order

sorted(d.items(), key=lambda x: x[1])
sorted(d.items(), key=lambda x: x[1])
# Columns that will be one-hot encoded

low_cardinality_cols = [col for col in object_cols if train_data[col].nunique() < 10]



# Columns that will be dropped from the dataset

high_cardinality_cols = list(set(object_cols)-set(low_cardinality_cols))



print('Categorical columns that will be one-hot encoded:', low_cardinality_cols)

print('\nCategorical columns that will be dropped from the dataset:', high_cardinality_cols)
# All categorical columns

object_cols = [col for col in train_data.columns if train_data[col].dtype == "object"]



# Columns that can be safely label encoded

good_label_cols = [col for col in object_cols if 

                   set(train_data[col]) == set(test_data[col])]

        

# Problematic columns that will be dropped from the dataset

bad_label_cols = list(set(object_cols)-set(good_label_cols))

        

print('Categorical columns that will be label encoded:', good_label_cols)

print('\nCategorical columns that will be dropped from the dataset:', bad_label_cols)
import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder

from sklearn.ensemble import RandomForestRegressor

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import cross_val_score



X = train_data

X_test = test_data



# Remove rows with missing target, separate target from predictors

X.dropna(axis=0, subset=['Survived'], inplace=True)

y = X.Survived

X.drop(['Survived'], axis=1, inplace=True)



# Break off validation set from training data

X_train, X_valid, y_train, y_valid = train_test_split(X, y, 

                                                                train_size=0.8, test_size=0.2,

                                                                random_state=0)

cols_with_missing = [col for col in X.columns

                     if X[col].isnull().any()]



# "Cardinality" means the number of unique values in a column

# Select categorical columns with relatively low cardinality (convenient but arbitrary)

categorical_cols = [cname for cname in X_train.columns if

                    X_train[cname].nunique() < 10 and 

                    X_train[cname].dtype == "object"]



# Select numerical columns

numerical_cols = [cname for cname in X_train.columns if 

                X_train[cname].dtype in ['int64', 'float64']]



# # Keep selected columns only

# my_cols = categorical_cols + numerical_cols

# X_train = X_train[my_cols].copy()

# X_valid = X_valid[my_cols].copy()

# X_test = X_test[my_cols].copy()



# # One-hot encode the data (to shorten the code, we use pandas)

# X_train = pd.get_dummies(X_train)

# X_valid = pd.get_dummies(X_valid)

# X_test = pd.get_dummies(X_test)

# X_train, X_valid = X_train.align(X_valid, join='left', axis=1)

# X_train, X_test = X_train.align(X_test, join='left', axis=1)
X_train.head()
X_valid.head()
X_test.head()
from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder



# Preprocessing for numerical data

numerical_transformer = SimpleImputer(strategy='mean')



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
from sklearn.ensemble import RandomForestClassifier



model = RandomForestClassifier(n_estimators=100, criterion='mae', max_leaf_nodes=100, random_state=0)
from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import cross_val_score



# Bundle preprocessing and modeling code in a pipeline

#Define a get_score function to get best n_estimaror

def get_score(n_estimators, max_leaf_nodes):

    my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),

                                  ('model', RandomForestClassifier(n_estimators=n_estimators,max_leaf_nodes=max_leaf_nodes, random_state=0))

                                 ])

    scores = -1 * cross_val_score(my_pipeline, X, y,

                              cv=3,

                              scoring='neg_mean_absolute_error')

    return scores.mean()
#Create a dictionnary of results that will be filled with MAE given n_estimators

#Here estimators go from [50, 100, ..., 400]

k=100

results = {}

for i in range(1,9):

    for k in range(1,8):

        results[50*i, 50*k] = get_score(50*i,50*k)



results
import matplotlib.pyplot as plt

%matplotlib inline



plt.plot(list(results.keys()),list(results.values()))

plt.show()
min(results.values())
my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),

                                  ('model', RandomForestClassifier(n_estimators=400,max_leaf_nodes=50,random_state=0))

                                 ])



scores = -1 * cross_val_score(my_pipeline, X, y,

                          cv=3,

                          scoring='neg_mean_absolute_error')

    

# Preprocessing of training data, fit model 

my_pipeline.fit(X, y)



# # Preprocessing of validation data, get predictions

preds = my_pipeline.predict(X_test)



# # Evaluate the model

# score = mean_absolute_error(y_valid, preds)

# print('MAE:', score)

output = pd.DataFrame({'PassengerId': X_test.index,#ICI je devrais avoir X_test

                       'Survived': preds})

output.to_csv('submission_final_5.csv', index=False)
from xgboost import XGBRegressor





# # Keep selected columns only

# my_cols = categorical_cols + numerical_cols

# X_train = X_train[my_cols].copy()

# X_valid = X_valid[my_cols].copy()

# X_test = X_test[my_cols].copy()



# X_train = pd.get_dummies(X_train)

# X_valid = pd.get_dummies(X_valid)

# X_test = pd.get_dummies(X_test)

# X_train, X_valid = X_train.align(X_valid, join='left', axis=1)

# X_train, X_test = X_train.align(X_test, join='left', axis=1)





# Define the model

my_model_2 = XGBRegressor(n_estimators=300000, learning_rate=1.2)



# Fit the model

my_model_2.fit(X, y,

              early_stopping_rounds = 5,

              eval_set = [(X_valid, y_valid)],

              verbose=False)



# Get predictions

predictions_2 = my_model_2.predict(X_test)



# # Calculate MAE

# mae_2 = mean_absolute_error(y_valid, predictions_2)



# print("Mean Absolute Error:" , mae_2)
output = pd.DataFrame({'PassengerId': X_test.index,#ICI je devrais avoir X_test

                       'Survived': predictions_2})

output.to_csv('submission_final_XGBoost.csv', index=False)