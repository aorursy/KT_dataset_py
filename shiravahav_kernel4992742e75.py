# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import sklearn as sk

from sklearn.linear_model import LogisticRegression



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

from sklearn.model_selection import train_test_split



# Read the data

X_full = pd.read_csv('../input/titanic/train.csv')

X_test_full = pd.read_csv('../input/titanic/test.csv')

X_gender_full = pd.read_csv('../input/titanic/gender_submission.csv')



# Remove rows with missing target, separate target from predictors

X_full.dropna(axis=0, subset=['Survived'], inplace=True)

y = X_full.Survived

X_full.drop(['Survived'], axis=1, inplace=True)



# To keep things simple, we'll use only numerical predictors

X = X_full.select_dtypes(exclude=['object'])

X_test = X_test_full.select_dtypes(exclude=['object'])

# Break off validation set from training data

X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,

                                                      random_state=0)
X_train.head()

X_train.shape


from sklearn.metrics import mean_absolute_error



# Function for comparing different approaches

def score_dataset(X_train, X_valid, y_train, y_valid):

    model = LogisticRegression(solver='liblinear', random_state=0)

    model.fit(X_train, y_train)

    preds = model.predict(X_valid)

    return mean_absolute_error(y_valid, preds)

from sklearn.impute import SimpleImputer



# Get names of columns with missing values

cols_with_missing = [col for col in X_train.columns

                     if X_train[col].isnull().any()]

# Make copy to avoid changing original data (when imputing)

X_train_plus = X_train.copy()

X_valid_plus = X_valid.copy()



# Make new columns indicating what will be imputed

for col in cols_with_missing:

    X_train_plus[col + '_was_missing'] = X_train_plus[col].isnull()

    X_valid_plus[col + '_was_missing'] = X_valid_plus[col].isnull()



# Imputation

my_imputer = SimpleImputer()

imputed_X_train_plus = pd.DataFrame(my_imputer.fit_transform(X_train_plus))

imputed_X_valid_plus = pd.DataFrame(my_imputer.transform(X_valid_plus))



# Imputation removed column names; put them back

imputed_X_train_plus.columns = X_train_plus.columns

imputed_X_valid_plus.columns = X_valid_plus.columns



print("MAE from Approach 3 (An Extension to Imputation):")

print(score_dataset(imputed_X_train_plus, imputed_X_valid_plus, y_train, y_valid))
# Preprocessed training and validation features

final_imputer = SimpleImputer(strategy='median')

final_X_train = pd.DataFrame(final_imputer.fit_transform(X_train))

final_X_valid = pd.DataFrame(final_imputer.transform(X_valid))

final_X_train.columns = X_train.columns

final_X_valid.columns = X_valid.columns



# Define and fit model

model =  LogisticRegression(solver='liblinear', random_state=0)

model.fit(final_X_train, y_train)



# Get validation predictions and MAE

preds_valid = model.predict(final_X_valid)

print("MAE (Your approach):")

print(mean_absolute_error(y_valid, preds_valid))
# Preprocess test data

final_X_test = pd.DataFrame(final_imputer.transform(X_test))



# Get test predictions

preds_test = model.predict(final_X_test)
# Save test predictions to file

output = pd.DataFrame({'PassengerId': X_gender_full.PassengerId,

                       'Survived': preds_test})

output.to_csv('submission.csv', index=False)