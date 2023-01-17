# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OrdinalEncoder

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import RepeatedStratifiedKFold

from sklearn.ensemble import RandomForestClassifier



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# Load training data

train_data = pd.read_csv("/kaggle/input/titanic/train.csv")

train_data.head(10)
# Load testing data

test_data = pd.read_csv("/kaggle/input/titanic/test.csv")

test_data.head(10)
# Select the name of the label column

label_col = 'Survived'



# Select the feature columns

feature_cols = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']



# Select the categorical columns

cat_cols = ['Sex', 'Embarked']



# Get data types of each column

col_dtypes = train_data[feature_cols].dtypes

col_dtypes
# Make a copy of the training data and split out the label column

X = train_data[feature_cols].copy(deep=True)

X_test = test_data[feature_cols].copy(deep=True)

y = train_data[label_col].copy(deep=True)



# Check the shape of the new datasets

X.shape, y.shape
# Create an imputer instance and fit the dataset

simple_imputer = SimpleImputer(strategy='most_frequent')

X_imputed = simple_imputer.fit_transform(X)

X_test_imputed = simple_imputer.transform(X_test)

X = pd.DataFrame(X_imputed)

X_test = pd.DataFrame(X_test_imputed)

X_test
# Fix column names and data types

X.columns = feature_cols

X_test.columns = feature_cols



for i, dtype in enumerate(col_dtypes):

    X.iloc[:,i] = X.iloc[:,i].astype(dtype)

    X_test.iloc[:,i] = X_test.iloc[:,i].astype(dtype)

    

X_test
# Encode category columns

ordinal_encoder = OrdinalEncoder()

ordinal_encoder.fit(X[cat_cols])

X[cat_cols] = ordinal_encoder.transform(X[cat_cols])

X_test[cat_cols] = ordinal_encoder.transform(X_test[cat_cols])

X_test
# Define a cross-validation object

cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=4, random_state=0)
model = RandomForestClassifier(random_state=0, max_features=3, max_samples=0.3, n_estimators=1100)



n_scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')

print('Accuracy: %.3f (%.3f)' % (np.mean(n_scores), np.std(n_scores)))
model.fit(X, y)

predictions = model.predict(X_test)

# Generate output for submission

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})

output.to_csv('my_randomforest_submission.csv', index=False)

print("Your submission was successfully saved!")