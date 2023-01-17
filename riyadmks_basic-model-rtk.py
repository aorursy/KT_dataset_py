# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split, GridSearchCV



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train = pd.read_csv("/kaggle/input/cat-in-the-dat-ii/train.csv")

test = pd.read_csv("/kaggle/input/cat-in-the-dat-ii/test.csv")
X = train.drop("target", axis=1)

y= train["target"]
numeric_features = X.select_dtypes(include=['float64']).columns

numeric_transformer = Pipeline(steps=[

    ('imputer', SimpleImputer(strategy='median')),

    ('scaler', StandardScaler())])
integer_features = X.select_dtypes(include=['int64']).columns

integer_transformer = Pipeline(steps=[

    ('imputer', SimpleImputer(strategy='median')),

    ('scaler', StandardScaler())])
categorical_features = X.select_dtypes(include=['object']).columns

categorical_transformer = Pipeline(steps=[

    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),

    ('onehot', OneHotEncoder(handle_unknown='ignore'))])
preprocessor = ColumnTransformer(

    transformers=[

        ('num', numeric_transformer, numeric_features),

        ('int', integer_transformer, integer_features),

        ('cat', categorical_transformer, categorical_features)])
# Append classifier to preprocessing pipeline.

# Now we have a full prediction pipeline.

clf = Pipeline(steps=[('preprocessor', preprocessor),

                      ('classifier', LogisticRegression())])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)



clf.fit(X_train, y_train)

print("model score: %.3f" % clf.score(X_test, y_test))
y_pred = clf.predict_proba(test)
sub = pd.DataFrame({'id': test['id'], 'target':y_pred[:,1]})
sub.to_csv('/kaggle/working/submission.csv', index=False)