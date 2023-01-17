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
train = pd.read_csv("/kaggle/input/health-insurance-cross-sell-prediction/train.csv")

test = pd.read_csv("/kaggle/input/health-insurance-cross-sell-prediction/test.csv")

sample_submission = pd.read_csv("/kaggle/input/health-insurance-cross-sell-prediction/sample_submission.csv")
train.head(5)
test.head(5)
sample_submission.head(5)
train.isnull().sum()
train.select_dtypes(include=["object"])
train["Vehicle_Age"].unique()
train.describe()
from sklearn.model_selection import train_test_split

train_X = train.drop(['Response'], axis=1)

train_y = train['Response']

X_train, X_test, y_train, y_test = train_test_split(train_X, train_y, random_state=0)
print(X_train.head())

print(y_train.head())
# cat_cols = X_train.select_dtypes(include=["object"])

# print(cat_cols)

# print(type(cat_cols))

# cat_cols = list(cat_cols)
# cat_cols_1h = pd.get_dummies(X_train[cat_cols])

# cat_cols_1h.describe()
# X_train = X_train.drop(cat_cols, axis=1)
# X_train = pd.concat([X_train, cat_cols_1h], axis=1, sort=False)

# X_train.tail(5)
def replace_categorical_fields_with_1h(X):

    cat_cols = X.select_dtypes(include=["object"])

    cat_cols = list(cat_cols)

    print(cat_cols)

    cat_cols_1h = pd.get_dummies(X[cat_cols])

    X = X.drop(cat_cols, axis=1)

    X = pd.concat([X, cat_cols_1h], axis=1, sort=False)

    return X
X_train = replace_categorical_fields_with_1h(X_train)

X_train.head()
from sklearn.ensemble import RandomForestClassifier

rf_classifier = RandomForestClassifier(max_depth=2, random_state=0)

rf_classifier.fit(X_train, y_train)
predictions = rf_classifier.predict(replace_categorical_fields_with_1h(X_test))
from sklearn.metrics import accuracy_score

accuracy_score(predictions, y_test)
real_predictions = rf_classifier.predict(replace_categorical_fields_with_1h(test))
submission = pd.DataFrame()
submission['id'] = test.id
submission['Response'] = real_predictions
submission.to_csv("submission.csv", index=False)