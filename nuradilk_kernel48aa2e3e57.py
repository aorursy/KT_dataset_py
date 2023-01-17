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
import numpy as np

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder

from sklearn.preprocessing import LabelEncoder



X = pd.read_csv('/kaggle/input/titanic/train.csv', index_col='PassengerId')

X_test = pd.read_csv('/kaggle/input/titanic/test.csv', index_col='PassengerId')

imp = SimpleImputer(missing_values=np.nan, strategy='mean')

enc = OneHotEncoder(handle_unknown='ignore', sparse=False)

enc_labe = LabelEncoder()

X.dropna(axis=0, subset=['Embarked'], inplace=True)

y = X.Survived

X.drop(['Survived', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

X_test.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace=True)



object_cols = [col for col in X.columns if X[col].dtype == "object" and col != "Cabin"]



X.Age.fillna(X.Age.mean(), inplace=True)



OH_cols_train = pd.DataFrame(enc.fit_transform(X[object_cols]))

OH_cols_test = pd.DataFrame(enc.transform(X_test[object_cols]))



OH_cols_train.index = X.index

OH_cols_test.index = X_test.index



X.drop(object_cols, axis=1, inplace=True)

X_test.drop(object_cols, axis=1, inplace=True)



X = pd.concat([X, OH_cols_train], axis=1)

X_test = pd.concat([X_test, OH_cols_test], axis=1)



# X.Cabin.fillna("nan", inplace=True)

# X_test.Cabin.fillna("nan", inplace=True)

# enc_labe.fit(list(X.Cabin) + list(X_test.Cabin))

# X.Cabin = enc_labe.transform(X.Cabin)

# X_test.Cabin = enc_labe.transform(X_test.Cabin)



# nanLabel = enc_labe.transform(["nan"])

# X.Cabin.replace(to_replace=nanLabel, value=np.nan, inplace=True)

# X_test.Cabin.replace(to_replace=nanLabel, value=np.nan, inplace=True)



# X.Cabin.fillna(round(X.Cabin.median()), inplace=True)

# X_test.Cabin.fillna(round(X.Cabin.median()), inplace=True)

X.Age = X.Age.round()

X
X_test.Age.fillna(X_test.Age.mean(), inplace=True)

X_test.Fare.fillna(X_test.Fare.mean(), inplace=True)

X_test.Age = X_test.Age.round()

X_test
from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(X, y,

                                                      train_size=0.8, test_size=0.2,

                                                      random_state=0)
from xgboost import XGBClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import mean_absolute_error

def score_dataset(X_train, X_valid, y_train, y_valid):

    model = DecisionTreeClassifier(random_state=0)

    model.fit(X_train, y_train)

    preds = model.predict(X_valid)

    return mean_absolute_error(y_valid, preds)
score_dataset(X_train, X_valid, y_train, y_valid)
from sklearn.ensemble import RandomForestClassifier



model = RandomForestClassifier(random_state=0, max_depth=10)

model.fit(X, y)


preds_test = model.predict(X_test)

# Save test predictions to file

output = pd.DataFrame({'PassengerId': X_test.index,

                       'Survived': preds_test})

output.to_csv('submission.csv', index=False)