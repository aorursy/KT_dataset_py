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
from sklearn.metrics import accuracy_score

from sklearn.preprocessing import LabelEncoder

from sklearn.impute import SimpleImputer

from sklearn.feature_selection import SelectKBest, f_classif

from sklearn.model_selection import train_test_split

import lightgbm as lgb

X = pd.read_csv('../input/titanic/train.csv', index_col="PassengerId")

test_data = pd.read_csv('../input/titanic/test.csv')
X.head()
test_data.Cabin = test_data.Cabin.fillna("Unknown")
test_data.head()
X.Cabin = X.Cabin.fillna("Unknown")
X.head()
X["Embarked"] = X["Embarked"].astype(str)

cat_features = ["Sex", "Embarked", "Name", "Ticket", "Cabin"]

encoder = LabelEncoder()

encoded_train = X[cat_features].apply(encoder.fit_transform)

encoded_test = test_data[cat_features].apply(encoder.fit_transform)





num_features = ["Survived","Pclass","Age","SibSp","Parch","Fare"]

test_features= ["Pclass","Age","SibSp","Parch","Fare"]



training_data = X[num_features].join(encoded_train)

test_data_joined = test_data[test_features].join(encoded_test)
print(training_data)
my_imputer = SimpleImputer()

imputed_X = pd.DataFrame(my_imputer.fit_transform(training_data))

imputed_X.columns = training_data.columns

imputed_test_data = pd.DataFrame(my_imputer.fit_transform(test_data_joined))

imputed_test_data.columns = test_data_joined.columns

feature_cols = imputed_X.columns.drop(["Name", "Ticket", "Cabin"])
print(imputed_X)
dropped_X = imputed_X.drop(["Name", "Ticket", "Cabin"], axis=1)

dropped_test = imputed_test_data.drop(["Name", "Ticket", "Cabin"], axis =1)
selector = SelectKBest(f_classif, k=6)



# Use the selector to retrieve the best features

X_new = selector.fit_transform(dropped_X[feature_cols], dropped_X['Survived']) 



# Get back the kept features as a DataFrame with dropped columns as all 0s

selected_features = pd.DataFrame(selector.inverse_transform(X_new),

                                index=dropped_X.index,

                                columns=feature_cols)



dropped_columns = selected_features.columns[selected_features.var() == 0]
print(dropped_columns)
dropped_X.drop(dropped_columns, axis=1)

dropped_test.drop(dropped_columns, axis =1)
y = dropped_X.Survived

X = dropped_X.drop(["Survived"], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.2, test_size =0.8, random_state=42)
train_data = lgb.Dataset(X_train, label=y_train)

valid_data = lgb.Dataset(X_test, label=y_test)
param = {'num_leaves': 31, 'objective': 'binary'}

param['metric'] = 'auc'
num_round = 500

bst = lgb.train(param, train_data, num_round, valid_sets=[valid_data], early_stopping_rounds=10)
#model.fit(X_train, y_train)
#y_pred = model.predict(X_test)
#acc = accuracy_score(y_pred, y_test)

#print(acc)
#full_X_train = pd.concat([X_train, X_test], axis = 0)

#full_y_train = pd.concat([y_train, y_test])
#model.fit(full_X_train, full_y_train)
pred = bst.predict(dropped_test).astype("int")
output = pd.DataFrame({"PassengerId": test_data.PassengerId, "Survived": pred})

output.to_csv("gender_submission.csv", index=False)

print("Your submission was sucessfully saved!")