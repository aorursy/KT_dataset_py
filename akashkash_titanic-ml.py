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
# data preparation - downloading data, choosing which variables we want to play with, transforming char variables into numerical 

# and checking which columns contain NaN values

df = pd.read_csv("/kaggle/input/titanic/train.csv")

test = pd.read_csv("/kaggle/input/titanic/test.csv")

features = ['PassengerId', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Embarked']



x = pd.get_dummies(df[features])

y = df['Survived']



missing_val_count_by_column = (x.isnull().sum())

print(missing_val_count_by_column[missing_val_count_by_column > 0])

print(len(test))

# imputation - changing NaN values to mean values of the selected column

from sklearn.impute import SimpleImputer



# make copy to avoid changing original data (when Imputing)

x_copy = x.copy()

# Imputation

my_imputer = SimpleImputer()

new_data = pd.DataFrame(my_imputer.fit_transform(x_copy))

new_data.columns = x.columns

# x is a fully cleaned "df" variable

x = new_data
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split



# training the model only on training data split

x_train, x_test, y_train, y_test = train_test_split(x,y)



model = RandomForestClassifier(n_estimators=1000, random_state=0)

model.fit(x_train,y_train)

predictions = model.predict(x_test)

mae = mean_absolute_error(predictions, y_test)

# # training on full data (2 csv files)



model2 = RandomForestClassifier(n_estimators=1000, random_state=0)

model2.fit(x,y)



# import from test.csv only features which we chose before

test_features = ['PassengerId', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Embarked']

test_final = pd.get_dummies(test[test_features])



# here we go again - let's clean another dataset

# make copy to avoid changing original data (when Imputing)

test_final_copy = test_final.copy()

# Imputation

my_imputer = SimpleImputer()

new_data2 = pd.DataFrame(my_imputer.fit_transform(test_final_copy))

new_data2.columns = test_final.columns

# x is a fully cleaned "df" variable

test_final = new_data2



# finally we can predict our results

predictions2 = model2.predict(test_final)



output = pd.DataFrame({"PassengerId": test.PassengerId, "Survived": predictions2})

output.to_csv("Titanic.csv", index=False)