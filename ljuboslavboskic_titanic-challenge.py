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
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder

from sklearn.ensemble import RandomForestClassifier

from sklearn.impute import SimpleImputer



## Reading Training/Testing Data

file_path_Training= '../input/titanic/train.csv'

file_path_Testing= '../input/titanic/test.csv'



data_train = pd.read_csv(file_path_Training)

data_test = pd.read_csv(file_path_Testing)
# Drop data which isn't used

data_train=data_train.drop("PassengerId",axis=1)

data_train=data_train.drop("Name",axis=1)

data_train=data_train.drop("Ticket",axis=1)

data_train=data_train.drop("Cabin",axis=1)



data_test=data_test.drop("Name",axis=1)

data_test=data_test.drop("Ticket",axis=1)

data_test=data_test.drop("Cabin",axis=1)

data = [data_train, data_test]



for ds in data:

    mean_age_train = data_train['Age'].mean()

    std_age_test = data_test['Age'].std()

    is_null = ds["Age"].isnull().sum()

    upper_bound_age = mean_age_train + std_age_test

    lower_bound_age = mean_age_train - std_age_test

    rand_age = np.random.randint(lower_bound_age, upper_bound_age, size = is_null)

    # fill NaN values in Age column with random values generated

    age_slice = ds["Age"].copy()

    age_slice[np.isnan(age_slice)] = rand_age

    ds["Age"] = age_slice

    ds["Age"] = data_train["Age"].astype(int)

data_train["Age"].isnull().sum()
# Check for NaN values

import pandas_profiling as pp

# Good way to visualize where we missing data

# pp.ProfileReport(data_train, title = 'Pandas Profiling report of "Train" set', html = {'style':{'full_width': True}})

for col in data_train.columns:

    isNull_train = data_train[col].isnull().values.any()

    if isNull_train == True:

        print("Training Data in column name {0} has null values.".format(col))

        data_train_Null = data_train[col]

for col in data_test.columns:

    isNull_test = data_test[col].isnull().values.any()

    if isNull_test == True:

        print("Testing Data in column name {0} has null values.".format(col)) 

        data_test_Null = data_test[col]
# Found NaN values, now need to fill entries

# Training Data (filling in NaN with string)

print('Starting Null value filling for training set')

train_NAN_count = data_train_Null.value_counts()

print(train_NAN_count)

value_max_train = train_NAN_count[train_NAN_count.eq(train_NAN_count.max())].index.tolist()



print('Filling in NaN with most common value.')

data_train["Embarked"] = data_train["Embarked"].fillna(str(value_max_train))
# Testing Data (filling in NaN with int)

print('Starting Null value filling for testing set')

mean_val = data_test['Fare'].mean()

print('Filling in NaN with mean value.')

data_test = data_test.fillna(mean_val)
# One hot encode 'Sex' and 'Embarked' [Object to int64]

from sklearn.preprocessing import LabelEncoder

one_hot = LabelEncoder()

# Sex feature

data_train["Sex"]= one_hot.fit_transform(data_train["Sex"]) 

data_test["Sex"]= one_hot.fit_transform(data_test["Sex"])

# Embarked feature

data_train["Embarked"]= one_hot.fit_transform(data_train["Embarked"]) 

data_test["Embarked"]= one_hot.fit_transform(data_test["Embarked"])
# Check Training and Testing Data

data_train.info()

data_test.info()
## Training and Testing datasets

# Features select for training 

X_train = data_train.drop("Survived", axis=1)

# Feature sellect for target [Survived]

Y_train = data_train["Survived"]

X_test  = data_test.drop("PassengerId", axis=1).copy()
# Z score data scaling

from sklearn.preprocessing import StandardScaler

z_score = StandardScaler()

X_train = z_score.fit_transform(X_train)

X_test = z_score.transform(X_test)
# Model Creation (Support Vector Classification)

from sklearn.svm import SVC

SVC_model = SVC(kernel = 'rbf', random_state = 0)

SVC_model.fit(X_train, Y_train)

y_pred = SVC_model.predict(X_test)
# Accuracy of Method

from sklearn.metrics import accuracy_score

SVC_model.score(X_train, Y_train)

SVC_model = round(SVC_model.score(X_train, Y_train) * 100, 2)

print(SVC_model)
import pandas as pd

output = pd.DataFrame({'PassengerId': data_test.PassengerId, 'Survived': y_pred})

output.to_csv('my_submission_RFR.csv', index=False)

print('Saved')