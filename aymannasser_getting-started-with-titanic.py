# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.impute import SimpleImputer

from sklearn.compose import ColumnTransformer

from sklearn.metrics import confusion_matrix, accuracy_score

from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler

from sklearn.model_selection import train_test_split



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_data = pd.read_csv("../input/titanic/train.csv")

train_data.head()
test_data = pd.read_csv("../input/titanic/test.csv")

test_data2 = test_data

test_data.head()
# Dropping some features

features = ['PassengerId', 'Name', 'Ticket', 'Cabin']

train_data = train_data.drop(features, axis='columns')

test_data = test_data.drop(features, axis='columns')

test_data

#Which columns have missing values?

display(train_data.isnull().sum().sort_values(ascending=False))

display(test_data.isnull().sum().sort_values(ascending=False))
# Filling Nan values for train data



# Age

train_data['Age'].fillna(train_data['Age'].mean(), inplace=True)



# Embarked

train_data['Embarked'].fillna('S', inplace = True)
# Filling Nan values for test data



#Fare

test_data['Fare'].fillna(test_data['Fare'].mean(), inplace=True)



#Age

test_data['Age'].fillna(test_data['Age'].mean(), inplace=True)

# Mapping categorical features



# Sex

train_data['Sex'] = train_data['Sex'].map({'male':0, 'female':1}).astype(int)

test_data['Sex'] = test_data['Sex'].map({'male':0, 'female':1}).astype(int)



# Embarked

train_data['Embarked'] = train_data['Embarked'].map({'S' : 0, 'C' : 1, 'Q': 2}).astype(int)

test_data['Embarked'] = test_data['Embarked'].map({'S' : 0, 'C' : 1, 'Q': 2}).astype(int)



train_data



# Loading features & labels of train set

X_train = train_data.iloc[:, 1:].values

y_train = train_data.iloc[:, 0].values



# Loading features & labels of test set

X_test = test_data.iloc[:, :].values

X_train
# Applying feature scaling to age feature

# stand_scaler = StandardScaler()

# X_train[:, [2,5]] = stand_scaler.fit_transform(X_train[:, [2,5]])



# # Applying the same scaler that applied to the training set

# X_test[:, [2,5]] = stand_scaler.transform(X_test[:, [2,5]])

# X_test

# Training data on SVM classifier

# from sklearn.svm import SVC

# from sklearn.ensemble import RandomForestClassifier

# classifier = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', max_depth=10, max_features='auto',random_state = 0)

# classifier.fit(X_train, y_train)

# print(accuracy_score(y_train,classifier.predict(X_train)))

# Training data on XGBoost classifier

from xgboost import XGBClassifier

classifier = XGBClassifier()

classifier.fit(X_train, y_train)

print(accuracy_score(y_train,classifier.predict(X_train)))

predictions = classifier.predict(X_test)



output = pd.DataFrame({'PassengerId': test_data2.PassengerId, 'Survived': predictions})

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")
# Measuring model accuracy

# from sklearn.metrics import confusion_matrix, accuracy_score



# print(confusion_matrix(y_val, y_predicted))

# print(accuracy_score(y_val, y_predicted))
