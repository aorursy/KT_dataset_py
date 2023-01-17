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
dataset_columns = ['Survived', 'Pclass', 'Sex', 'Age', 'Parch', 'Fare']
dataset = pd.read_csv('/kaggle/input/titanic/train.csv')
dataset = dataset[dataset_columns]
#dataset['Embarked'].fillna('None', inplace=True)
print(dataset.head())
dataset_columns = ['Pclass', 'Sex', 'Age', 'Parch', 'Fare']
dataset_test = pd.read_csv('/kaggle/input/titanic/test.csv')
submission = pd.DataFrame(dataset_test['PassengerId'])
dataset_test = dataset_test[dataset_columns]
print(dataset_test.head())
print(submission.head())
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
dataset['Sex'] = le.fit_transform(dataset['Sex'])
print(dataset.head())
dataset_test['Sex'] = le.fit_transform(dataset_test['Sex'])
print(dataset_test.head())
from sklearn.compose import ColumnTransformer 

#dataset_columns = ['Cherbourg','Nan','Queenstown','Southampton','Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']
#ct = ColumnTransformer([('ohe',OneHotEncoder(handle_unknown='ignore'),['Embarked'])], remainder='passthrough') 
#dataset = pd.DataFrame(ct.fit_transform(dataset),columns=dataset_columns)
#print(dataset.head())
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
dataset[['Age','Fare']] = sc.fit_transform(dataset[['Age','Fare']])
print(dataset.head())
dataset_test[['Age','Fare']] = sc.fit_transform(dataset_test[['Age','Fare']])
print(dataset_test.head())
from sklearn.impute import SimpleImputer
imputer_train = SimpleImputer()
dataset = imputer_train.fit_transform(dataset)
dataset_test = imputer_train.fit_transform(dataset_test)
survived_index = 0
from sklearn.model_selection import train_test_split
dataset_train, dataset_validate = train_test_split(dataset, test_size = 0.25, shuffle = True)

x_train = np.concatenate((dataset_train[:,:survived_index],dataset_train[:,survived_index+1:]), axis=1)
y_train = dataset_train[:,survived_index]

x_validate = np.concatenate((dataset_validate[:,:survived_index],dataset_validate[:,survived_index+1:]), axis=1)
y_validate = dataset_validate[:,survived_index]
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

classifier_rf = RandomForestClassifier()
classifier_rf.fit(x_train,y_train)

y_validate_pred = classifier_rf.predict(x_validate)
print(accuracy_score(y_validate, y_validate_pred))
y_test_pred = classifier_rf.predict(dataset_test)
(y_test_pred[:5])
from sklearn.neighbors import KNeighborsClassifier

classifier_knn = KNeighborsClassifier()
classifier_knn.fit(x_train,y_train)

y_validate_pred = classifier_knn.predict(x_validate)
print(accuracy_score(y_validate, y_validate_pred))
y_test_pred = classifier_knn.predict(dataset_test)
(y_test_pred[:5])
from sklearn.svm import SVC

classifier_svm = SVC()
classifier_svm.fit(x_train,y_train)

y_validate_pred = classifier_svm.predict(x_validate)
print(accuracy_score(y_validate, y_validate_pred))
y_test_pred = classifier_svm.predict(dataset_test)
(y_test_pred[:5])