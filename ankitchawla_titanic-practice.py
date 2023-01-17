# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import LabelEncoder

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_data = pd.read_csv('/kaggle/input/titanic/train.csv')
test_data = pd.read_csv('/kaggle/input/titanic/test.csv')
print(train_data.head())
print(test_data.head())
train_data.describe()
test_data.describe()
amount = train_data.count()
percent = (amount/train_data.isnull().count())*100
missing = 100 - percent

missing_data_train = pd.concat([amount,percent,missing],axis = 1,keys=['amount','percent_available','percent_missing'])
print(missing_data_train)
amount = test_data.count()
percent = (amount/test_data.isnull().count())*100
missing = 100 - percent

missing_data_test = pd.concat([amount,percent,missing],axis = 1,keys=['amount','percent_available','percent_missing'])
print(missing_data_test)
train_data['Age'] = train_data['Age'].fillna(train_data['Age'].mean())
train_data['Embarked'] = train_data['Embarked'].fillna(train_data['Embarked'].mode()[0])

test_data['Age'] = test_data['Age'].fillna(test_data['Age'].mean())
test_data['Fare'] = test_data['Fare'].fillna(test_data['Fare'].mean())
encoder = LabelEncoder()
train_data['Sex'] = encoder.fit_transform(train_data['Sex'])
for i in range(0,len(train_data['Embarked'])):
    if train_data['Embarked'][i] == 'S':
        train_data['Embarked'][i] = 0
    elif train_data['Embarked'][i] == 'C':
        train_data['Embarked'][i] =  1
    elif train_data['Embarked'][i] == 'Q':
        train_data['Embarked'][i] = 2
print(train_data.head())
encoder = LabelEncoder()
test_data['Sex'] = encoder.fit_transform(test_data['Sex'])
for i in range(0,len(test_data['Embarked'])):
    if test_data['Embarked'][i] == 'S':
        test_data['Embarked'][i] = 0
    elif test_data['Embarked'][i] == 'C':
        test_data['Embarked'][i] =  1
    elif test_data['Embarked'][i] == 'Q':
        test_data['Embarked'][i] = 2
print(test_data.head())
train_data = train_data.drop('Cabin',axis = 1)
test_data = test_data.drop('Cabin',axis = 1)

train_data = train_data.drop('Name',axis = 1)
test_data = test_data.drop('Name',axis = 1)

train_data = train_data.drop('Ticket',axis = 1)
test_data = test_data.drop('Ticket',axis = 1)

train_data = train_data.drop('PassengerId',axis=1)
Id = test_data['PassengerId']
test_data = test_data.drop('PassengerId',axis=1)

X_train = train_data.iloc[:,1:]
y_train = train_data.iloc[:,0]
print(y_train)
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(random_state=0)
scores = cross_val_score(clf,X_train,y_train,cv = 10)
print(scores)
print(np.mean(scores))
from sklearn.ensemble import RandomForestClassifier
rclf = RandomForestClassifier()
scores = cross_val_score(clf,X_train,y_train,cv=10)
print(scores)
print(np.mean(scores))
rclf.fit(X_train,y_train)
X_test = test_data
print(X_test.head())
prediction = rclf.predict(X_test)
print(prediction)
submission = pd.DataFrame({"PassengerID":Id,"Survived":prediction})
submission.to_csv('Submission_1.csv')
print(submission.head())
