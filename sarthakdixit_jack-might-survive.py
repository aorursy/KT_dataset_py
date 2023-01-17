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
train = pd.read_csv(r'../input/titanic/train.csv')
train.head()
train.describe()
test = pd.read_csv(r'../input/titanic/test.csv')
test.head()
test.describe()
# Total Number of Females Survived
female = train.loc[train.Sex == "female"]["Survived"]
percent = sum(female)/len(female)
print("Female survival rate: ",percent)
# Total Number of Males Survived
male = train.loc[train.Sex == "male"]["Survived"]
percent = sum(male)/len(male)
print("Male survival rate: ",percent) # Damn, so low
# Dropping Unwanted Columns
train = train.drop(["PassengerId", "Name", "Ticket", "Fare", "Cabin", "Embarked"], axis=1)
test = test.drop(["Name", "Ticket", "Fare", "Cabin", "Embarked"], axis=1)
train.head()
test.head()
# Cleaning Training Data
train['Pclass'].fillna(0, inplace = True)
train['Age'].fillna(train['Age'].median(), inplace = True)
train['SibSp'].fillna(0, inplace = True)
train['Parch'].fillna(0, inplace = True)
# Cleaning Testing Data
test['Pclass'].fillna(0, inplace = True)
test['Age'].fillna(test['Age'].median(), inplace = True)
test['SibSp'].fillna(0, inplace = True)
test['Parch'].fillna(0, inplace = True)
# Encoding
from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder() 
train['Sex']= label_encoder.fit_transform(train['Sex'])
test['Sex']= label_encoder.fit_transform(test['Sex'])
# Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train =  pd.DataFrame(sc_X.fit_transform(train.drop(["Survived"],axis = 1),),
        columns=['Pclass', 'Sex', 'Age', 'SibSp', 'Parch'])
y = train.Survived
X_test =  pd.DataFrame(sc_X.fit_transform(test.drop(["PassengerId"],axis = 1),),
        columns=['Pclass', 'Sex', 'Age', 'SibSp', 'Parch'])
# SVM
from sklearn import svm
clf = svm.SVC(kernel='linear')
clf.fit(X_train, y)
y_pred = clf.predict(X_test)
output = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': pred})
output.to_csv('my_titanic.csv', index=False)
print("Your submission was successfully saved!")