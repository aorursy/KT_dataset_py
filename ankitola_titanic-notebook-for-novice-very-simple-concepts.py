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
import matplotlib.pyplot as plt

import seaborn as sns
data_raw = pd.read_csv('../input/titanic/train.csv')



data_val = pd.read_csv('../input/titanic/test.csv')



data1 = data_raw.copy(deep = True)



data_cleaner = [data1, data_val]



data_raw.info()



data_raw.isnull().sum()

data2 = data_val.copy()
for dataset in data_cleaner:    

    #complete missing age with median

    dataset['Age'].fillna(dataset['Age'].median(), inplace = True)



    #complete embarked with mode

    dataset['Embarked'].fillna(dataset['Embarked'].mode()[0], inplace = True)



    #complete missing fare with median

    dataset['Fare'].fillna(dataset['Fare'].median(), inplace = True)

    

#delete the cabin feature/column and others previously stated to exclude in train dataset

drop_column = ['PassengerId','Cabin', 'Ticket','Name']

drop_column2 = ['Cabin', 'Ticket','Name']

data1.drop(drop_column, axis=1, inplace = True)

data_val.drop(drop_column2, axis=1, inplace = True)



print(data1.isnull().sum())

print("-"*10)

print(data_val.isnull().sum())
data1.info()

data_val.info()
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

data1[['Sex']] = le.fit_transform(data1[['Sex']])
data1.head()
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

data_val[['Sex']] = le.fit_transform(data_val[['Sex']])

data_val.head()
for dataset in data_cleaner:

    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
data1.head()
data_val.head()
X_train = data1.drop(['Survived'], axis = 1)

y_train = data1['Survived']

X_test = data_val.drop(['PassengerId'], axis = 1)
X_train.head()
print(y_train)
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train[['Age','Fare','SibSp','Parch']] = sc.fit_transform(X_train[['Age','Fare','SibSp','Parch']])

X_test[['Age','Fare','SibSp','Parch']] = sc.transform(X_test[['Age','Fare','SibSp','Parch']])
X_train.head()
from sklearn.svm import SVC

classifier = SVC(kernel = 'rbf', random_state = 0)

classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
from sklearn.model_selection import cross_val_score

accuracies = cross_val_score(estimator=classifier, cv=10,X = X_train, y = y_train)

print("Accuracy :{:.2f}%".format(accuracies.mean()*100))

print("Standard Deviation :{:.2f}%".format(accuracies.std()*100))
submission = pd.DataFrame({

        "PassengerId": data_val["PassengerId"],

        "Survived": y_pred

    })
submission.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")