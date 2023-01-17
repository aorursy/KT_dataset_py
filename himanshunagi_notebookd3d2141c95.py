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
train = pd.read_csv("/kaggle/input/titanic/train.csv")

train.head(10)
train.info()
test = pd.read_csv("/kaggle/input/titanic/test.csv")

test.head()
test.info()
train.drop(['Name', 'Age', 'Ticket', 'Fare', 'Cabin'], inplace= True, axis = 1)

test.drop(['Name', 'Age', 'Ticket', 'Fare', 'Cabin'], inplace= True, axis = 1)
print(train.info(), test.info())
print(train.isnull().sum(), test.isnull().sum())
data = [train, test]



for dataset in data:

    dataset.Embarked = dataset.Embarked.fillna('S')
train.isnull().sum()
genderMap = {"male": 0, "female": 1}

data = [train, test]



for dataset in data:

    dataset['Sex'] = dataset['Sex'].map(genderMap)
train['Sex'].value_counts()
embarkedMap = {"S": 0, "C": 1, "Q": 2}

data = [train, test]



for dataset in data:

    dataset['Embarked'] = dataset['Embarked'].map(embarkedMap)
train['Embarked'].value_counts()
print(train.info(), test.info())
X_train = train.drop(['Survived', 'PassengerId'], axis=1)

Y_train = train['Survived']
X_test = test.drop("PassengerId", axis=1)
from sklearn.linear_model import LogisticRegression



clf = LogisticRegression(random_state=0).fit(X_train, Y_train)

Y_pred = clf.predict(X_test)
print(Y_pred)
acc_logistic = round(clf.score(X_train, Y_train)*100, 2)

print (acc_logistic)
output = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': Y_pred})

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")