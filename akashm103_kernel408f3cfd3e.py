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
from sklearn.linear_model import SGDClassifier,LogisticRegression

from sklearn.metrics import classification_report

from sklearn.ensemble import RandomForestClassifier
train_data=pd.read_csv("/kaggle/input/titanic/train.csv")

train_data.shape
test_data=pd.read_csv("/kaggle/input/titanic/test.csv")

test_data.head()
features = ["Sex","Age","Survived","Pclass","SibSp","Parch","Fare"]

features1=["Sex","Age","Pclass","SibSp","Parch","Fare"]

x_train=train_data[features]

x_test=test_data[features1]

x_train
dict = {'male':0, 'female':1}

x_train['Sex']=x_train["Sex"].map(dict)

x_test["Sex"]=x_test['Sex'].map(dict)

# features = ["Sex","Age","Survived","Pclass","SibSp","Parch","Fare"]

# features1=["Sex","Age","Pclass","SibSp","Parch","Fare"]

# x_train = pd.get_dummies(train_data[features])

# x_test = pd.get_dummies(test_data[features1])

x_train.head(10)

#x_train=x_train[['Sex','Age']]
import random

x_train['Age'].median()

x_train['Age'].std()

x=pd.isnull(x_train['Age'])

y=pd.isnull(x_test['Age'])

combine=[x_train,x_test]

ages=[]

i=0

for dataset in combine:

    i+=1

    print(i)

    x=pd.isnull(dataset['Age'])

    for _ in range(np.size(dataset[x],0)):

        ages.append(np.random.uniform(dataset['Age'].median(),dataset['Age'].std()))

    #print(len(ages))

    dataset.loc[dataset.Age.isnull(),'Age']=ages

    ages.clear()

    #print(dataset[x])

    dataset['Sex']=dataset['Sex'].astype("int")

print(x_test.describe())
x_test['Fare'].fillna(x_test['Fare'].dropna().median(), inplace=True)

x_test.describe()

x_train.describe()
# x_train=x_train.dropna(axis=0,how='any')

# x_test=x_test.dropna(axis=0,how='any')

y_train=x_train["Survived"]

x_train=x_train.drop(columns=["Survived"])

print(x_train)
# model=LogisticRegression(max_iter=80000,verbose=1)

# model.fit(x_train,y_train)

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)

model.fit(x_train, y_train)
from sklearn.metrics import accuracy_score
predictions=model.predict(x_test)

predictions

predictions1=model.predict(x_train)

print(accuracy_score(y_train,predictions1))

#print(predictions.shape)
output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")