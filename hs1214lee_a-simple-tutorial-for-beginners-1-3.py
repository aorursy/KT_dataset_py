import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import pandas as pd



train = pd.read_csv('../input/titanic/train.csv')

test = pd.read_csv('../input/titanic/test.csv')

print('Shape of train dataset: {}'.format(train.shape))

print('Shape of test dataset: {}'.format(test.shape))
train.head()
test.head()
train.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'],axis = 1,inplace = True)

train.head()
test.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'],axis = 1,inplace = True)

test.head()
train.isnull().values.any()
train.isnull().sum()
train['Age'].value_counts()
train['Age'] = train['Age'].fillna(24)

train.isnull().sum()
test.isnull().sum()
test['Age'].value_counts()
test['Fare'].value_counts()
test['Age'] = test['Age'].fillna(24)

test['Fare'] = test['Fare'].fillna(7.75)

test.isnull().sum()
train_y = train['Survived']

train_x = train.drop('Survived', axis=1)

train_x.head()
train_y.head()
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()

model.fit(train_x, train_y)
from sklearn import metrics

pred = model.predict(train_x)

metrics.accuracy_score(pred, train_y)
import time



timestamp = int(round(time.time() * 1000))



pred = model.predict(test)

output = pd.DataFrame({"PassengerId":test.PassengerId , "Survived" : pred})

output.to_csv("submission_" + str(timestamp) + ".csv",index = False)