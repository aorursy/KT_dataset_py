import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

from sklearn.tree import DecisionTreeClassifier

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
train = pd.read_csv('/kaggle/input/titanic/train.csv')

test = pd.read_csv('/kaggle/input/titanic/test.csv')

submission = pd.read_csv('/kaggle/input/titanic/gender_submission.csv')
submission
train.head()
test.head()
train.describe(include='all')
train.shape
train.isnull().sum()
train.info()
all = pd.concat([train, test], axis=0)
all.head()
all.isna().sum()
all.Age = all.Age.fillna(np.mean(all.Age))

all.Fare = all.Fare.fillna(np.mean(all.Fare))

all.Embarked = all.Embarked.fillna('S')
all = all.drop(['Cabin','Name','Sex','Ticket','Cabin','Embarked'], axis=1)
test = all[all.Survived.isnull()]

train = all[~(all.Survived.isnull())]
print(train.shape, test.shape)
X_train = train.drop(['Survived'],axis=1)

y_train = train.Survived

X_test = test.drop(['Survived'],axis=1)
dtc = DecisionTreeClassifier()

dtc.fit(X_train, y_train)
prediction = dtc.predict(X_test)

submission = pd.concat([test.PassengerId, pd.Series(prediction, index=test.index)],axis=1)

submission.columns=['PassengerId', 'Survived']

submission['Survived'] =submission['Survived'].astype(int)
submission.to_csv('submission_first.csv', index=False)
submission.head()