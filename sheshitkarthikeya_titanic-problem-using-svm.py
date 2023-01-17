# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')

train.head()
test = pd.read_csv('../input/test.csv')

test.head()
PassengerIds = test['PassengerId']

PassengerIds
train['Sex'] = train['Sex'].apply(lambda x : 1 if x=='male' else 0)

train.head()
train['Age'] = train['Age'].fillna(np.mean(train['Age']))

train['Fare'] = train['Fare'].fillna(np.mean(train['Fare']))
train = train[['Survived','Pclass','Sex','Age','SibSp','Parch','Fare']]
train.head()
X = train.drop('Survived',axis = 1)

#df.drop(columns=['B', 'C'])

Y = train['Survived']
from sklearn.model_selection import train_test_split

X_train , X_test , Y_train , Y_test = train_test_split(X , Y , train_size = 0.5 , random_state = 42 )
X_train.head()
from sklearn import svm

clf = svm.SVC()

clf.fit(X_train, Y_train)
clf.score(X_train , Y_train)
clf.fit(X_test , Y_test)
clf.score(X_test,Y_test)
test['Sex'] = test['Sex'].apply(lambda x : 1 if x=='male' else 0)

test['Age'] = test['Age'].fillna(np.mean(test['Age']))

test['Fare'] = test['Fare'].fillna(np.mean(test['Fare']))

test = test[['Pclass','Sex','Age','SibSp','Parch','Fare']]

results = clf.predict(test)
results
submission_df = {"PassengerId": PassengerIds,

                 "Survived": results}

submission = pd.DataFrame(submission_df)
submission
submission.to_csv("submission.csv",index = False)
print(check_output(["ls", "../working"]).decode("utf8"))