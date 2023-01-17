import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
df1=pd.read_csv('../input/train.csv')

df2=pd.read_csv('../input/test.csv')
df1.head()
df2.head()
train = df1.drop(['Ticket','Fare','Embarked','Cabin','Name'] , axis=1)

test = df2.drop(['Ticket','Fare','Embarked','Cabin','Name'] , axis=1)

train.head()
train['Sex'].replace(['male','female'],[0,1],inplace=True)
X=train.drop(['Survived'],axis=1)

y=pd.DataFrame(train['Survived'])

X['Age'] = X['Age'].replace(np.nan, X['Age'].mean())

X.head()
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)
from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB 

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier

from mlxtend.classifier import StackingClassifier



clf1 = KNeighborsClassifier()

clf2 = RandomForestClassifier()

clf3 = GaussianNB()

clf4 = SVC()

meta_clf = LogisticRegression()

stacking_clf = StackingClassifier(classifiers=[clf1, clf2, clf3, clf4], meta_classifier=meta_clf)



clf1.fit(X_train, y_train)

clf2.fit(X_train, y_train)

clf3.fit(X_train, y_train)

clf4.fit(X_train, y_train)

stacking_clf.fit(X_train, y_train)



print('RNN Score:',clf1.score(X_test, y_test))

print('RF Score:',clf2.score(X_test, y_test))

print('GNB Score:',clf3.score(X_test, y_test))

print('SVC Score:',clf4.score(X_test, y_test))

print('Stacking Score:',stacking_clf.score(X_test, y_test))
test['Sex'].replace(['male','female'],[0,1],inplace=True)
test['Age'] = test['Age'].replace(np.nan, test['Age'].mean())
predictions = stacking_clf.predict(test)
PassengerId = test['PassengerId']
StackingSubmission = pd.DataFrame({ 'PassengerId': PassengerId,

                            'Survived': predictions })

StackingSubmission.to_csv("StackingSubmission.csv", index=False)