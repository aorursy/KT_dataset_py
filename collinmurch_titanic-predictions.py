import pandas as pd
import numpy as np

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')

train_df.head()
train_df.info()
# Cabin is too difficult to deal with, and name/ticket seem to be useless
train_df = train_df.drop(['Ticket', 'Cabin', 'Name'], axis=1)
test_df = test_df.drop(['Ticket', 'Cabin', 'Name'], axis=1)
combine = [train_df, test_df]
# Change Embarked to be in terms of integers
def embarked_num(x):
    if x is 'S':
        return 0
    elif x is 'C':
        return 1
    elif x is 'Q':
        return 2
    else:
        return 3

for df in combine:
     df['Embarked'] = df['Embarked'].apply(lambda x: embarked_num(x))
    
train_df.head()
# Change sex to be in terms of integers
def sex_num(x):
    if x is 'male':
        return 0
    elif x is 'female':
        return 1
    else:
        return 2

for df in combine:
    df['Sex'] = df['Sex'].apply(lambda x: sex_num(x))
# Check fare mean
for df in combine:
    print(df['Fare'].mean())
# Fill NaN's with fare mean
for df in combine:
    df['Fare'] = df['Fare'].fillna(value=33.5)
# Check age mean
for df in combine:
    print(df['Age'].mean())
# Fill NaN's with age mean
for df in combine:
    df['Age'] = df['Age'].fillna(value=31)
# Drop Survived and PassengerId from X_train, and remove Survived from Y_train
X_train = train_df.drop(['Survived', 'PassengerId'], axis=1)
Y_train = train_df['Survived']

X_test = test_df.drop('PassengerId', axis=1)

X_train.shape, Y_train.shape, X_test.shape
logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
logreg_pred = logreg.predict(X_test)

logreg_accuracy = round(logreg.score(X_train, Y_train) * 100, 2)
logreg_accuracy
tree_clf = DecisionTreeClassifier()
tree_clf.fit(X_train, Y_train)
tree_pred = tree_clf.predict(X_test)

tree_accuracy = round(tree_clf.score(X_train, Y_train) * 100, 2)
tree_accuracy
knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train, Y_train)
Y_pred = knn_clf.predict(X_test)

knn_accuracy = round(knn_clf.score(X_train, Y_train) * 100, 2)
knn_accuracy
rforest_clf = RandomForestClassifier()
rforest_clf.fit(X_train, Y_train)

rforest_pred = rforest_clf.predict(X_test)

rforest_accuracy = round(rforest_clf.score(X_train, Y_train) * 100, 2)
rforest_accuracy
svc_clf = SVC()

svc_clf.fit(X_train, Y_train)

svc_pred = svc_clf.predict(X_test)

svc_accuracy = round(svc_clf.score(X_train, Y_train) * 100, 2)
svc_accuracy
sgd_clf = SGDClassifier()

sgd_clf.fit(X_train, Y_train)

sgd_pred = sgd_clf.predict(X_test)

sgd_accuracy = round(sgd_clf.score(X_train, Y_train) * 100, 2)
sgd_accuracy
final_pred = tree_pred
submission = pd.DataFrame({
    'PassengerId': test_df['PassengerId'],
    'Survived': final_pred
})
submission
submission.to_csv('submision.csv', index=False)