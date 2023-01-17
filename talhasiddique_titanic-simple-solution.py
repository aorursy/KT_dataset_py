import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
train = pd.read_csv('../input/titanic/train.csv')
test = pd.read_csv('../input/titanic/test.csv')
test_df = test.copy()
train.tail()
train.info()
train.describe()
train[['Pclass','Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived',ascending=False)
train[['Sex','Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived',ascending=False)
train[['Parch','Survived']].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived',ascending=False)
train[['SibSp','Survived']].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived',ascending=False)
train = train.drop(['Name','PassengerId','Ticket','Cabin'],axis=1)
train.head()
test = test.drop(['Name','PassengerId','Ticket','Cabin'],axis=1)
test.head()
train.shape, test.shape
train['Sex'] = train['Sex'].map( {'female': 1, 'male': 0} ).astype(int)
test['Sex'] = test['Sex'].map( {'female': 1, 'male': 0} ).astype(int)
test['Age'].median()
train['Age'] = train['Age'].fillna(28).astype(int)
test['Age'] = test['Age'].fillna(27).astype(int)
freq_port = train.Embarked.dropna().mode()[0]
train['Embarked'] = train['Embarked'].fillna(freq_port)
test['Embarked'] = test['Embarked'].fillna(freq_port)
train['Embarked'] = train['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
test['Embarked'] = test['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
train.tail()
test['Fare'].fillna(test['Fare'].dropna().median(), inplace=True)
test.head()
X_train = train.drop(['Survived'],axis=1)
Y_train = train['Survived']
X_test = test
X_train.shape, Y_train.shape, X_test.shape
model = LogisticRegression()
model.fit(X_train,Y_train)
Y_pred = model.predict(X_test)
lr_acc = round(model.score(X_train,Y_train)*100,2)
print(lr_acc)
model = SVC()
model.fit(X_train,Y_train)
Y_pred = model.predict(X_test)
svm_acc = round(model.score(X_train,Y_train)*100,2)
print(svm_acc) 
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train,Y_train)
Y_pred = model.predict(X_test)
knn_acc = round(model.score(X_train,Y_train)*100,2)
print(knn_acc)
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train,Y_train)
Y_pred = model.predict(X_test)
rfc_acc = round(model.score(X_train,Y_train) * 100,2)
print(rfc_acc)
model = DecisionTreeClassifier()
model.fit(X_train,Y_train)
Y_pred = model.predict(X_test)
dtc_acc = round(model.score(X_train,Y_train)*100,2)
print(dtc_acc)
model = LinearSVC()
model.fit(X_train,Y_train)
Y_pred = model.predict(X_test)
lsvc_acc = round(model.score(X_train,Y_train)*100,2)
print(lsvc_acc)
models = pd.DataFrame({
    'Model': [
              'Logistic Regression',
              'Support Vector Machine Classifier',
              'K Nearest Neighbors',
              'Random Forest',
              'Decision Tree',
              'Linear Support Vector Classifier'
              ],
    'Score': [
              lr_acc,
              svm_acc,
              knn_acc,
              rfc_acc,
              dtc_acc,
              lsvc_acc
             ]
})
models.sort_values(by='Score',ascending=False)
submission = pd.DataFrame({
    "PassengerId": test_df['PassengerId'],
    "Survived": Y_pred
})
submission.head()
#submission.to_csv('submission.csv',index=False)