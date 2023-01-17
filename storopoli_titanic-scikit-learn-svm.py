from sklearn import svm

from sklearn.metrics import accuracy_score

import pandas as pd
train = pd.read_csv("/kaggle/input/titanic/train.csv")

test = pd.read_csv("/kaggle/input/titanic/test.csv")
print('Train Shape:', train.shape)

print('Test Shape:', test.shape)
train.info()
train.head(3)
train.describe()
train.isnull().sum()
test.isnull().sum()
print(train['Embarked'].value_counts())

print(test['Embarked'].value_counts())
train['Age'] = train['Age'].fillna(train['Age'].mean())

test['Age'] =  test['Age'].fillna(test['Age'].mean())

train['Embarked'].fillna('S',inplace=True)

test['Embarked'].fillna('S',inplace=True)
train['Embarked_no'] = train['Embarked'].map({'S':0,'C':1,'Q':1})

test['Embarked_no'] = test['Embarked'].map({'S':0,'C':1,'Q':1})

train['Sex_no'] = train['Sex'].map({'male':0, 'female':1})

test['Sex_no'] = test['Sex'].map({'male':0, 'female':1})
features = ['Pclass','SibSp','Parch','Embarked_no','Sex_no']

X_train = train[features]

X_test = test[features]

y_train = train['Survived']
clf = svm.SVC(kernel='linear')

clf.fit(X_train, y_train)
y_pred_linear = clf.predict(X_train)

y_pred_linear_sub = clf.predict(X_test)
print("Linear Kernel Accuracy:", round(accuracy_score(y_train, y_pred_linear), 3))
clf = svm.SVC(kernel='rbf')

clf.fit(X_train, y_train)
y_pred_rbf = clf.predict(X_train)

y_pred_rbf_sub = clf.predict(X_test)
print("Radial Kernel Accuracy:", round(accuracy_score(y_train, y_pred_rbf), 3))
submission_linear = pd.DataFrame([test['PassengerId'], y_pred_linear_sub]).T

submission_linear.columns = ['PassengerId','Survived']

submission_linear.to_csv('SVM_linear_submission.csv',index=False)



submission_rbf = pd.DataFrame([test['PassengerId'], y_pred_rbf_sub]).T

submission_rbf.columns = ['PassengerId','Survived']

submission_rbf.to_csv('SVM_rbf_submission.csv',index=False)