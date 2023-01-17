import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import missingno
import seaborn as sns
plt.style.use('seaborn-whitegrid')
from sklearn import metrics
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
train = pd.read_csv("../input/titanic/train.csv")
test = pd.read_csv("../input/titanic/test.csv")
gender_submission = pd.read_csv("../input/titanic/gender_submission.csv")
len(train)
len(test)
train.head()
train.dtypes
train.isnull().sum()
missingno.bar(train)
dataF = pd.DataFrame()
train.PassengerId.head()
train.PassengerId.nunique()
train.Survived.head()
train.Survived.nunique()
train.Survived.isnull().sum()
dataF['Survived'] = train['Survived']
dataF
train.Pclass.head()
train.Pclass.isnull().sum()
train.Pclass.nunique()
dataF['Pclass'] = train['Pclass']
dataF
train.Name.head()
train.Name.isnull().sum()
train.Name.nunique()
train.Sex.head()
train.Sex.isnull().sum()
train.Sex.nunique()
dataF['Sex'] = train['Sex']
dataF
train.Age.head()
train.Age.isnull().sum()
train.SibSp.head()
train.SibSp.isnull().sum()
train.SibSp.nunique()
train.SibSp.value_counts()
dataF['SibSp'] = train['SibSp']
dataF
train.Parch.head()
train.Parch.isnull().sum()
train.Parch.nunique()
train.Parch.value_counts()
dataF['Parch'] = train['Parch']
dataF
#fig = plt.figure(figsize=(15, 8))
#sns.countplot(x=dataF.Parch, data=dataF)
train.Ticket.head()
train.Ticket.isnull().sum()
train.Ticket.nunique()
train.Ticket.value_counts()
train.Fare.head()
train.Fare.isnull().sum()
train.Fare.nunique()
train.Cabin.head()
train.Cabin.isnull().sum()
train.Embarked.head()
train.Embarked.isnull().sum()
train.Embarked.nunique()
train.Embarked.value_counts()
dataF['Embarked'] = train['Embarked']
dataF = dataF.dropna(subset=['Embarked'])
dataF
len(dataF)
dataF.head()
train.head()
dataF.head()
dataF_embarked_encode = pd.get_dummies(dataF['Embarked'], prefix='embarked')
dataF_sex_encode = pd.get_dummies(dataF['Sex'], prefix='sex')
dataF_pclass_encode = pd.get_dummies(dataF['Pclass'], prefix='pclass')
dataF_encoded = pd.concat([dataF, dataF_embarked_encode, dataF_sex_encode, dataF_pclass_encode], axis=1)
dataF_encoded = dataF_encoded.drop(['Pclass', 'Sex', 'Embarked'], axis=1)
dataF.head()
dataF_encoded.head()
dataF_encoded.head()
X_train = dataF_encoded.drop('Survived', axis=1)
y_train = dataF_encoded.Survived
X_train.head()
y_train.head()
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
log_acc = round(log_reg.score(X_train, y_train) * 100, 2)
print("Accuracy: %s" % log_acc)
GNB = GaussianNB()
GNB.fit(X_train, y_train)
GNB_acc = round(GNB.score(X_train, y_train) * 100, 2)
print("Accuracy: %s" % GNB_acc)
LSVM = LinearSVC()
LSVM.fit(X_train, y_train)
LSVM_acc = round(LSVM.score(X_train, y_train) * 100, 2)
print("Accuracy: %s" % LSVM_acc)
Dec_tree = DecisionTreeClassifier()
Dec_tree.fit(X_train, y_train)
Dec_acc = round(Dec_tree.score(X_train, y_train) * 100, 2)
print("Accuracy: %s" % Dec_acc)
models = pd.DataFrame({
    'Model': ['Logistic Regression', 'Naive Bayes', 'Linear SVC', 
              'Decision Tree'],
    'Score': [log_acc, GNB_acc, LSVM_acc, Dec_acc]})
print("-Accuracy Scores-")
models.sort_values(by='Score', ascending=False)
X_train.head()
test.head()
test_embarked_encode = pd.get_dummies(test['Embarked'], prefix='embarked')
test_sex_encode = pd.get_dummies(test['Sex'], prefix='sex')
test_pclass_encode = pd.get_dummies(test['Pclass'], prefix='pclass')
test_encoded = pd.concat([test, test_embarked_encode, test_sex_encode, test_pclass_encode], axis=1)
test.head()
test_encoded.head()
test_encoded = test_encoded.drop(['Pclass', 'Sex', 'Embarked', 'Name',
                                  'Age', 'PassengerId','Ticket','Fare',
                                  'Cabin'], axis=1)
test_encoded.head()
X_train.head()
pred = Dec_tree.predict(test_encoded)
pred[:10]
gender_submission.head()
result = pd.DataFrame()
result['PassengerId'] = test['PassengerId']
result['Survived'] = pred
result.head()
result.to_csv('./sub.csv', index=False)