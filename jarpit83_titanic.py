import os
print(os.listdir("../input"))

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')
train = pd.read_csv("../input/train.csv")
submission_test = pd.read_csv("../input/test.csv")
train.head()
round( (train.isnull().sum() / len(train)) * 100, 2)
round( (submission_test.isnull().sum() / len(submission_test)) * 100, 2)

len(train[train.Age.isna()])
train['Age'].mean()
submission_test['Age'].mean()
train.Age.fillna(29.5, inplace=True)
submission_test.Age.fillna(30.5, inplace=True)
train.isnull().sum()
submission_test.isnull().sum()
train[['PassengerId', 'Embarked']].groupby("Embarked").count()
train.Embarked.fillna('S', inplace=True)
train.isnull().sum()
submission_test.Fare.fillna(submission_test.Fare.mean(), inplace=True)
submission_test.isnull().sum()
train['family_size'] = train.SibSp + train.Parch
submission_test['family_size'] = submission_test.SibSp + submission_test.Parch

train.drop(['Ticket', 'Name', 'Cabin'], axis=1, inplace=True)
submission_test.drop(['Ticket', 'Name', 'Cabin'], axis=1, inplace=True)

train.head()
submission_test.head()
plt.figure(num=1, figsize=(10, 4), dpi=100, facecolor='w', edgecolor='k')


plt.subplot(1, 3, 1)
sns.boxplot(y="Age", data=train)

plt.subplot(1, 3, 2)
sns.distplot(train.Age, bins=10)

plt.subplot(1, 3, 3)
plt.hist("Age", data=train) 

plt.show()

plt.figure(num=2, figsize=(12, 4), dpi=80, facecolor='w', edgecolor='k')


plt.subplot(1, 3, 1)
sns.barplot(x="Sex", y="Survived", data=train)
plt.ylabel("Survival Rate")
plt.title("Survival based on Gender")

plt.subplot(1, 3, 2)
sns.barplot(x="Pclass", y="Survived", data=train)
plt.ylabel("Survival Rate")
plt.title("Survival Based on Class")

plt.subplot(1, 3, 3)
sns.barplot(x="Embarked", y="Survived", data=train)
plt.ylabel("Survival Rate")
plt.title("Survival Based on Embarked")


plt.show()

plt.figure(num=3, figsize=(12, 4), dpi=80, facecolor='w', edgecolor='k')

plt.subplot(1, 2, 1)
sns.barplot(x="Pclass", y="Survived", hue="Sex", data=train)
plt.ylabel("Survival Rate")
plt.title("Survival based on Gender & PClass")

plt.subplot(1, 2, 2)
sns.barplot(x="Embarked", y="Survived", hue="Sex", data=train)
plt.ylabel("Survival Rate")
plt.title("Survival Based on Gender and Class")


plt.show()
plt.figure(num=4, figsize=(12, 4), dpi=80, facecolor='w', edgecolor='k')

plt.subplot(1, 3, 1)
sns.barplot(x="family_size", y="Survived", data=train)
plt.ylabel("Survival Rate")
plt.title("Survival Based on FamilySize")

plt.subplot(1, 3, 2)
sns.barplot(x="Parch", y="Survived", data=train)
plt.ylabel("Survival Rate")
plt.title("Survival Based on Parch")

plt.subplot(1, 3, 3)
sns.barplot(x="SibSp", y="Survived", data=train)
plt.ylabel("Survival Rate")
plt.title("Survival Based on SibSp")


plt.show()
plt.figure(num=5, figsize=(16, 4), dpi=80, facecolor='w', edgecolor='k')


plt.subplot(1, 3, 1)
sns.distplot(train.Fare)

plt.subplot(1, 3, 2)
sns.swarmplot(x='Pclass', y='Fare', hue='Survived',data=train)

plt.subplot(1, 3, 3)
sns.boxplot(x='Pclass', y='Fare', hue="Survived", data=train)

plt.show()
train.head()
submission_test.head()

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
X=train.iloc[:, 2:].values
y=train.iloc[:, 1].values
submission_X = submission_test.iloc[:, 1:].values
le_x = LabelEncoder()
X[:, 1] = le_x.fit_transform( X[:, 1] )
X[:, 6] = le_x.fit_transform(X[:, 6])

submission_X[:, 1] = le_x.fit_transform( submission_X[:, 1] )
submission_X[:, 6] = le_x.fit_transform( submission_X[:, 6] )

onehotencoder = OneHotEncoder(categorical_features = [0, 1, 6])
X = onehotencoder.fit_transform(X).toarray()

submission_X = onehotencoder.fit_transform(submission_X).toarray()
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)

submission_X = scaler.fit_transform(submission_X)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)

from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
classifier1 = SVC(kernel = 'linear')
classifier1.fit(X_train, y_train)

y_pred = classifier1.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
pd.DataFrame(cm, ['Actual: NOT', 'Actual: SURVIVED'], ['Predicted: NO', 'Predicted: SURVIVED'])
acc_svm_linear = round(classifier1.score(X_test, y_test) * 100, 2)

classifier2 = SVC(kernel = 'rbf')
classifier2.fit(X_train, y_train)

y_pred = classifier2.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
pd.DataFrame(cm, ['Actual: NOT', 'Actual: SURVIVED'], ['Predicted: NO', 'Predicted: SURVIVED'])
acc_svm_rbf = round(classifier2.score(X_test, y_test) * 100, 2)

classifier3 = SVC(kernel = 'poly')
classifier3.fit(X_train, y_train)

y_pred = classifier3.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
pd.DataFrame(cm, ['Actual: NOT', 'Actual: SURVIVED'], ['Predicted: NO', 'Predicted: SURVIVED'])
acc_svm_poly = round(classifier3.score(X_test, y_test) * 100, 2)

classifier4 = SVC(kernel = 'sigmoid')
classifier4.fit(X_train, y_train)

y_pred = classifier4.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
pd.DataFrame(cm, ['Actual: NOT', 'Actual: SURVIVED'], ['Predicted: NO', 'Predicted: SURVIVED'])

acc_svm_sigmoid = round(classifier4.score(X_test, y_test) * 100, 2)

from sklearn.ensemble import RandomForestClassifier
classifier_rf = RandomForestClassifier(n_estimators = 10, criterion = 'entropy')
classifier_rf.fit(X_train, y_train)

y_pred = classifier_rf.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
pd.DataFrame(cm, ['Actual: NOT', 'Actual: SURVIVED'], ['Predicted: NO', 'Predicted: SURVIVED'])

acc_svm_rf = round(classifier_rf.score(X_test, y_test) * 100, 2)

from sklearn.linear_model import LogisticRegression
classifier_lr = LogisticRegression()
classifier_lr.fit(X_train, y_train)

y_pred = classifier_rf.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
pd.DataFrame(cm, ['Actual: NOT', 'Actual: SURVIVED'], ['Predicted: NO', 'Predicted: SURVIVED'])
acc_svm_lr = round(classifier_lr.score(X_test, y_test) * 100, 2)


accs = {'Classifiers':['SVM-L', 'SVM-RBF', 'SVM-Poly', 'SVM-Sigmoid', 'Random_Forest', 'Logistic_Regression'],
       'Accuracy':[acc_svm_linear, acc_svm_rbf, acc_svm_poly, acc_svm_sigmoid, acc_svm_rf, acc_svm_lr]}
acc_df = pd.DataFrame.from_dict(accs)
plt.figure(num=6, figsize=(10, 8), dpi=80, facecolor='w', edgecolor='k')

sns.barplot(y="Accuracy", x="Classifiers",  data=acc_df)
plt.show()
submission_X_predict = classifier2.predict(submission_X)
submission_X_survived = pd.Series(submission_X_predict, name="Survived")
Submission = pd.concat([submission_test.PassengerId, submission_X_survived],axis=1)

Submission.head()
Submission.to_csv('submission.csv', index=False)
