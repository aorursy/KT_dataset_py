import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
%matplotlib inline
sns.set()
train = pd.read_csv('../input/titanic/train.csv')
test = pd.read_csv('../input/titanic/test.csv')
print(train.shape,test.shape)
train.head()
test.head()
train.info()
test.info()
train.skew()
test.skew()
train.duplicated().sum()
train.isnull().sum()
## Categorical v/s Categorical
plt.figure(figsize=(10,6))
plt.title('Count of Passengers Survived')
sns.countplot(train['Pclass'][train['Survived']==1])
plt.show()
plt.figure(figsize=(10,6))
plt.title('Count of gender of passengers Survived')
sns.countplot(train['Sex'][train['Survived']==1])
plt.show()
plt.figure(figsize=(10,6))
plt.title('Count of Passengers having Spouses/Siblings Dead')
sns.countplot(train['SibSp'][train['Survived']==0])
plt.show()
plt.figure(figsize=(10,6))
plt.title('Count of passengers having parents Dead')
sns.countplot(train['Parch'][train['Survived']==0])
plt.show()
plt.figure(figsize=(10,6))
plt.title('Count of passengers having specific Starting point Dead')
sns.countplot(train['Embarked'][train['Survived']==0])
plt.show()
# Numeric v/s Categorical
plt.figure(figsize=(12,5))
sns.distplot(train['Age'][train['Survived']==0])
sns.distplot(train['Age'][train['Survived']==1])
plt.legend(['Dead','Survived'])
plt.show()
plt.figure(figsize=(10,5))
sns.pointplot(x='Embarked', y='Age', hue='Survived', data = train)
plt.show()
train['Age'].fillna(train['Age'].median(), inplace = True)
test['Age'].fillna(test['Age'].median(), inplace = True)
train.isnull().sum()
train.drop(['Name'] , axis = 1, inplace = True)
test.drop(['Name'] , axis = 1, inplace = True)
test.isnull().sum()
train.head()
test.head()
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
train['Sex'] = labelencoder.fit_transform(train['Sex'])
train.head(2)
plt.figure(figsize=(10,5))
sns.countplot(train['Embarked'])
train['Embarked'].fillna('S', inplace = True)
train['Embarked'] = labelencoder.fit_transform(train['Embarked'])
train.isnull().sum()
test.isnull().sum()
test['Fare'].fillna(test.groupby('Pclass')['Fare'].transform('median'), inplace = True)
test.head(50)
train.drop(['Cabin'], axis = 1, inplace= True)
test.drop(['Cabin'], axis = 1, inplace= True)
print(test.info(), train.info())
train.drop(['Ticket','PassengerId'], axis = 1, inplace= True)
test.drop(['Ticket'], axis = 1, inplace= True)
train_data = train.drop('Survived', axis = 1)
target = train['Survived']
train_data.shape, target.shape
test.head()
test['Sex'] = labelencoder.fit_transform(test['Sex'])
test['Embarked'] = labelencoder.fit_transform(test['Embarked'])
train_data.head(10)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
train.info()
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
k_fold = KFold(n_splits=10, shuffle=True, random_state=0)
clf = KNeighborsClassifier(n_neighbors=13)
scoring = 'accuracy'
score = cross_val_score(clf, train_data, target, cv = k_fold, n_jobs=1, scoring=scoring)
print(score)
round(np.mean(score)*100, 2)
clf = DecisionTreeClassifier()
scoring = 'accuracy'
score = cross_val_score(clf, train_data, target, cv=k_fold, n_jobs=1, scoring= scoring)
print(score)
round(np.mean(score)*100, 2)
clf = RandomForestClassifier(n_estimators=13)
scoring = 'accuracy'
score = cross_val_score(clf, train_data, target, cv=k_fold, n_jobs=1, scoring= scoring)
print(score)
round(np.mean(score)*100, 2)
clf = GaussianNB()
scoring = 'accuracy'
score = cross_val_score(clf, train_data, target, cv=k_fold, n_jobs=1, scoring= scoring)
print(score)
round(np.mean(score)*100, 2)
clf = SVC()
scoring = 'accuracy'
score = cross_val_score(clf, train_data, target, cv=k_fold, n_jobs=1, scoring= scoring)
print(score)
round(np.mean(score)*100, 2)
clf = RandomForestClassifier(n_estimators=13)
clf.fit(train_data, target)
test_data = test.drop("PassengerId", axis=1).copy()
prediction = clf.predict(test_data)
submission = pd.DataFrame({
    'PassengerId' : test['PassengerId'],
    'Survived' : prediction
})
submission.to_csv('submission.csv', index=False)
submission = pd.read_csv('submission.csv')
submission.head()
