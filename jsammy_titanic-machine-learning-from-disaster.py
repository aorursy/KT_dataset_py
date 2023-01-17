import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
train=pd.read_csv('../input/titanic/train.csv')
train
train.shape
train.dtypes
train.isnull().sum()
train.describe()
train.describe(include='object')
sns.countplot("Sex",hue="Survived",data=train)
sns.countplot("Embarked",hue="Survived",data=train)
sns.countplot("Pclass",hue="Survived",data=train)
sns.countplot("Survived",data=train)
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='median')
train[['Age']] = imputer.fit_transform(train[['Age']])
train["Age"].isnull().sum()
train['Title'] = train.Name.str.extract('([A-Za-z]+)\.', expand=False)
pd.crosstab(train['Title'],train['Sex'])
train['Title'] = train['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
train['Title'] = train['Title'].replace('Mlle', 'Miss')
train['Title'] = train['Title'].replace('Mme', 'Mrs')
train['Title'] = train['Title'].replace('Ms', 'Miss')
train['Title']
train["Title"].replace({"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5},inplace=True)
train["Title"].fillna(0)
train.head(5)
train = train.drop(['Name', 'PassengerId','Cabin','Ticket'], axis=1)
train.head(5)
train["Sex"].replace({"female":0 ,"male":1},inplace=True)
train["Embarked"].replace({"Q":1, "S":2, "C":3},inplace=True)
train["Embarked"]=train["Embarked"].fillna(0)
train.isnull().sum()
train.head(5)
sns.heatmap(train.corr(),cmap="coolwarm",annot=True)
test=pd.read_csv("../input/titanic/test.csv")
test
test[['Age']] = imputer.fit_transform(test[['Age']])
test['Title'] = test.Name.str.extract('([A-Za-z]+)\.', expand=False)
pd.crosstab(test['Title'],test['Sex'])
test['Title'] = test['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
test['Title'] = test['Title'].replace('Mlle', 'Miss')
test['Title'] = test['Title'].replace('Mme', 'Mrs')
test['Title'] = test['Title'].replace('Ms', 'Miss')
test["Title"].replace({"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5},inplace=True)
test["Title"].fillna(0)
test = test.drop(['Name','Cabin','Ticket'], axis=1)
test["Sex"].replace({"female":0 ,"male":1},inplace=True)
test["Embarked"].replace({"Q":1, "S":2, "C":3},inplace=True)
test["Embarked"]=test["Embarked"].fillna(0)
test.head()
test.isnull().sum()
test[['Fare']] = imputer.fit_transform(test[['Fare']])
X_train = train.drop("Survived", axis=1)
y_train = train["Survived"]
X_test = test.drop("PassengerId", axis=1).copy()
X_train.shape, y_train.shape, X_test.shape
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
#LOGISTIC REGRESSION 
lr = LogisticRegression()
lr.fit(X_train, y_train)
y_pred =  lr.predict(X_test)
y_pred
acc_lr = round(lr.score(X_train,y_train)*100, 2)
acc_lr
print("The Coefficients are ",lr.coef_)
print("The Intercept is",lr.intercept_)
#K-Neighbors
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
acc_knn = round(knn.score(X_train, y_train) * 100, 2)
acc_knn
#RANDOM FOREST
ran =  RandomForestClassifier(n_estimators = 100)
ran.fit(X_train,y_train)
y_pred = ran.predict(X_test)
acc_ran = round(ran.score(X_train,y_train)*100, 2)
acc_ran
models =  pd.DataFrame({
    'Model': ['Logistic Regression', 'K-Neighbors','Random Forest'],
    'Score': [acc_lr, acc_knn, acc_ran]})
models.sort_values(by='Score', ascending=False)
submission= pd.DataFrame({"PassengerId" : test["PassengerId"], "Survived" : y_pred})
submission.PassengerId = submission.PassengerId.astype(int)
submission.Survived = submission.Survived.astype(int)
submission.to_csv('submission.csv', header=True, index=False)
submission.head(10)
