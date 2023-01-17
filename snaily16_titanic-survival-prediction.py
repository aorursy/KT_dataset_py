#import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')
train_df.columns.values
#preview the data
train_df.head(10)
train_df.describe()
train_df.describe(include=['O'])
train_df.isnull().sum().sort_values(ascending=False)
train_df[['Pclass','Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train_df[['Sex','Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train_df[['Parch','Survived']].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train_df[['SibSp','Survived']].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)
g = sns.FacetGrid(train_df, col='Sex', row='Survived', margin_titles=True)
g.map(plt.hist, 'Age', bins=20)
grid = sns.FacetGrid(train_df, col = 'Survived', row = 'Pclass', size=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend();
grid = sns.FacetGrid(train_df, row='Embarked', size=2.2, aspect = 1.6)
grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')
grid.add_legend()
grid = sns.FacetGrid(train_df, row='Embarked', col='Survived', size = 2.2, aspect=1.6)
grid.map(sns.barplot, 'Sex', 'Fare', alpha=.5, ci=None)
grid.add_legend()
train_df = train_df.drop(['Cabin','Ticket'], axis = 1)
test_df = test_df.drop(['Cabin', 'Ticket'], axis=1)
combine = [train_df, test_df]

for data in combine:
    data['Title'] = data.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
    
pd.crosstab(train_df['Title'], [train_df['Sex'], train_df['Survived']])
grid = sns.countplot(x='Title', data=train_df)
grid = plt.setp(grid.get_xticklabels(), rotation=45)
for data in combine:
    data['Title'] = data['Title'].replace(['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    data['Title'] = data['Title'].replace('Ms', 'Miss')
    data['Title'] = data['Title'].replace('Mme', 'Mrs')
    data['Title'] = data['Title'].replace('Mlle', 'Miss')
    
train_df[['Title','Survived']].groupby(['Title'], as_index=False).mean()
grid = sns.countplot(x='Title', data=train_df)
grid = plt.setp(grid.get_xticklabels(), rotation=45)
title_mapping = {'Mr':1, 'Miss':2, 'Mrs':3, 'Master':4, 'Rare':5}
for data in combine:
    data['Title'] = data['Title'].map(title_mapping)
    data['Title'] = data['Title'].fillna(0)
    
train_df.head()
train_df = train_df.drop(['Name', 'PassengerId'], axis = 1)
test_df = test_df.drop(['Name'], axis=1)
combine = [train_df, test_df]
train_df.shape, test_df.shape
for data in combine:
    data['Sex'] = data['Sex'].map({'female':1, 'male':0}).astype(int)
    
train_df.head()
for data in combine:
    mean = train_df['Age'].mean()
    std = test_df['Age'].std()
    null_count = data['Age'].isnull().sum()
    rand_age = np.random.randint(mean - std, mean + std, size = null_count)
    
    age_slice = data["Age"].copy()
    age_slice[np.isnan(age_slice)] = rand_age
    data["Age"] = age_slice
    data["Age"] = train_df["Age"].astype(int)
    
train_df["Age"].isnull().sum()
train_df.head()
train_df['AgeBand']= pd.cut(train_df['Age'],5)
train_df[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True)
for data in combine:
    data.loc[data['Age'] <=16, 'Age'] = 0
    data.loc[(data['Age'] >16 ) & (data['Age'] <= 32), 'Age']=1
    data.loc[(data['Age'] > 32) & (data['Age'] <= 48), 'Age'] = 2
    data.loc[(data['Age'] > 48) & (data['Age'] <= 64), 'Age'] = 3
    data.loc[ data['Age'] > 64, 'Age']
train_df.head()
train_df = train_df.drop(['AgeBand'], axis=1)
combine = [train_df, test_df]
for data in combine:
    data['FamilySize'] = data['SibSp'] + data['Parch'] + 1
    
train_df[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived', ascending=False)
for data in combine:
    data['IsAlone']=0
    data.loc[data['FamilySize'] == 1, 'IsAlone']=1
    
train_df[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean()
train_df = train_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
test_df = test_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
combine = [train_df, test_df]

train_df.head()
for data in combine:
    data['AgeClass'] = data['Age']* data['Pclass']
freq = train_df.Embarked.dropna().mode()[0]
freq
for data in combine:
    data['Embarked'] = data['Embarked'].fillna(freq)
    
train_df[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False)
#convert categorical feature to numeric
for data in combine:
    data['Embarked'] = data['Embarked'].map({'S':0,'C':1,'Q':2}).astype(int)
    
train_df.head()
train_df["Fare"].isnull().sum(), test_df["Fare"].isnull().sum()
test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True)
train_df['FareBand'] = pd.qcut(train_df['Fare'], 4)
train_df[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand', ascending=True)
for data in combine:
    data.loc[ data['Fare'] <= 7.91, 'Fare'] = 0
    data.loc[(data['Fare'] > 7.91) & (data['Fare'] <= 14.454), 'Fare'] = 1
    data.loc[(data['Fare'] > 14.454) & (data['Fare'] <= 31), 'Fare']   = 2
    data.loc[ data['Fare'] > 31, 'Fare'] = 3
    data['Fare'] = data['Fare'].astype(int)

train_df = train_df.drop(['FareBand'], axis=1)
combine = [train_df, test_df]
    
train_df.head()
test_df.head()
X_train = train_df.drop('Survived', axis=1)
Y_train = train_df['Survived']
X_test = test_df.drop('PassengerId', axis=1).copy()
X_train.shape, Y_train.shape, X_test.shape
# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
sgd = SGDClassifier(max_iter=5, tol=None)
sgd.fit(X_train, Y_train)
Y_pred = sgd.predict(X_test)

acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)
print(acc_sgd, '%')
logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_test)

acc_log = round(logreg.score(X_train, Y_train) * 100, 2)
print(acc_log, '%')
pcpt = Perceptron(max_iter=5)
pcpt.fit(X_train, Y_train)
Y_pred = pcpt.predict(X_test)

acc_pcpt = round(pcpt.score(X_train, Y_train)*100, 2)
print(acc_pcpt, '%')
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, Y_train)
Y_pred = rf.predict(X_test)

acc_rf = round(rf.score(X_train, Y_train) * 100, 2)
print(acc_rf, '%')
dt = DecisionTreeClassifier()
dt.fit(X_train, Y_train)
Y_pred = dt.predict(X_test)

acc_dt = round(dt.score(X_train, Y_train)*100,2)
print(acc_dt, '%')
svc = LinearSVC()
svc.fit(X_train, Y_train)
Y_pred = svc.predict(X_test)

acc_svc = round(svc.score(X_train, Y_train)*100, 2)
print(acc_svc, '%')
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, Y_train)
Y_pred = knn.predict(X_test)

acc_knn = round(knn.score(X_train, Y_train)*100, 2)
print(acc_knn, '%')
gnb = GaussianNB()
gnb.fit(X_train, Y_train)
Y_pred = gnb.predict(X_test)

acc_gnb = round(gnb.score(X_train, Y_train)*100, 2)
print(acc_gnb, '%')
models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 
              'Random Forest', 'Naive Bayes', 'Perceptron', 
              'Stochastic Gradient Decent', 'Linear SVC', 
              'Decision Tree'],
    'Score': [acc_svc, acc_knn, acc_log, 
              acc_rf, acc_gnb, acc_pcpt, 
              acc_sgd, acc_svc, acc_dt]})
models.sort_values(by='Score', ascending=False)
# Random Forest
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, Y_train)
Y_pred = rf.predict(X_test)

acc_rf = round(rf.score(X_train, Y_train) * 100, 2)
print(acc_rf, '%')
submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": Y_pred
    })

submission.to_csv('submission.csv', index=False)