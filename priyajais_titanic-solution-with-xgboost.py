import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
train = pd.read_csv('../input/titanic/train.csv')
test = pd.read_csv('../input/titanic/test.csv')
train.info()
print('*'*45)
test.info()
train['Ticket'].value_counts()
train.describe(include=['O'])
print(train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False))
sns.countplot(train['Pclass'], hue = train.Survived)
print(train[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False))
sns.countplot(train['Sex'], hue = train.Survived)
print(train[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False))
sns.countplot(train['Embarked'], hue = train.Survived)
print(train[['SibSp', 'Survived']].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False))
sns.countplot(train['SibSp'], hue = train.Survived)
train[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)
grid = sns.FacetGrid(train, col='Survived', row='Pclass', size=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend();
grid = sns.FacetGrid(train, row='Embarked', size=2.2, aspect=1.6)
grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')
grid.add_legend()
train.columns
columns = ['Pclass', 'Name', 'Sex', 'Age', 'SibSp',
       'Parch', 'Fare', 'Embarked']
df = train[columns].append(test[columns])
df['Title'] = df.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

pd.crosstab(df['Title'],df['Sex'])
df['Title'] = df['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
df['Title'] = df['Title'].replace('Mlle', 'Miss')
df['Title'] = df['Title'].replace('Ms', 'Miss')
df['Title'] = df['Title'].replace('Mme', 'Mrs')
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
df['Title'] = df['Title'].map(title_mapping)
df.head()
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
df = df.drop(['Parch', 'SibSp'], axis=1)
df['Sex'].fillna(df['Sex'].mode(),inplace =True)
df['Embarked'].fillna(df['Embarked'].mode(),inplace =True)
df['Fare'].fillna(0,inplace =True)
df = pd.get_dummies(df, columns = ['Sex','Embarked'],drop_first=True)
df.drop(['Name'],axis=1,inplace= True)
df.head()
df.isnull().sum()
df['Age'] = df['Age'].round()
df['Age'].fillna(df['Age'].mean(),inplace = True)
df.isnull().sum()
df['AgeRange'] = pd.cut(df['Age'],5)
df.loc[ df['Age'] <= 16, 'Age'] = 0
df.loc[(df['Age'] > 16) & (df['Age'] <= 32), 'Age'] = 1
df.loc[(df['Age'] > 32) & (df['Age'] <= 48), 'Age'] = 2
df.loc[(df['Age'] > 48) & (df['Age'] <= 64), 'Age'] = 3
df.loc[ df['Age'] > 64, 'Age']=4
df.head(7)
df['FareBand'] = pd.qcut(df['Fare'], 4)
df['FareBand']
df.loc[ df['Fare'] <= 7.91, 'Fare'] = 0
df.loc[(df['Fare'] > 7.91) & (df['Fare'] <= 14.454), 'Fare'] = 1
df.loc[(df['Fare'] > 14.454) & (df['Fare'] <= 31), 'Fare']   = 2
df.loc[df['Fare'] > 31, 'Fare'] = 3

df = df.drop(['FareBand','AgeRange'], axis=1)
    
df.head(10)
df['Agec'] = df['Age']*df.Pclass
df = df.drop(['Age','Pclass'], axis=1)
df.head()
y_train = train['Survived']
x_train = df.loc[:y_train.shape[0]-1]
x_test = df.loc[y_train.shape[0]-1:]
x_test.drop(x_test.index[0],inplace = True)
df.head()
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import KFold,cross_val_score
import xgboost as xgb

models = []
models.append(('LR', LogisticRegression()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('RF', RandomForestClassifier(n_estimators=10)))
models.append(('SVM', SVC(kernel = 'linear')))
models.append(('XGB', xgb.XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.05)))
# evalutate each model in turn
results = []
names = []
for name, model in models:
    kfold = KFold(n_splits=10)
    cv_results = cross_val_score(model, x_train, y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)
model = xgb.XGBClassifier()
model.fit(x_train, y_train)
predictions = model.predict(x_test)
submission = pd.DataFrame({ 'PassengerId': test['PassengerId'],
                            'Survived': predictions })
submission.to_csv("submission.csv", index=False)
