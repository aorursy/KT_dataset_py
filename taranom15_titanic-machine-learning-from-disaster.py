# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd
from pandas import get_dummies

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import matplotlib as mpl
from matplotlib import style

# Algorithms
import sklearn
from sklearn.linear_model import Perceptron, SGDClassifier, LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, cross_val_predict, cross_val_score
from sklearn.metrics import accuracy_score, log_loss, make_scorer, classification_report, confusion_matrix
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

import scipy

import warnings
import json
import sys
import csv
import os
import re
sns.set(style='white', context='notebook',palette='colorblind')
s_palette=['#7A7E85','#04BC50'] 
sex_palette=['#4178D8','#CD3747']
#pylab.rcParams['figure.figsize'] = 12,8
warnings.filterwarnings('ignore')
#mpl.style.use('ggplot')
#sns.set_style('white')
#%matplotlib inline
train = pd.read_csv('../input/train.csv')
print(train.head())
train.info()
print('*'*50)
print(train.describe())
print('%'*50)
print(train.describe(include=['O']))
print('&'*50)

corr = train.corr()
corr
test = pd.read_csv('../input/test.csv')
print(test.head(3))
print('*'*50)
print(test.columns)
print('*'*50)
test.info()
print('*'*50)
test.describe()
combine = [train, test]
pd.plotting.scatter_matrix(train,figsize=(15,15),color='#2C93E4', hist_kwds={'color':['#1F6AA5']})
plt.figure()
train.hist(figsize=(15,16), color ='#1F6AA5' )
plt.figure()
for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

train[["Title", "Sex",'PassengerId']].groupby(['Title','Sex']).count()
for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
 
train[["Title", "Sex",'PassengerId']].groupby(['Title','Sex']).count()
train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()
train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)
f,ax=plt.subplots(1,2,figsize=(15,5))

sns.swarmplot(x='Pclass',y='Age',data=train,ax=ax[0])

ax[1]= sns.boxplot(x="Pclass", y="Age", data=train)
ax[1]= sns.stripplot(x="Pclass", y="Age", data=train, jitter=True, edgecolor="gray")

plt.show()
_ = sns.FacetGrid(train, col='Survived', row = 'Pclass')
_.map(plt.hist, 'Age', alpha=0.75, bins=20)
_.add_legend()
_ = sns.FacetGrid(train, row='Pclass', col='Sex', height=2.2, aspect=1.6)
_.map(plt.hist, 'Age', alpha=0.75 ,bins=20)
_.add_legend()
_ = sns.FacetGrid(train, hue="Survived", col="Pclass", margin_titles=True,palette=s_palette )
_.map(plt.scatter, "Fare", "Age",edgecolor="w").add_legend();
#test['Fare'].fillna(test['Fare'].dropna().median(), inplace=True)     
test['Fare']=test['Fare'].fillna(0)

for dataset in combine:
    dataset['Fare'] = dataset['Fare'].astype(int)    
f,ax=plt.subplots(1,3,figsize=(20,5))
train["Age"].plot.hist(ax=ax[0],bins=20,edgecolor='black',color='#2C93E4');
ax[0].set_title('All')
x0=list(range(0,85,5))
ax[0].set_xticks(x0)

train[train['Survived']==0].Age.plot.hist(ax=ax[1],bins=20,edgecolor='black',color='#A5AEA9')
ax[1].set_title('Survived= 0')
x1=list(range(0,85,5))
ax[1].set_xticks(x1)

train[train['Survived']==1].Age.plot.hist(ax=ax[2],color='#04BC50',bins=20,edgecolor='black')
ax[2].set_title('Survived= 1')
x2=list(range(0,85,5))
ax[2].set_xticks(x2)

plt.show()
for dataset in combine:
    age_avg = dataset['Age'].mean()
    age_std = dataset['Age'].std()
    age_null_count = dataset['Age'].isnull().sum()
    
    age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)
    dataset['Age'][np.isnan(dataset['Age'])] = age_null_random_list
    dataset['Age'] = dataset['Age'].astype(int)
    
train['CategoricalAge'] = pd.cut(train['Age'], 5)

train[['CategoricalAge', 'Survived']].groupby(['CategoricalAge'], as_index=False).mean()
f,ax=plt.subplots(1,2,figsize=(15,5))
train['Survived'].value_counts().plot.pie(explode=[0,0.1],autopct='%1.1f%%',ax=ax[0],colors=s_palette)
ax[0].set_title('Survived')
ax[0].set_ylabel('')
sns.barplot(x='Sex', y='Survived', hue=None, data=train,ax=ax[1],palette=sex_palette)
plt.show()
f,ax=plt.subplots(1,2,figsize=(15,5))


sns.countplot('Sex',hue='Survived',data=train,ax=ax[0],palette=s_palette)
ax[0].set_title('Sex:Survived vs Dead')

sns.violinplot("Sex","Age", hue="Survived", data=train,split=True,ax=ax[1],palette=s_palette)
ax[1].set_title('Sex and Age vs Survived')
ax[1].set_yticks(range(0,110,10))

plt.show()
train[["Sex",'PassengerId']].groupby(['Sex'], as_index=False).count().sort_values(by='PassengerId', ascending=False)
train[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean()
fig, axes = plt.subplots(nrows=1, ncols=2,figsize=(10, 4))
women = train[train['Sex']=='female']
men = train[train['Sex']=='male']

ax = sns.distplot(women[women['Survived']==1].Age.dropna(), bins=18, label = 'survived', ax = axes[0], kde =False, color='#04BC50')
ax = sns.distplot(women[women['Survived']==0].Age.dropna(), bins=40, label = 'not survived', ax = axes[0], kde =False, color='#7A7E85')
ax.legend()
ax.set_title('Female')

ax = sns.distplot(men[men['Survived']==1].Age.dropna(), bins=18, label = 'survived', ax = axes[1], kde = False,color='#04BC50')
ax = sns.distplot(men[men['Survived']==0].Age.dropna(), bins=40, label = 'not survived', ax = axes[1], kde = False, color='#7A7E85')
ax.legend()
_ = ax.set_title('Male')
sns.scatterplot(y='Sex', x='Age', hue='Survived', data=train, palette=s_palette)
sns.pairplot(train, hue="Sex", palette=sex_palette)
sns.jointplot(x='Fare',y='Age',data=train,color='#2C93E4',kind='reg')
train[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)
for dataset in combine:
    dataset['Relatives'] = dataset['SibSp'] + dataset['Parch']
    dataset.loc[dataset['Relatives'] > 0, 'NotAlone'] = 0
    dataset.loc[dataset['Relatives'] == 0, 'NotAlone'] = 1
    dataset['NotAlone'] = dataset['NotAlone'].astype(int)
train[['Relatives', 'Survived']].groupby(['Relatives'], as_index=False).mean().sort_values(by='Survived', ascending=False)
_ = sns.factorplot('Relatives','Survived', data=train, aspect = 2.5, )
train[['NotAlone', 'Survived']].groupby(['NotAlone'], as_index=False).mean()
#train["Deck"]=train.Cabin.str[0]
#test["Deck"]=test.Cabin.str[0]
import re
deck = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "U": 8}

for dataset in combine:
    dataset['Cabin'] = dataset['Cabin'].fillna("U0")
    dataset['Deck'] = dataset['Cabin'].map(lambda x: re.compile("([a-zA-Z]+)").search(x).group())
    dataset['Deck'] = dataset['Deck'].map(deck)
    dataset['Deck'] = dataset['Deck'].fillna(0)
    dataset['Deck'] = dataset['Deck'].astype(int) 
train["Deck"].unique()
sns.barplot(x='Deck', y= "Survived", hue=None, data=train)
sns.violinplot("Deck","Age", hue="Survived", data=train,split=True,palette=s_palette)
train[['Deck', 'Survived']].groupby(['Deck'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train[['Deck', 'PassengerId']].groupby(['Deck'], as_index=False).count()
_ = sns.FacetGrid(train, col="Pclass", sharex=False, gridspec_kws={"width_ratios": [5, 3, 3]})
_.map(sns.boxplot, "Deck", "Age");
_ = sns.FacetGrid(train, row='Embarked', height=2.2, aspect=1.6)
_.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep', order=None, hue_order=None)
_.add_legend()
train[["Embarked", "Survived"]].groupby(['Embarked'], as_index=False).count().sort_values(by='Survived', ascending=False)
freq_port = train.Embarked.dropna().mode()[0]

for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)

train[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).count().sort_values(by='Survived', ascending=False)
_ = sns.FacetGrid(train, row='Embarked', col='Survived', height=2.2, aspect=1.6)
_.map(sns.barplot, 'Sex', 'Fare', alpha=.5, ci=None, order=None)
_.add_legend()
train = train.drop(['Ticket', 'Cabin', 'Name', 'PassengerId','CategoricalAge'], axis=1)
test = test.drop(['Ticket', 'Cabin', 'Name'], axis=1)
combine = [train, test]

train.head()
test.head()
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)
train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()
combine = [train, test]

for dataset in combine:
    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)
train.head()
for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

train.head()
combine = [train, test]

for dataset in combine:
    dataset['Age'] = dataset['Age'].astype(int)
    dataset.loc[ dataset['Age'] <= 11, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 11) & (dataset['Age'] <= 18), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 18) & (dataset['Age'] <= 22), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 22) & (dataset['Age'] <= 27), 'Age'] = 3
    dataset.loc[(dataset['Age'] > 27) & (dataset['Age'] <= 33), 'Age'] = 4
    dataset.loc[(dataset['Age'] > 33) & (dataset['Age'] <= 40), 'Age'] = 5
    dataset.loc[(dataset['Age'] > 40) & (dataset['Age'] <= 66), 'Age'] = 6
    dataset.loc[ dataset['Age'] > 66, 'Age'] = 6
train['Age'].value_counts()
for dataset in combine:
    dataset['Age*Class'] = dataset.Age * dataset.Pclass

train.loc[:, ['Age*Class', 'Age', 'Pclass']].head(10)
for dataset in combine:
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
#    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3
    dataset.loc[(dataset['Fare'] > 31) & (dataset['Fare'] <= 99), 'Fare']   = 3
    dataset.loc[(dataset['Fare'] > 99) & (dataset['Fare'] <= 250), 'Fare']   = 4
    dataset.loc[ dataset['Fare'] > 250, 'Fare'] = 5
    dataset['Fare'] = dataset['Fare'].astype(int)

for dataset in combine:
    dataset['Fare_Per_Person'] = dataset['Fare']/(dataset['Relatives']+1)
    dataset['Fare_Per_Person'] = dataset['Fare_Per_Person'].astype(int)
combine=[train,test]
train.head(10)
test.head(10)
train.corr()["Survived"]
X_train = train.drop("Survived", axis=1)
y_train = train["Survived"]
   
X_test  = test.drop("PassengerId", axis=1).copy()
X_train.shape, y_train.shape, X_test.shape
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
acc_log = round(logreg.score(X_train, y_train) * 100, 2)
acc_log
coeff = pd.DataFrame(train.columns.delete(0))
coeff.columns = ['Feature']
coeff["Correlation"] = pd.Series(logreg.coef_[0])

coeff.sort_values(by='Correlation', ascending=False)
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, y_train)
Y_pred = knn.predict(X_test)
acc_knn = round(knn.score(X_train, y_train) * 100, 2)
acc_knn
svc = SVC()
svc.fit(X_train, y_train)
Y_pred = svc.predict(X_test)
acc_svc = round(svc.score(X_train, y_train) * 100, 2)
acc_svc
gaussian = GaussianNB()
gaussian.fit(X_train, y_train)
y_pred = gaussian.predict(X_test)
acc_gaussian = round(gaussian.score(X_train, y_train) * 100, 2)
acc_gaussian
linear_svc = LinearSVC()
linear_svc.fit(X_train, y_train)
y_pred = linear_svc.predict(X_test)
acc_linear_svc = round(linear_svc.score(X_train, y_train) * 100, 2)
acc_linear_svc
sgd = SGDClassifier()
sgd.fit(X_train, y_train)
y_pred = sgd.predict(X_test)
acc_sgd = round(sgd.score(X_train, y_train) * 100, 2)
acc_sgd
perceptron = Perceptron()
perceptron.fit(X_train, y_train)
y_pred = perceptron.predict(X_test)
acc_perceptron = round(perceptron.score(X_train, y_train) * 100, 2)
acc_perceptron
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, y_train)
y_pred = decision_tree.predict(X_test)
acc_decision_tree = round(decision_tree.score(X_train, y_train) * 100, 2)
acc_decision_tree
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, y_train)
y_pred = random_forest.predict(X_test)
random_forest.score(X_train, y_train)
acc_random_forest = round(random_forest.score(X_train, y_train) * 100, 2)
acc_random_forest
models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 'Random Forest', 'Naive Bayes', 'Perceptron', 'Stochastic Gradient Decent', 'Linear SVC', 'Decision Tree'],
    'Score': [acc_svc, acc_knn, acc_log, acc_random_forest, acc_gaussian, acc_perceptron, acc_sgd, acc_linear_svc, acc_decision_tree]})
models.sort_values(by='Score', ascending=False)
rf = RandomForestClassifier(n_estimators=100)
scores = cross_val_score(rf, X_train, y_train, cv=5, scoring = "accuracy")
print("Scores:", scores)
print("Mean:", scores.mean())
print("Standard Deviation:", scores.std())
predictions = cross_val_predict(random_forest, X_train, y_train, cv=5)
confusion_matrix(y_train, predictions)
rf = RandomForestClassifier(n_estimators=100)
scores = cross_val_score(rf, X_train, y_train, cv=10, scoring = "accuracy")
print("Scores:", scores)
print("Mean:", scores.mean())
print("Standard Deviation:", scores.std())
predictions = cross_val_predict(random_forest, X_train, y_train, cv=10)
confusion_matrix(y_train, predictions)
importances = pd.DataFrame({'feature':X_train.columns,'importance':np.round(random_forest.feature_importances_,3)})
importances = importances.sort_values('importance',ascending=False).set_index('feature')
importances.head(15)
importances.plot.bar()
train = train.drop("NotAlone", axis=1)
test = test.drop("NotAlone", axis=1)

train = train.drop("Parch", axis=1)
test = test.drop("Parch", axis=1)

X_train = train.drop("Survived", axis=1)
y_train = train["Survived"]
X_test  = test.drop("PassengerId", axis=1).copy()
X_train.shape, y_train.shape, X_test.shape
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, y_train)
y_pred = random_forest.predict(X_test)
random_forest.score(X_train, y_train)
acc_random_forest = round(random_forest.score(X_train, y_train) * 100, 2)
acc_random_forest
rf = RandomForestClassifier(n_estimators=100)
scores = cross_val_score(rf, X_train, y_train, cv=5, scoring = "accuracy")
print("Scores:", scores)
print("Mean:", scores.mean())
print("Standard Deviation:", scores.std())
predictions = cross_val_predict(random_forest, X_train, y_train, cv=5)
confusion_matrix(y_train, predictions)
importances = pd.DataFrame({'feature':X_train.columns,'importance':np.round(random_forest.feature_importances_,3)})
importances = importances.sort_values('importance',ascending=False).set_index('feature')
importances.head(15)
importances.plot.bar()
clf.best_params_
submission = pd.DataFrame({ "PassengerId": test["PassengerId"], "Survived": y_prediction })
print(submission.head())
submission.to_csv('submissionrf2.csv', index=False)