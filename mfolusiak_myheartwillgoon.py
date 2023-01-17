# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



# data analysis and wrangling

import pandas as pd

import numpy as np

import random as rnd



# visualization

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



# machine learning

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

#print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.



train_df = pd.read_csv('../input/train.csv')

test_df  = pd.read_csv('../input/test.csv')

combine = [train_df,test_df]



print(train_df.columns.values)
train_df.head()

train_df.tail()
train_df.info()

print('_'*40)

test_df.info()

print('_'*40)
train_df.sort_values(by='Survived').describe(percentiles=[.61, .62, .63])
train_df.sort_values(by='Parch').describe(percentiles=[.75, .80])
train_df.sort_values(by='SibSp').describe(percentiles=[.68, .69])
train_df.sort_values(by='Age').describe(percentiles=[.1, .2, .3, .4, .5, .6, .7, .8, .9, .99])
train_df.sort_values(by='Fare').describe(percentiles=[.1, .2, .3, .4, .5, .6, .7, .8, .9, .99])
train_df.Name.unique().size == train_df.Name.size
train_df.describe(include=['O'])
train_df.describe()
train_df[['Pclass','Survived']].groupby('Pclass', as_index=False).mean().sort_values(by='Survived', ascending=False)
train_df[['Sex','Survived']].groupby('Sex', as_index=False).mean().sort_values(by='Survived', ascending=False)
train_df[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train_df[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)
g = sns.FacetGrid(train_df, col='Survived')

g.map(plt.hist, 'Age', bins=20)
grid = sns.FacetGrid(train_df, col='Survived', row='Pclass', size=2.2, aspect=1.6)

grid.map(plt.hist, 'Age', alpha=0.5, bins=20)

grid.add_legend();
# grid = sns.FacetGrid(train_df, col='Embarked')

grid = sns.FacetGrid(train_df, row='Embarked', size=2.2, aspect=1.6)

grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')

grid.add_legend()
grid = sns.FacetGrid(train_df, row='Embarked', col='Survived', size=2.2, aspect=1.6)

grid.map(sns.barplot, 'Sex', 'Fare', alpha=.5, ci=None)

grid.add_legend()
print("Before", train_df.shape, test_df.shape, combine[0].shape, combine[1].shape)

train_df = train_df.drop(['Ticket','Cabin'],axis=1)

test_df = test_df.drop(['Ticket','Cabin'],axis=1)

combine = [train_df, test_df]

print("After", train_df.shape, test_df.shape, combine[0].shape, combine[1].shape)
for dataset in combine:

    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)



pd.crosstab(train_df['Title'], train_df['Sex'])
for dataset in combine:

    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',\

 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')



    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

    

train_df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()

title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

for dataset in combine:

    dataset['Title'] = dataset['Title'].map(title_mapping)

    dataset['Title'] = dataset['Title'].fillna(0)



train_df.head()
train_df = train_df.drop(['Name', 'PassengerId'], axis=1)

test_df = test_df.drop(['Name'], axis=1)

combine = [train_df, test_df]

train_df.shape, test_df.shape
for dataset in combine:

    dataset['Sex'] = dataset['Sex'].map({'female':1, 'male':0}).astype(int)



train_df.head()
grid = sns.FacetGrid(train_df, row='Pclass', col='Sex', size=2.2, aspect=1.6)

grid.map(plt.hist, 'Age', alpha=.5, bins=20)

grid.add_legend()
guess_ages = np.zeros((2,3))

guess_ages
for dataset in combine: # each of two datasets!

    for i in range(0,2):

        for j in range(0,3):

            guess_age_df = dataset[(dataset['Sex'] == i) & (dataset['Pclass'] == j+1)]['Age'].dropna()

            ageguess = guess_age_df.median()

            # convert to nearest 0.5

            guess_ages[i][j] = int(ageguess/0.5 +0.5) *0.5

    for i in range(0,2):

        for j in range(0,3):

            dataset.loc[ (dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j+1),'Age' ] = guess_ages[i][j]

    dataset['Age'] = dataset['Age'].astype(int)

    

train_df.head()
train_df['AgeBand'] = pd.cut(train_df['Age'],10)

train_df[['AgeBand','Survived']].groupby('AgeBand',as_index=False).mean().sort_values(by='AgeBand', ascending=True)
for dataset in combine:

    dataset.loc[                     dataset['Age']<= 8 , 'Age'] = 0

    dataset.loc[(dataset['Age']> 8)&(dataset['Age']<=16), 'Age'] = 1

    dataset.loc[(dataset['Age']>16)&(dataset['Age']<=24), 'Age'] = 2

    dataset.loc[(dataset['Age']>24)&(dataset['Age']<=32), 'Age'] = 3

    dataset.loc[(dataset['Age']>32)&(dataset['Age']<=40), 'Age'] = 4

    dataset.loc[(dataset['Age']>40)&(dataset['Age']<=48), 'Age'] = 5

    dataset.loc[(dataset['Age']>48)&(dataset['Age']<=56), 'Age'] = 6

    dataset.loc[(dataset['Age']>56)&(dataset['Age']<=64), 'Age'] = 7

    dataset.loc[(dataset['Age']>64)&(dataset['Age']<=72), 'Age'] = 8

    dataset.loc[(dataset['Age']>72)                     , 'Age'] = 9

train_df.head()
train_df = train_df.drop(['AgeBand'], axis=1)

combine = [train_df, test_df]

train_df.head()
train_df.groupby('Age').mean()
for dataset in combine:

    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1



train_df[['FamilySize','Survived']].groupby(['FamilySize'],as_index=False).mean().sort_values(by='Survived',ascending=False)

#train_df[['FamilySize','Survived']].groupby(['FamilySize'],as_index=False).count()
for dataset in combine:

    dataset['IsAlone'] = 0

    dataset.loc[ dataset['FamilySize'] == 1 , 'IsAlone' ] = 1



train_df[['IsAlone','Survived']].groupby(['IsAlone'],as_index=False).mean()
train_df = train_df.drop(['Parch','SibSp','FamilySize'],axis=1)

test_df  = test_df.drop (['Parch','SibSp','FamilySize'],axis=1)

combine = [train_df, test_df]
for dataset in combine:

    dataset['Age*Class'] = dataset['Age'] * dataset['Pclass']



train_df.loc[:,['Age*Class','Age','Pclass']].head(10)
freq_port = train_df.Embarked.dropna().mode()[0]

freq_port
for dataset in combine:

    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)



train_df[['Embarked', 'Survived']].groupby(['Embarked'],as_index=False).mean().sort_values(by='Survived',ascending=False)
for dataset in combine:

    dataset['Embarked'] = dataset['Embarked'].map( {'S':0, 'C':1, 'Q':2} ).astype(int)



train_df.head()
test_df.Fare.fillna(test_df.Fare.dropna().median(), inplace=True)

test_df.head()
train_df['FareBand'] = pd.qcut(train_df['Fare'], 4)

train_df[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand', ascending=True)
for dataset in combine:

    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0

    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1

    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2

    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3

    dataset['Fare'] = dataset['Fare'].astype(int)



train_df = train_df.drop(['FareBand'], axis=1)

combine = [train_df, test_df]

    

train_df.head(10)
test_df.head(10)
X_train = train_df.drop("Survived", axis=1)

Y_train = train_df["Survived"]

X_test  = test_df.drop("PassengerId", axis=1).copy()

X_train.shape, Y_train.shape, X_test.shape
# Logistic regression

logreg = LogisticRegression()

logreg.fit(X_train,Y_train)

acc_log = logreg.score(X_train,Y_train)

acc_log
corr_coef= pd.DataFrame(train_df.columns.delete(0),columns=['Feature'])

corr_coef['Correlation'] = pd.Series(logreg.coef_[0])

corr_coef.sort_values(by='Correlation',ascending=False)
# Support vector machines

svc = SVC()

svc.fit(X_train,Y_train)

acc_svc = svc.score(X_train,Y_train)

acc_svc
# k nearest neighbors - primitive lazy learning technique

knn = KNeighborsClassifier()

knn.fit(X_train,Y_train)

acc_knn = knn.score(X_train,Y_train)

acc_knn
# Gaussian Naive Bayes

gaussian = GaussianNB()

gaussian.fit(X_train, Y_train)

acc_gaussian = gaussian.score(X_train, Y_train)

acc_gaussian
# Perceptron

perceptron = Perceptron()

perceptron.fit(X_train, Y_train)

acc_perceptron = perceptron.score(X_train, Y_train)

acc_perceptron
# Stochastic Gradient Descent

sgd = SGDClassifier()

sgd.fit(X_train, Y_train)

acc_sgd = sgd.score(X_train, Y_train)

acc_sgd
# Decision Tree

decision_tree = DecisionTreeClassifier()

decision_tree.fit(X_train, Y_train)

acc_decision_tree = decision_tree.score(X_train, Y_train)

acc_decision_tree
#Random Forest

random_forest = RandomForestClassifier()

random_forest.fit(X_train,Y_train)

Y_pred = random_forest.predict(X_test)

acc_random_forest = random_forest.score(X_train,Y_train)

acc_random_forest
# Evaluate models

models = pd.DataFrame({

    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 

              'Random Forest', 'Naive Bayes', 'Perceptron', 

              'Stochastic Gradient Decent', 'Linear SVC', 

              'Decision Tree'],

    'Score': [acc_svc, acc_knn, acc_log, 

              acc_random_forest, acc_gaussian, acc_perceptron, 

              acc_sgd, acc_svc, acc_decision_tree]})

models.sort_values(by='Score', ascending=False)
submission = pd.DataFrame.from_dict({'PassengerId': test_df['PassengerId'],'Survived' : Y_pred})

submission
# Submit

#submission.to_csv('../output/submission.csv',index=False)