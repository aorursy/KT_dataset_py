# data analysis and wrangling

import pandas as pd

import numpy as np

import random as rnd

\



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

from xgboost import XGBClassifier
train_df = pd.read_csv('/kaggle/input/train.csv')

test_df = pd.read_csv('/kaggle/input/test.csv')

combine = [train_df, test_df]
print(train_df.columns.values)
# preview the data

train_df.head()
train_df.tail()
train_df.info()

print('_'*40)

test_df.info()
train_df.describe()

# Review survived rate using `percentiles=[.61, .62]` knowing our problem description mentions 38% survival rate.

# Review Parch distribution using `percentiles=[.75, .8]`

# SibSp distribution `[.68, .69]`

# Age and Fare `[.1, .2, .3, .4, .5, .6, .7, .8, .9, .99]`
train_df.describe(include=['O'])
train_df[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train_df[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train_df[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train_df[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)
g = sns.FacetGrid(train_df, col='Survived')

g.map(plt.hist, 'Age', bins=20)
# grid = sns.FacetGrid(train_df, col='Pclass', hue='Survived')

grid = sns.FacetGrid(train_df, col='Survived', row='Pclass', size=2.2, aspect=1.6)

grid.map(plt.hist, 'Age', alpha=.5, bins=20)

grid.add_legend();
# grid = sns.FacetGrid(train_df, col='Embarked')

grid = sns.FacetGrid(train_df, row='Embarked', size=2.2, aspect=1.6)

grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')

grid.add_legend()
# grid = sns.FacetGrid(train_df, col='Embarked', hue='Survived', palette={0: 'k', 1: 'w'})

grid = sns.FacetGrid(train_df, row='Embarked', col='Survived', size=2.2, aspect=1.6)

grid.map(sns.barplot, 'Sex', 'Fare', alpha=.5, ci=None)

grid.add_legend()
print("Before", train_df.shape, test_df.shape, combine[0].shape, combine[1].shape)



train_df = train_df.drop(['Ticket', 'Cabin'], axis=1)

test_df = test_df.drop(['Ticket', 'Cabin'], axis=1)

combine = [train_df, test_df]



"After", train_df.shape, test_df.shape, combine[0].shape, combine[1].shape
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

    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)



train_df.head()
# grid = sns.FacetGrid(train_df, col='Pclass', hue='Gender')

grid = sns.FacetGrid(train_df, row='Pclass', col='Sex', size=2.2, aspect=1.6)

grid.map(plt.hist, 'Age', alpha=.5, bins=20)

grid.add_legend()
guess_ages = np.zeros((2,3))

guess_ages
for dataset in combine:

    for i in range(0, 2):

        for j in range(0, 3):

            guess_df = dataset[(dataset['Sex'] == i) & \

                                  (dataset['Pclass'] == j+1)]['Age'].dropna()



            # age_mean = guess_df.mean()

            # age_std = guess_df.std()

            # age_guess = rnd.uniform(age_mean - age_std, age_mean + age_std)



            age_guess = guess_df.median()



            # Convert random age float to nearest .5 age

            guess_ages[i,j] = int( age_guess/0.5 + 0.5 ) * 0.5

            

    for i in range(0, 2):

        for j in range(0, 3):

            dataset.loc[ (dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j+1),\

                    'Age'] = guess_ages[i,j]



    dataset['Age'] = dataset['Age'].astype(int)

##full['Fare'] = full.groupby("Pclass")['Fare'].transform(lambda x: x.fillna(x.median()))

train_df.head()
#version 20 cut 4 instead 3     .775

#version 22 qcut 3              .784

train_df['AgeBand'] = pd.qcut(train_df['Age'], 3)

train_df[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True)
for dataset in combine:    

    dataset.loc[ dataset['Age'] <= 24, 'Age'] = 0

    dataset.loc[(dataset['Age'] > 24) & (dataset['Age'] <= 32), 'Age'] = 1

    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 80), 'Age'] = 2

  #  dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3

 #   dataset.loc[ dataset['Age'] > 60, 'Age'] = 3

train_df.head()
train_df = train_df.drop(['AgeBand'], axis=1)

combine = [train_df, test_df]

train_df.head()
#version 5 returnning IsAlone deleting SibSp and Parch

for i in combine:#dataset in combine:

    #dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

    i['Fam_Size'] = np.where((i['SibSp']+i['Parch']) == 0 , 0,#'Solo',

                           np.where((i['SibSp']+i['Parch']) <= 3,1,2))#'Nuclear', 'Big'))

    del i['SibSp']

    del i['Parch']

#train_df[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived', ascending=False)
#for dataset in combine:

#    dataset['IsAlone'] = 0

#    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1



#train_df[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean()
#version 26 drop familySize

#version 27 drop IsAlone,SibSp, Parch return FamilySize   .760

#version 28 drop IsAlone return SibSp, Parch        .789

#train_df = train_df.drop([ 'IsAlone'], axis=1)  #['Parch', 'SibSp', 'FamilySize'], axis=1)

#test_df = test_df.drop([ 'IsAlone'], axis=1)

#combine = [train_df, test_df]



train_df.head()
#for dataset in combine:

#    dataset['Age*Class'] = dataset.Age * dataset.Pclass



#train_df.loc[:, ['Age*Class', 'Age', 'Pclass']].head(10)
freq_port = train_df.Embarked.dropna().mode()[0]

freq_port
for dataset in combine:

    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)

    

train_df[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False)
for dataset in combine:

    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)



train_df.head()
test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True)

test_df.head()
#version 19 cut 5 instead 3     .794

#version 23 cut 4               .799

train_df['FareBand'] = pd.cut(train_df['Fare'], 4)

train_df[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand', ascending=True)
for dataset in combine:

    dataset.loc[ dataset['Fare'] <= 128, 'Fare'] = 0

    dataset.loc[(dataset['Fare'] > 128) & (dataset['Fare'] <= 256.2), 'Fare'] = 1

    dataset.loc[(dataset['Fare'] > 265.2) & (dataset['Fare'] <= 384.3), 'Fare']   = 2

   # dataset.loc[(dataset['Fare'] > 307.4) & (dataset['Fare'] <= 410), 'Fare']   = 3

    dataset.loc[ dataset['Fare'] > 384.3, 'Fare'] = 3

    dataset['Fare'] = dataset['Fare'].astype(int)



train_df = train_df.drop(['FareBand'], axis=1)

combine = [train_df, test_df]

    

train_df.head(10)
#version 4

train_df = train_df.drop(['Embarked', 'Title'], axis=1)

test_df = test_df.drop(['Embarked', 'Title'], axis=1)

combine = [train_df, test_df]



train_df.head()
test_df.head(10)
#version 5 dropped Pclass , Fare  returned IsAlone    .76

#version 6 like version 5    .799

#version 7 returned Pclass

#version 9 returned Fare with cut of 3 not qcut 

#version 10 +drop Pclass    .799

#version 11 +drop Age       .779

#version 12 +drop Fare      .765

#version 13 returned Age with cut 4     .779

#version 14 Age cut 5       .773

#version 15 Age cut 3 and return Fare    .799  same case as version 6

#version 16 drop Pclass, SibSp, Parch, Age  .765

#version 17 return Age   .765

#version 18 return to the best state dropping only Pclass   .799

train_df = train_df.drop(['Pclass'], axis=1)   #['Pclass', 'Fare'], axis=1)

test_df = test_df.drop(['Pclass'], axis=1)

combine = [train_df, test_df]



train_df.head()
X_train = train_df.drop("Survived", axis=1)

Y_train = train_df["Survived"]

X_test  = test_df.drop("PassengerId", axis=1).copy()

X_train.shape, Y_train.shape, X_test.shape
# Logistic Regression



logreg = LogisticRegression()

logreg.fit(X_train, Y_train)

Y_pred = logreg.predict(X_test)

acc_log = round(logreg.score(X_train, Y_train) * 100, 2)

acc_log
coeff_df = pd.DataFrame(train_df.columns.delete(0))

coeff_df.columns = ['Feature']

coeff_df["Correlation"] = pd.Series(logreg.coef_[0])



coeff_df.sort_values(by='Correlation', ascending=False)
# Support Vector Machines



svc = SVC()

svc.fit(X_train, Y_train)

Y_pred = svc.predict(X_test)

acc_svc = round(svc.score(X_train, Y_train) * 100, 2)

acc_svc
knn = KNeighborsClassifier(n_neighbors = 3)

knn.fit(X_train, Y_train)

Y_pred = knn.predict(X_test)

acc_knn = round(knn.score(X_train, Y_train) * 100, 2)

acc_knn
# Gaussian Naive Bayes



gaussian = GaussianNB()

gaussian.fit(X_train, Y_train)

Y_pred = gaussian.predict(X_test)

acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)

acc_gaussian
# Perceptron



perceptron = Perceptron()

perceptron.fit(X_train, Y_train)

Y_pred = perceptron.predict(X_test)

acc_perceptron = round(perceptron.score(X_train, Y_train) * 100, 2)

acc_perceptron
# Linear SVC



linear_svc = LinearSVC()

linear_svc.fit(X_train, Y_train)

Y_pred = linear_svc.predict(X_test)

acc_linear_svc = round(linear_svc.score(X_train, Y_train) * 100, 2)

acc_linear_svc
# Stochastic Gradient Descent



sgd = SGDClassifier()

sgd.fit(X_train, Y_train)

Y_pred = sgd.predict(X_test)

acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)

acc_sgd
# Decision Tree



decision_tree = DecisionTreeClassifier(random_state = 0, criterion="gini", max_depth = 13)

decision_tree.fit(X_train, Y_train)

Y_pred = decision_tree.predict(X_test)

acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)

acc_decision_tree
# Random Forest

random_forest = RandomForestClassifier(criterion='gini', #'entropy',

                             n_estimators=500,

                             max_depth=10,

                             min_samples_split=5,#10,

                             min_samples_leaf=1,

                             max_features='auto',

                             oob_score=True,

                             random_state=2,

                             n_jobs=-1)

#random_forest = RandomForestClassifier(random_state = 2, criterion = 'gini', max_depth= 10, max_features = 'auto', n_estimators=500)

#random_forest = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=1)

random_forest.fit(X_train, Y_train)

Y_pred = random_forest.predict(X_test)

random_forest.score(X_train, Y_train)

acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)

acc_random_forest

# Set our parameter grid

#param_grid = { 

#    'criterion' : ['gini', 'entropy'],

#    'n_estimators': [100, 300, 500],

#    'max_features': ['auto', 'log2'],

#    'max_depth' : [3, 5, 7]    

#}

#from sklearn.model_selection import GridSearchCV

#randomForest = RandomForestClassifier(random_state = 2)

## Grid search

#randomForest_CV = GridSearchCV(estimator = randomForest, param_grid = param_grid, cv = 5)

#randomForest_CV.fit(X_train, Y_train)

## Print best hyperparameters

#randomForest_CV.best_params_

# {'criterion': 'entropy',

# 'max_depth': 7,

# 'max_features': 'auto',

# 'n_estimators': 100} ''
XGBClassifier

xg = XGBClassifier(n_estimators=300, max_depth=13, random_state=2)

xg.fit(X_train, Y_train)

Y_pred_xg = xg.predict(X_test)

xg.score(X_train, Y_train)

acc_xg = round(xg.score(X_train, Y_train) * 100, 2)

acc_xg
models = pd.DataFrame({

    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 

              'Random Forest', 'Naive Bayes', 'Perceptron', 

              'Stochastic Gradient Decent', 'Linear SVC', 

              'Decision Tree'],

    'Score': [acc_svc, acc_knn, acc_log, 

              acc_random_forest, acc_gaussian, acc_perceptron, 

              acc_sgd, acc_linear_svc, acc_decision_tree]})

models.sort_values(by='Score', ascending=False)
submission = pd.DataFrame({

        "PassengerId": test_df["PassengerId"],

        "Survived": Y_pred

    })

submission.to_csv('submission.csv', index=False)