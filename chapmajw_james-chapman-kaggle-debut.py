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

from sklearn.cross_validation import KFold

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import train_test_split

from sklearn.metrics import make_scorer, accuracy_score

from sklearn.model_selection import GridSearchCV



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.



train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')

combine = [train_df, test_df]



# preview the data

train_df.head()

test_df.head()

list(train_df.columns.values)
train_df.head()

test_df.head()
train_df.info()

print('_'*40)

test_df.info()
train_df.describe()

# Review survived rate using `percentiles=[.61, .62]` knowing our problem description mentions 38% survival rate.

# Review Parch distribution using `percentiles=[.75, .8]`

# SibSp distribution `[.68, .69]`

# Age and Fare `[.1, .2, .3, .4, .5, .6, .7, .8, .9, .99]`
train_df.describe(include=['O'])
#percentage survival rates for different categories. Just a simple mean the key bit being groupby



train_df[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train_df[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train_df[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)
g = sns.FacetGrid(train_df, col='Survived')

g.map(plt.hist, 'Age', bins=20)
grid = sns.FacetGrid(train_df, col='Pclass', hue='Survived')

grid.map(plt.hist, 'Age', alpha=.5, bins=20)

grid.add_legend();
# grid = sns.FacetGrid(train_df, col='Pclass', hue='Survived')

grid = sns.FacetGrid(train_df, col='SibSp', row='Sex')

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
#TITLES
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
embark_dummies_titanic  = pd.get_dummies(train_df['Title'])

embark_dummies_titanic.drop(['Rare'], axis=1, inplace=True)

train_df = train_df.join(embark_dummies_titanic)

train_df.drop(['Title'], axis=1,inplace=True)



embark_dummies_titanic  = pd.get_dummies(test_df['Title'])

embark_dummies_titanic.drop(['Rare'], axis=1, inplace=True)

test_df = test_df.join(embark_dummies_titanic)

test_df.drop(['Title'], axis=1,inplace=True)
train_df = train_df.drop(['Name', 'PassengerId'], axis=1)

test_df = test_df.drop(['Name'], axis=1)

combine = [train_df, test_df]

train_df.shape, test_df.shape
#SEX
embark_dummies_titanic  = pd.get_dummies(train_df['Sex'])

embark_dummies_titanic.drop(['male'], axis=1, inplace=True)

train_df = train_df.join(embark_dummies_titanic)

train_df.drop(['Sex'], axis=1,inplace=True)



embark_dummies_titanic  = pd.get_dummies(test_df['Sex'])

embark_dummies_titanic.drop(['male'], axis=1, inplace=True)

test_df = test_df.join(embark_dummies_titanic)

test_df.drop(['Sex'], axis=1,inplace=True)



combine = [train_df, test_df]

train_df.shape, test_df.shape
#AGES
guess_ages = np.zeros((2,3))

guess_ages
for dataset in combine:

    for i in range(0, 2):

        for j in range(0, 3):

            guess_df = dataset[(dataset['female'] == i) & (dataset['Pclass'] == j+1)]['Age'].dropna()

            

            # age_mean = guess_df.mean()

            # age_std = guess_df.std()

            # age_guess = rnd.uniform(age_mean - age_std, age_mean + age_std)

            

            age_guess = guess_df.mean()



            # Convert random age float to nearest .5 age

            guess_ages[i,j] = int( age_guess/0.5 + 0.5 ) * 0.5



          

    for i in range(0, 2):

        for j in range(0, 3):

            dataset.loc[ (dataset.Age.isnull()) & (dataset.female == i) & (dataset.Pclass == j+1),\

                    'Age'] = guess_ages[i,j]



    dataset['Age'] = dataset['Age'].astype(int)



train_df.head()
train_df['AgeBand'] = pd.cut(train_df['Age'], 5)

train_df[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True)
for dataset in combine:    

    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 1

    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 0

    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 0

    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 0

    dataset.loc[ dataset['Age'] > 64, 'Age'] = 0

train_df.head()
train_df = train_df.drop(['AgeBand'], axis=1)

combine = [train_df, test_df]

train_df.head()
for dataset in combine:

    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1



train_df[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train_df = train_df.drop(['Parch', 'SibSp'], axis=1)

test_df = test_df.drop(['Parch', 'SibSp'], axis=1)

combine = [train_df, test_df]



train_df.head()
freq_port = train_df.Embarked.dropna().mode()[0]

freq_port
#EMBARKED
for dataset in combine:

    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)

    

train_df[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False)
embark_dummies_titanic  = pd.get_dummies(train_df['Embarked'])

embark_dummies_titanic.drop(['C'], axis=1, inplace=True)

train_df = train_df.join(embark_dummies_titanic)

train_df.drop(['Embarked'], axis=1,inplace=True)



embark_dummies_titanic  = pd.get_dummies(test_df['Embarked'])

embark_dummies_titanic.drop(['C'], axis=1, inplace=True)

test_df = test_df.join(embark_dummies_titanic)

test_df.drop(['Embarked'], axis=1,inplace=True)



combine = [train_df, test_df]

train_df.shape, test_df.shape
test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True)

test_df.head()
train_df['FareBand'] = pd.qcut(train_df['Fare'], 4)

train_df[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand', ascending=True)
for dataset in combine:

    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 1

    dataset.loc[dataset['Fare'] > 7.91,  'Fare']   = 0

    dataset['Fare'] = dataset['Fare'].astype(int)



train_df = train_df.drop(['FareBand'], axis=1)



combine = [train_df, test_df]

    
#train_df.drop(['Fare'], axis=1,inplace=True)

#test_df.drop(['Fare'], axis=1,inplace=True)
X_all = train_df.drop(['Survived'], axis=1)

y_all = train_df['Survived']



num_test = 0.20

X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=num_test, random_state=23)
# Choose the type of classifier. 

clf = RandomForestClassifier(n_estimators=100)
def run_kfold(clf):

    kf = KFold(891, n_folds=10)

    outcomes = []

    fold = 0

    for train_index, test_index in kf:

        fold += 1

        X_train, X_test = X_all.values[train_index], X_all.values[test_index]

        y_train, y_test = y_all.values[train_index], y_all.values[test_index]

        clf.fit(X_train, y_train)

        predictions = clf.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)

        outcomes.append(accuracy)

        #print("Fold {0} accuracy: {1}".format(fold, accuracy))  

        mean_outcome = np.mean(outcomes)

        #print("Mean Accuracy: {0}".format(mean_outcome)) 

        return mean_outcome



acc_random_forest = run_kfold(clf)
# Logistic Regression

# Choose the type of classifier. 

clf = LogisticRegression()



# Type of scoring used to compare parameter combinations

acc_scorer = make_scorer(accuracy_score)



def run_kfold(clf):

    kf = KFold(891, n_folds=10)

    outcomes = []

    fold = 0

    for train_index, test_index in kf:

        fold += 1

        X_train, X_test = X_all.values[train_index], X_all.values[test_index]

        y_train, y_test = y_all.values[train_index], y_all.values[test_index]

        clf.fit(X_train, y_train)

        predictions = clf.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)

        outcomes.append(accuracy)

        #print("Fold {0} accuracy: {1}".format(fold, accuracy))     

    mean_outcome = np.mean(outcomes)

    #print("Mean Accuracy: {0}".format(mean_outcome)) 

    return mean_outcome



acc_log = run_kfold(clf)
# Gaussian Naive Bayes



# Choose the type of classifier. 

clf = GaussianNB()



# Type of scoring used to compare parameter combinations

acc_scorer = make_scorer(accuracy_score)



def run_kfold(clf):

    kf = KFold(891, n_folds=10)

    outcomes = []

    fold = 0

    for train_index, test_index in kf:

        fold += 1

        X_train, X_test = X_all.values[train_index], X_all.values[test_index]

        y_train, y_test = y_all.values[train_index], y_all.values[test_index]

        clf.fit(X_train, y_train)

        predictions = clf.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)

        outcomes.append(accuracy)

        #print("Fold {0} accuracy: {1}".format(fold, accuracy))     

    mean_outcome = np.mean(outcomes)

    #print("Mean Accuracy: {0}".format(mean_outcome)) 

    return mean_outcome



acc_gaussian = run_kfold(clf)
# KNN



# Choose the type of classifier. 

clf = KNeighborsClassifier()



# Type of scoring used to compare parameter combinations

acc_scorer = make_scorer(accuracy_score)



def run_kfold(clf):

    kf = KFold(891, n_folds=10)

    outcomes = []

    fold = 0

    for train_index, test_index in kf:

        fold += 1

        X_train, X_test = X_all.values[train_index], X_all.values[test_index]

        y_train, y_test = y_all.values[train_index], y_all.values[test_index]

        clf.fit(X_train, y_train)

        predictions = clf.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)

        outcomes.append(accuracy)

        #print("Fold {0} accuracy: {1}".format(fold, accuracy))     

    mean_outcome = np.mean(outcomes)

    #print("Mean Accuracy: {0}".format(mean_outcome)) 

    return mean_outcome



acc_knn = run_kfold(clf)
# Support Vector Machines



# Choose the type of classifier. 

clf = SVC()



# Type of scoring used to compare parameter combinations

acc_scorer = make_scorer(accuracy_score)



def run_kfold(clf):

    kf = KFold(891, n_folds=10)

    outcomes = []

    fold = 0

    for train_index, test_index in kf:

        fold += 1

        X_train, X_test = X_all.values[train_index], X_all.values[test_index]

        y_train, y_test = y_all.values[train_index], y_all.values[test_index]

        clf.fit(X_train, y_train)

        predictions = clf.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)

        outcomes.append(accuracy)

        #print("Fold {0} accuracy: {1}".format(fold, accuracy))     

    mean_outcome = np.mean(outcomes)

    #print("Mean Accuracy: {0}".format(mean_outcome)) 

    return mean_outcome



acc_svc = run_kfold(clf)
# Perceptron



# Choose the type of classifier. 

clf = Perceptron(max_iter=100, penalty='l2')



# Type of scoring used to compare parameter combinations

acc_scorer = make_scorer(accuracy_score)



def run_kfold(clf):

    kf = KFold(891, n_folds=10)

    outcomes = []

    fold = 0

    for train_index, test_index in kf:

        fold += 1

        X_train, X_test = X_all.values[train_index], X_all.values[test_index]

        y_train, y_test = y_all.values[train_index], y_all.values[test_index]

        clf.fit(X_train, y_train)

        predictions = clf.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)

        outcomes.append(accuracy)

        #print("Fold {0} accuracy: {1}".format(fold, accuracy))     

    mean_outcome = np.mean(outcomes)

    #print("Mean Accuracy: {0}".format(mean_outcome)) 

    return mean_outcome



acc_perceptron = run_kfold(clf)
# Linear SVC



# Choose the type of classifier. 

clf = LinearSVC()



# Type of scoring used to compare parameter combinations

acc_scorer = make_scorer(accuracy_score)



def run_kfold(clf):

    kf = KFold(891, n_folds=10)

    outcomes = []

    fold = 0

    for train_index, test_index in kf:

        fold += 1

        X_train, X_test = X_all.values[train_index], X_all.values[test_index]

        y_train, y_test = y_all.values[train_index], y_all.values[test_index]

        clf.fit(X_train, y_train)

        predictions = clf.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)

        outcomes.append(accuracy)

        #print("Fold {0} accuracy: {1}".format(fold, accuracy))     

    mean_outcome = np.mean(outcomes)

    #print("Mean Accuracy: {0}".format(mean_outcome)) 

    return mean_outcome



acc_linear_svc = run_kfold(clf)
# Stochastic Gradient Descent



# Choose the type of classifier. 

clf = SGDClassifier()



# Type of scoring used to compare parameter combinations

acc_scorer = make_scorer(accuracy_score)



def run_kfold(clf):

    kf = KFold(891, n_folds=10)

    outcomes = []

    fold = 0

    for train_index, test_index in kf:

        fold += 1

        X_train, X_test = X_all.values[train_index], X_all.values[test_index]

        y_train, y_test = y_all.values[train_index], y_all.values[test_index]

        clf.fit(X_train, y_train)

        predictions = clf.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)

        outcomes.append(accuracy)

        #print("Fold {0} accuracy: {1}".format(fold, accuracy))     

    mean_outcome = np.mean(outcomes)

    #print("Mean Accuracy: {0}".format(mean_outcome)) 

    return mean_outcome



acc_sgd = run_kfold(clf)
# Decision Tree



# Choose the type of classifier. 

clf = DecisionTreeClassifier()



# Type of scoring used to compare parameter combinations

acc_scorer = make_scorer(accuracy_score)



def run_kfold(clf):

    kf = KFold(891, n_folds=10)

    outcomes = []

    fold = 0

    for train_index, test_index in kf:

        fold += 1

        X_train, X_test = X_all.values[train_index], X_all.values[test_index]

        y_train, y_test = y_all.values[train_index], y_all.values[test_index]

        clf.fit(X_train, y_train)

        predictions = clf.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)

        outcomes.append(accuracy)

        #print("Fold {0} accuracy: {1}".format(fold, accuracy))     

    mean_outcome = np.mean(outcomes)

    #print("Mean Accuracy: {0}".format(mean_outcome)) 

    return mean_outcome



acc_decision_tree = run_kfold(clf)
models = pd.DataFrame({

    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 

              'Random Forest', 'Naive Bayes', 'Perceptron', 

              'Stochastic Gradient Decent', 'Linear SVC', 

              'Decision Tree'],

    'Score': [acc_svc, acc_knn, acc_log, 

              acc_random_forest, acc_gaussian, acc_perceptron, 

              acc_sgd, acc_linear_svc, acc_decision_tree]})

models.sort_values(by='Score', ascending=False)
svc = SVC()

svc.fit(X_all, y_all)
ids = test_df['PassengerId']

predictions = svc.predict(test_df.drop('PassengerId', axis=1))





output = pd.DataFrame({ 'PassengerId' : ids, 'Survived': predictions })

output.to_csv('titanic-predictions.csv', index = False)

output.head()