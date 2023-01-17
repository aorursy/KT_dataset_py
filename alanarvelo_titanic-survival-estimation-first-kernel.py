# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))





# Any results you write to the current directory are saved as output.

# data analysis and wrangling

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
train_df = pd.read_csv("../input/train.csv")

test_df = pd.read_csv("../input/test.csv")

gender_sub_df = pd.read_csv("../input/gender_submission.csv")

train_df.shape, test_df.shape, gender_sub_df.shape

combine = [train_df, test_df]
print(train_df.columns.values)
train_df.head()
train_df.info()

print('_'*50)

test_df.info()
train_df.describe()
train_df.describe(include=['O'])
test_df.describe(include=['O'])
train_df[['Pclass', 'Survived']].groupby(["Pclass"], as_index=False).mean().sort_values(by='Survived', ascending=False)
train_df[['Sex', 'Survived']].groupby(["Sex"], as_index=False).mean().sort_values(by='Survived', ascending=False)
train_df[['SibSp', 'Survived']].groupby(["SibSp"], as_index=False).mean().sort_values(by='Survived', ascending=False)
train_df[['Parch', 'Survived']].groupby(["Parch"], as_index=False).mean().sort_values(by='Survived', ascending=False)
g=sns.FacetGrid(train_df, col="Survived")

g.map(plt.hist, 'Age', bins=20)
grid =sns.FacetGrid(train_df, col="Survived", row='Pclass', size=2.2, aspect=1.6)

grid.map(plt.hist, 'Age', bins=20, alpha=.5)

grid.add_legend()
grid = sns.FacetGrid(train_df, row = 'Embarked', size=2.2, aspect=1.6)

grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')

grid.add_legend()
grid = sns.FacetGrid(train_df, col='Survived', row='Embarked', size=2.2, aspect=1.6)

grid.map(sns.barplot, 'Sex', 'Fare', alpha=.5, ci=None)

grid.add_legend()
print("Before", train_df.shape, test_df.shape, combine[0].shape, combine[1].shape)



train_df.drop(['Cabin', 'Ticket'], axis=1, inplace=True)

test_df.drop(['Cabin', 'Ticket'], axis=1, inplace=True)

combine = [train_df, test_df]



print("After", train_df.shape, test_df.shape, combine[0].shape, combine[1].shape)
for dataset in combine:

    dataset["Title"] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

    

pd.crosstab(train_df.Title, train_df.Sex)
train_df.Title.unique()
for dataset in combine:

    dataset["Title"].replace(['Don', 'Rev', 'Dr', 'Major', 'Lady', 'Sir', 'Col', 'Capt',

       'Countess', 'Jonkheer'], 'Rare', inplace=True)

    

    dataset["Title"].replace('Ms', 'Miss', inplace=True)

    dataset["Title"].replace('Mlle', 'Miss', inplace=True)

    dataset["Title"].replace('Mme', 'Mrs', inplace=True)

    

train_df[["Title", "Survived"]].groupby(["Title"], as_index=False).mean()
train_df.head()
title_mapping = {'Mr':1, 'Miss':2, 'Mrs':3, 'Master':4, 'Rare':5}

for dataset in combine:

    dataset.Title = dataset.Title.map(title_mapping)

    dataset.Title.fillna(0, inplace=True)
train_df.head()
train_df.drop(["PassengerId", "Name"], axis=1, inplace=True)

test_df.drop(["Name"], axis=1, inplace=True)
combine = [train_df, test_df]

train_df.shape, test_df.shape
for dataset in combine:

    dataset['Sex'] = dataset['Sex'].map( {'female':1, 'male':0} ).astype(int)
grid = sns.FacetGrid(train_df, row='Pclass', col='Sex', size=2.2, aspect=1.6)

grid.map(plt.hist, 'Age', alpha=.5, bins=20)

grid.add_legend()
guess_ages = np.zeros([2, 3])

guess_ages
for dataset in combine:

    for i in range(0,2):

        for j in range(0,3):

            guess_df = dataset[ (dataset['Sex'] == i) & \

                               (dataset['Pclass'] == j+1) ][ 'Age' ].dropna()

            

            age_guess = guess_df.median()

            

            #convert random age float to nearest .5 age

            guess_ages[i,j] = int( age_guess/0.5 + 0.5 ) * 0.5



    for i in range(0,2):

        for j in range(0,3):

            dataset.loc[ (dataset.Age.isnull()) & (dataset.Sex == i) & \

                        (dataset.Pclass == j+1), 'Age'] = guess_ages[i, j]

        

    dataset['Age'] = dataset['Age'].astype(int)
train_df.head()
train_df['AgeBand'] = pd.cut(train_df['Age'], 5)

train_df[['AgeBand', 'Survived']].groupby(["AgeBand"], as_index=False).mean().sort_values(by='AgeBand', ascending=True)
for dataset in combine:

    dataset.loc[ dataset['Age'] <= 16, 'Age' ] = 0

    dataset.loc[ (dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age' ] = 1

    dataset.loc[ (dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age' ] = 2

    dataset.loc[ (dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age' ] = 3

    dataset.loc[ dataset['Age'] > 64, 'Age' ] = 4
train_df.head()
train_df.drop('AgeBand', axis=1, inplace=True)

combine = [train_df, test_df]

train_df.head()
for dataset in combine:

    dataset["FamilySize"] = dataset['SibSp'] + dataset['Parch'] + 1



train_df[["FamilySize", "Survived"]].groupby(["FamilySize"], as_index=False).mean().sort_values(by='Survived', ascending=False)
for dataset in combine:

    dataset["IsAlone"] = 0

    dataset.loc[ dataset["FamilySize"] == 1, "IsAlone" ] = 1



train_df[["IsAlone", "Survived"]].groupby(["IsAlone"], as_index=False).mean()
for dataset in combine:

    dataset.drop(["Parch", "SibSp", "FamilySize"], axis=1, inplace=True)



train_df.shape, test_df.shape
combine = [train_df, test_df]
train_df.head()
for dataset in combine:

    dataset["Age*Class"] = dataset.Age * dataset.Pclass



train_df.loc[:, ['Age*Class', 'Age', 'Pclass'] ].head(10)
freq_port = train_df.Embarked.dropna().mode()[0]
for dataset in combine:

    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)



train_df[['Embarked', 'Survived']].groupby(['Embarked'], as_index = False).mean()
for dataset in combine:

    dataset['Embarked'] = dataset['Embarked'].map( { 'S':0, 'C':1, 'Q':2 } ).astype(int)



train_df.head()
test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True)
train_df.head()
train_df["Fareband"] = pd.qcut(train_df["Fare"], 4)

train_df[["Fareband", "Survived"]].groupby(["Fareband"], as_index=False).mean().sort_values(by='Fareband', ascending=True)
for dataset in combine:

    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare' ] = 0

    dataset.loc[ (dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare' ] = 1

    dataset.loc[ (dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31.0), 'Fare' ] = 2

    dataset.loc[ dataset['Fare'] > 31.0, 'Fare' ] = 3

    dataset["Fare"] = dataset["Fare"].astype(int)



train_df.drop("Fareband", axis=1, inplace=True)

combine = [train_df, test_df]



train_df.head(10)

    
test_df.Fare.unique()
X_train = train_df.drop("Survived", axis=1)

Y_train = train_df["Survived"]

X_test = test_df.drop("PassengerId", axis=1).copy()

X_train.shape, Y_train.shape, X_test.shape
#Logistic Regression



logreg = LogisticRegression()

logreg.fit(X_train, Y_train)

Y_pred = logreg.predict(X_test)

acc_log = round(logreg.score(X_train, Y_train)*100, 2)

acc_log
coeff_df = pd.DataFrame(train_df.columns.delete(0))

coeff_df.columns = ["Features"]

coeff_df["Correlation"] = pd.Series(logreg.coef_[0])



coeff_df.sort_values(by='Correlation', ascending=False)
# Support Vector Machine



svc = SVC()

svc.fit(X_train, Y_train)

Y_pred = svc.predict(X_test)

acc_svc = round(svc.score(X_train, Y_train)*100, 2)

acc_svc
knn = KNeighborsClassifier(n_neighbors=3)

knn.fit(X_train, Y_train)

Y_pred = knn.predict(X_test)

acc_knn = round(knn.score(X_train, Y_train)*100, 2)

acc_knn
gaussian = GaussianNB()

gaussian.fit(X_train, Y_train)

Y_pred = gaussian.predict(X_test)

acc_gaussian = round(gaussian.score(X_train, Y_train)*100, 2)

acc_gaussian
perceptron = Perceptron()

perceptron.fit(X_train, Y_train)

Y_pred = perceptron.predict(X_test)

acc_perceptron = round(perceptron.score(X_train, Y_train)*100, 2)

acc_perceptron

linear_svc = LinearSVC()

linear_svc.fit(X_train, Y_train)

Y_pred = linear_svc.predict(X_test)

acc_linear_svc = round(linear_svc.score(X_train, Y_train) * 100, 2)

acc_linear_svc
sgd = SGDClassifier()

sgd.fit(X_train, Y_train)

Y_pred = sgd.predict(X_test)

acc_sgd = round(sgd.score(X_train, Y_train)*100, 2)

acc_sgd
# Decision Tree



decision_tree = DecisionTreeClassifier()

decision_tree.fit(X_train, Y_train)

Y_pred = decision_tree.predict(X_test)

acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)

acc_decision_tree
# Random Forest



random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(X_train, Y_train)

Y_pred = random_forest.predict(X_test)

random_forest.score(X_train, Y_train)

acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)

acc_random_forest
submission = pd.DataFrame( {

    "PassengerId": test_df["PassengerId"],            

    "Survived": Y_pred

            } ) 
submission.to_csv('./submission.csv', index=False)