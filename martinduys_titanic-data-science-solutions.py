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
# read titanic training & test csv files as a pandas DataFrame

train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')
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

# Sibling distribution `[.65, .7]`

# Age and Fare `[.1, .2, .3, .4, .5, .6, .7, .8, .9, .99]`
train_df.describe(include=['O'])
g = sns.FacetGrid(train_df, col='Survived')

g.map(plt.hist, 'Age', bins=20)
grid = sns.FacetGrid(train_df, col='Pclass', hue='Survived', size=2.2, aspect=1.2, legend_out=False)

grid.map(plt.hist, 'Age', alpha=.5, bins=20)

grid.add_legend();
grid = sns.FacetGrid(train_df, col='Embarked', size=2.2, aspect=1.2, legend_out=False)

grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')

grid.add_legend()
grid = sns.FacetGrid(train_df, col='Embarked', hue='Survived', palette={0: 'k', 1: 'w'}, size=2.2, aspect=1.2, legend_out=False)

grid.map(sns.barplot, 'Sex', 'Fare', alpha=.5, ci=None)

grid.add_legend()
train_df = train_df.drop(['Ticket', 'Cabin'], axis=1)

test_df = test_df.drop(['Ticket', 'Cabin'], axis=1)
train_df['Title'] = train_df.Name.str.extract('(\w+\.)', expand=False)

sns.barplot(hue="Survived", x="Age", y="Title", data=train_df, ci=False)
test_df['Title'] = test_df.Name.str.extract('(\w+\.)', expand=False)



train_df = train_df.drop(['Name', 'PassengerId'], axis=1)

test_df = test_df.drop(['Name'], axis=1)

test_df.describe(include=['O'])
train_df['Gender'] = train_df['Sex'].map( {'female': 1, 'male': 0} ).astype(int)

train_df.loc[:, ['Gender', 'Sex']].head()
test_df['Gender'] = test_df['Sex'].map( {'female': 1, 'male': 0} ).astype(int)

test_df.loc[:, ['Gender', 'Sex']].head()
train_df = train_df.drop(['Sex'], axis=1)

test_df = test_df.drop(['Sex'], axis=1)

train_df.head()
grid = sns.FacetGrid(train_df, col='Pclass', hue='Gender', size=2.2, aspect=1.2, legend_out=False)

grid.map(plt.hist, 'Age', alpha=.5, bins=20)

grid.add_legend();
guess_ages = np.zeros((2,3))

guess_ages
for i in range(0, 2):

    for j in range(0, 3):

        guess_df = train_df[(train_df['Gender'] == i) & \

                              (train_df['Pclass'] == j+1)]['Age'].dropna()

        

        # Correlation of AgeFill is -0.014850

        # age_mean = guess_df.mean()

        # age_std = guess_df.std()

        # age_guess = rnd.uniform(age_mean - age_std, age_mean + age_std)

        

        # Correlation of AgeFill is -0.011304

        age_guess = guess_df.median()



        # Convert random age float to nearest .5 age

        guess_ages[i,j] = int( age_guess/0.5 + 0.5 ) * 0.5

        

guess_ages
train_df['AgeFill'] = train_df['Age']



for i in range(0, 2):

    for j in range(0, 3):

        train_df.loc[ (train_df.Age.isnull()) & (train_df.Gender == i) & (train_df.Pclass == j+1),\

                'AgeFill'] = guess_ages[i,j]



train_df[train_df['Age'].isnull()][['Gender','Pclass','Age','AgeFill']].head(10)
guess_ages = np.zeros((2,3))



for i in range(0, 2):

    for j in range(0, 3):

        guess_df = test_df[(test_df['Gender'] == i) & \

                              (test_df['Pclass'] == j+1)]['Age'].dropna()



        # Correlation of AgeFill is -0.014850

        # age_mean = guess_df.mean()

        # age_std = guess_df.std()

        # age_guess = rnd.uniform(age_mean - age_std, age_mean + age_std)



        # Correlation of AgeFill is -0.011304

        age_guess = guess_df.median()



        guess_ages[i,j] = int( age_guess/0.5 + 0.5 ) * 0.5



test_df['AgeFill'] = test_df['Age']



for i in range(0, 2):

    for j in range(0, 3):

        test_df.loc[ (test_df.Age.isnull()) & (test_df.Gender == i) & (test_df.Pclass == j+1),\

                'AgeFill'] = guess_ages[i,j]



test_df[test_df['Age'].isnull()][['Gender','Pclass','Age','AgeFill']].head(10)
train_df = train_df.drop(['Age'], axis=1)

test_df = test_df.drop(['Age'], axis=1)

train_df.head()
# Logistic Regression Score is 0.81032547699214363

# Parch correlation is -0.065878 and SibSp correlation is -0.370618



# Decision: Retain Parch and SibSp as separate features



# Logistic Regression Score is 0.80808080808080807

# FamilySize correlation is -0.233974



# train_df['FamilySize'] = train_df['SibSp'] + train_df['Parch']

# test_df['FamilySize'] = test_df['SibSp'] + test_df['Parch']

# train_df.loc[:, ['Parch', 'SibSp', 'FamilySize']].head(10)
# train_df = train_df.drop(['Parch', 'SibSp'], axis=1)

# test_df = test_df.drop(['Parch', 'SibSp'], axis=1)

# train_df.head()
test_df['Age*Class'] = test_df.AgeFill * test_df.Pclass

train_df['Age*Class'] = train_df.AgeFill * train_df.Pclass

train_df.loc[:, ['Age*Class', 'AgeFill', 'Pclass']].head(10)
freq_port = train_df.Embarked.dropna().mode()[0]

freq_port
train_df['EmbarkedFill'] = train_df['Embarked']

train_df.loc[train_df['Embarked'].isnull(), 'EmbarkedFill'] = freq_port

train_df[train_df['Embarked'].isnull()][['Embarked','EmbarkedFill']].head(10)
test_df['EmbarkedFill'] = test_df['Embarked']

train_df = train_df.drop(['Embarked'], axis=1)

test_df = test_df.drop(['Embarked'], axis=1)

train_df.head()
Ports = list(enumerate(np.unique(train_df['EmbarkedFill'])))

Ports_dict = { name : i for i, name in Ports }              

train_df['Port'] = train_df.EmbarkedFill.map( lambda x: Ports_dict[x]).astype(int)



Ports = list(enumerate(np.unique(test_df['EmbarkedFill'])))

Ports_dict = { name : i for i, name in Ports }

test_df['Port'] = test_df.EmbarkedFill.map( lambda x: Ports_dict[x]).astype(int)



train_df[['EmbarkedFill', 'Port']].head(10)
Titles = list(enumerate(np.unique(train_df['Title'])))

Titles_dict = { name : i for i, name in Titles }           

train_df['TitleBand'] = train_df.Title.map( lambda x: Titles_dict[x]).astype(int)



Titles = list(enumerate(np.unique(test_df['Title'])))

Titles_dict = { name : i for i, name in Titles }           

test_df['TitleBand'] = test_df.Title.map( lambda x: Titles_dict[x]).astype(int)



train_df[['Title', 'TitleBand']].head(10)
train_df = train_df.drop(['EmbarkedFill', 'Title'], axis=1)

test_df = test_df.drop(['EmbarkedFill', 'Title'], axis=1)

train_df.head()
test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True)



train_df['Fare'] = train_df['Fare'].round(2)

test_df['Fare'] = test_df['Fare'].round(2)



test_df.head(10)
X_train = train_df.drop("Survived", axis=1)

Y_train = train_df["Survived"]

X_test  = test_df.drop("PassengerId", axis=1).copy()

X_train.shape, Y_train.shape, X_test.shape
# Logistic Regression



logreg = LogisticRegression()

logreg.fit(X_train, Y_train)

Y_pred = logreg.predict(X_test)

logreg.score(X_train, Y_train)
# get Correlation Coefficient for each feature using Logistic Regression

coeff_df = pd.DataFrame(train_df.columns.delete(0))

coeff_df.columns = ['Features']

coeff_df["Correlation"] = pd.Series(logreg.coef_[0])



# preview

coeff_df.sort_values(by='Correlation', ascending=False)
# Support Vector Machines



svc = SVC()

svc.fit(X_train, Y_train)

Y_pred = svc.predict(X_test)

svc.score(X_train, Y_train)
knn = KNeighborsClassifier(n_neighbors = 3)

knn.fit(X_train, Y_train)

Y_pred = knn.predict(X_test)

knn.score(X_train, Y_train)
# Gaussian Naive Bayes



gaussian = GaussianNB()

gaussian.fit(X_train, Y_train)

Y_pred = gaussian.predict(X_test)

gaussian.score(X_train, Y_train)
# Random Forests



random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(X_train, Y_train)

Y_pred = random_forest.predict(X_test)

random_forest.score(X_train, Y_train)
submission = pd.DataFrame({

        "PassengerId": test_df["PassengerId"],

        "Survived": Y_pred

    })



# submission.to_csv('submission.csv', index=False)