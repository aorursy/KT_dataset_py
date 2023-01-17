# data analysis and wrangling

import pandas as pd

import numpy as np

import random as rnd



# visualization

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

sns.set_style("whitegrid")



# machine learning

from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import AdaBoostClassifier

from sklearn.neural_network import MLPClassifier
# get titanic & test csv files as a DataFrame

train_df = pd.read_csv("../input/train.csv")

test_df = pd.read_csv("../input/test.csv")

combine = [train_df, test_df]
# view column labels

print(train_df.columns.values)
# preview the data

train_df.head()
train_df.head(3).T
# missing values, data types

train_df.info()

print('-'*40)

test_df.info()
# describe numeric columns

train_df.describe()
# describe categorical columns

train_df.describe(include=['O'])
# just for fun, examine the records of ten year olds (there are only two) 

train_df[train_df.Age == 10].stack()
# count passengers by sex

plt.subplot(211) # 3 digit convenience notation for arguments (last digit represents plot number)

sns.countplot(x='Sex', data=train_df, palette='Greens_d')



# survival rate by sex

# note that barplot plots mean() on y by default

plt.subplot(212)

sns.barplot(x='Sex', y='Survived', data=train_df, palette='Greens_d') 
# count passengers by sex

train_df.groupby('Sex').size()
# survival rates by sex

train_df.groupby(['Sex'])['Survived'].mean().sort_values()
# size of groups in passenger class

plt.subplots(figsize=(8,6))

plt.subplot(211) 

sns.countplot(x='Pclass', data=train_df, palette='Purples_d') # _d = dark palette



# survival rate by sex

plt.subplot(212)

sns.barplot(x='Pclass', y='Survived', data=train_df, palette='Purples_d') 
# count passengers by passenger class

train_df.groupby(['Pclass']).size()
# survival rates by passenger class

train_df.groupby(['Pclass'])['Survived'].mean().sort_values(ascending=False)
# count the number of passengers for first 25 ages

train_df.groupby('Age').size().head(25)



# another way to do the above

#train_df['Age'].value_counts().sort_index().head(25) 
# convert ages to ints

age = train_df[['Age','Survived']].dropna() # returns a copy with blanks removed

age['Age'] = age['Age'].astype(int) # floors floats



# count passengers by age (smoothed via gaussian kernels)

plt.subplots(figsize=(18,6))

plt.subplot(311)

sns.kdeplot(age['Age'], shade=True, cut=0)



# count passengers by age (no smoothing)

plt.subplot(312)

sns.countplot(x='Age', data=age, palette='GnBu_d')



# survival rates by age

plt.subplot(313)

sns.barplot(x='Age', y='Survived', data=age, ci=None, palette='Oranges_d') # takes mean by default
# bin age into groups

train_df['AgeGroup'] = pd.cut(train_df['Age'],[0,4,15,25,35,45,65,100])

test_df['AgeGroup'] = pd.cut(test_df['Age'],[0,4,15,25,35,45,65,100])



# survival by age group

train_df.groupby('AgeGroup')['Survived'].mean()
# survival by age group and sex

train_df[['Survived','AgeGroup', 'Sex']].groupby(['Sex', 'AgeGroup']).mean()
# count passengers by age group and sex

sns.factorplot(x='AgeGroup', col='Sex', data=train_df, kind='count')



# survival by age group and sex

sns.factorplot(x='AgeGroup', y='Survived', col='Sex', data=train_df, kind='bar')
# calculate family size

train_df['FamilySize'] = train_df['SibSp'] + train_df['Parch'] + 1

test_df['FamilySize'] = test_df['SibSp'] + test_df['Parch'] + 1



# count passengers by age group and sex

plt.subplot(211)

sns.countplot(x='FamilySize', data=train_df)



# survival by age group and sex

plt.subplot(212)

sns.barplot(x='FamilySize', y='Survived', data=train_df)
# deck is the first letter of cabin

train_df['Deck'] = train_df['Cabin'].dropna().apply(lambda x: str(x)[0])

train_df[['PassengerId','Name', 'Cabin', 'Deck']].head(2).T
# count passengers by the deck their cabin is on

plt.subplots(figsize=(8,6))

plt.subplot(211) 

sns.countplot(x='Deck', data=train_df)



# survival rate by deck

plt.subplot(212)

sns.barplot(x='Deck', y='Survived', data=train_df) 
# number of males/females without an age

def get_na(dataset):

    na_males = dataset[dataset.Sex == 'male'].loc[:,'AgeGroup'].isnull().sum()

    na_females = dataset[dataset.Sex == 'female'].loc[:,'AgeGroup'].isnull().sum()

    return {'male': na_males, 'female': na_females}



# number of males and females by age group

def get_counts(dataset):

    return dataset.groupby(['Sex', 'AgeGroup']).size()



# randomly generate a list of age groups based on age group frequency (for each sex separately) 

def generate_age_groups(num, freq):

    age_groups = {}

    for sex in ['male','female']:

        relfreq = freq[sex] / freq[sex].sum()

        age_groups[sex] = np.random.choice(freq[sex].index, size=num[sex], replace=True, p=relfreq)    

    return age_groups



# insert the new age group values

def insert_age_group_values(dataset, age_groups):

    for sex in ['male','female']:

        tmp = pd.DataFrame(dataset[(dataset.Sex == sex) & dataset.Age.isnull()]) # filter on sex and null ages 

        tmp['AgeGroup'] = age_groups[sex] # index age group values

        dataset = dataset.combine_first(tmp) # uses tmp to fill holes

    return dataset



# fill holes for train_df

na = get_na(train_df)

counts = get_counts(train_df)

counts['female']

age_groups = generate_age_groups(na, counts)

age_groups['female']

train_df = insert_age_group_values(train_df, age_groups)

train_df.info() # check all nulls have been filled    

print('-'*40)



# repeat for test_df

na = get_na(test_df)

counts = get_counts(train_df) # reuse the frequencies taken over the training data as it is larger

age_groups = generate_age_groups(na, counts)

test_df = insert_age_group_values(test_df, age_groups)

test_df.info() # check all nulls have been filled     
# Sex -> Female



# training set

dummy = pd.get_dummies(train_df['Sex'])

dummy.columns = ['Female','Male']

train_df = train_df.join(dummy['Female'])



# test set

dummy = pd.get_dummies(test_df['Sex'])

dummy.columns = ['Female','Male']

test_df = test_df.join(dummy['Female'])



train_df[['Name', 'Sex', 'Female']].head(2).T

#train_df.columns
# Pclass -> PClass_1, PClass_2



# training set

dummy  = pd.get_dummies(train_df['Pclass'])

dummy.columns = ['PClass_1','PClass_2','PClass_3']

train_df = train_df.join(dummy[['PClass_1', 'PClass_2']])



# test set

dummy  = pd.get_dummies(test_df['Pclass'])

dummy.columns = ['PClass_1','PClass_2','PClass_3']

test_df = test_df.join(dummy[['PClass_1', 'PClass_2']])



train_df[['Name', 'Pclass', 'PClass_1', 'PClass_2']].head(2).T

#train_df.columns
# AgeGroup -> binary features



# training set

dummy  = pd.get_dummies(train_df['AgeGroup'])

dummy.columns = ['Ages_4','Ages_15','Ages_25','Ages_35','Ages_45','Ages_65','Ages_100']

train_df = train_df.join(dummy)



# test set

dummy  = pd.get_dummies(test_df['AgeGroup'])

dummy.columns = ['Ages_4','Ages_15','Ages_25','Ages_35','Ages_45','Ages_65','Ages_100']

test_df = test_df.join(dummy)
# Fare



# there is a single missing "Fare" value

test_df['Fare'].fillna(test_df['Fare'].median(), inplace=True)



# convert from float to int (floor)

#train_df['Fare'] = train_df['Fare'].astype(int)

#test_df['Fare'] = test_df['Fare'].astype(int)
# Embarked -> PortC, PortQ



# Fill missing values with the most occurred value

print(train_df.groupby('Embarked').size().sort_values())

train_df['Embarked'] = train_df['Embarked'].fillna('S')



# training set

dummy = pd.get_dummies(train_df['Embarked'])

#dummy.columns

dummy.columns = ['Port_C','Port_Q','Port_S']

#train_df = train_df.join(dummy[['Port_C','Port_Q']])



# test set

dummy  = pd.get_dummies(test_df['Embarked'])

dummy.columns = ['Port_C','Port_Q','Port_S']

#test_df = test_df.join(dummy[['Port_C','Port_Q']])
# drop the attributes that will be unused

train_df.drop(['PassengerId', 'Pclass', 'Name', 'Sex', 'Age', 

                   'SibSp', 'Parch', 'Ticket', 'Cabin', 'Fare', 

                   'Embarked', 'Deck', 'AgeGroup'], axis=1, inplace=True)



test_df.drop(['Pclass', 'Name', 'Sex', 'Age', 

                   'SibSp', 'Parch', 'Ticket', 'Cabin', 'Fare',

                   'Embarked', 'AgeGroup'], axis=1, inplace=True)



train_df.head(10).T
# split the datasets into matched input and ouput pairs

X_train = train_df.drop("Survived", axis=1) # X = inputs

Y_train = train_df["Survived"] # Y = outputs

X_test  = test_df.drop("PassengerId", axis=1).copy()

X_train.shape, Y_train.shape, X_test.shape

# Logistic Regression



logreg = LogisticRegression()

scores = cross_val_score(logreg, X_train, Y_train, cv=10)

acc_log = round(scores.mean() * 100, 2)

acc_log

#Y_pred = logreg.predict(X_test)
logreg.fit(X_train, Y_train)

coeff_df = pd.DataFrame(train_df.columns.delete(0))

coeff_df.columns = ['Feature']

coeff_df["Correlation"] = pd.Series(logreg.coef_[0])



coeff_df.sort_values(by='Correlation', ascending=False)
# Gaussian Naive Bayes



gaussian = GaussianNB()

scores = cross_val_score(gaussian, X_train, Y_train, cv=10)

acc_gaussian = round(scores.mean() * 100, 2)

acc_gaussian
# Perceptron (a single layer neural net)



perceptron = Perceptron()

scores = cross_val_score(perceptron, X_train, Y_train, cv=10)

acc_perceptron = round(scores.mean() * 100, 2)

acc_perceptron
# Neural Network (a multi layer neural net)



neural_net = MLPClassifier()

scores = cross_val_score(neural_net, X_train, Y_train, cv=10)

acc_neural_net = round(scores.mean() * 100, 2)

acc_neural_net
# Stochastic Gradient Descent



sgd = SGDClassifier()

scores = cross_val_score(sgd, X_train, Y_train, cv=10)

acc_sgd = round(scores.mean() * 100, 2)

acc_sgd
# Linear SVC



linear_svc = LinearSVC()

scores = cross_val_score(linear_svc, X_train, Y_train, cv=10)

acc_linear_svc = round(scores.mean() * 100, 2)

acc_linear_svc
# Support Vector Machine



svc = SVC() # uses a rbf kernel by default (i.e. can discover non-linear boundaries)

scores = cross_val_score(svc, X_train, Y_train, cv=10)

acc_svc = round(scores.mean() * 100, 2)

acc_svc
# Decision Tree



decision_tree = DecisionTreeClassifier()

scores = cross_val_score(decision_tree, X_train, Y_train, cv=10)

acc_decision_tree = round(scores.mean() * 100, 2)

acc_decision_tree
# Random Forest - an ensemble model



random_forest = RandomForestClassifier(n_estimators=100)

scores = cross_val_score(random_forest, X_train, Y_train, cv=10)

acc_random_forest = round(scores.mean() * 100, 2)

acc_random_forest
# AdaBoost - an ensemble method



ada_boost = AdaBoostClassifier(n_estimators=100)

scores = cross_val_score(ada_boost, X_train, Y_train, cv=10)

acc_ada_boost = round(scores.mean() * 100, 2)

acc_ada_boost
# k-Nearest Neighbors - a non-parametric method



knn = KNeighborsClassifier(n_neighbors = 5)

scores = cross_val_score(knn, X_train, Y_train, cv=10)

acc_knn = round(scores.mean() * 100, 2)

acc_knn
models = pd.DataFrame({

    'Model': ['Support Vector Machine', 'kNN', 'Logistic Regression', 

              'Random Forest', 'Naive Bayes', 'Perceptron', 

              'Stochastic Gradient Descent', 'Linear SVC', 

              'Decision Tree', 'AdaBoost', 'Neural Network'],

    'Score': [acc_svc, acc_knn, acc_log, 

              acc_random_forest, acc_gaussian, acc_perceptron, 

              acc_sgd, acc_linear_svc, acc_decision_tree, 

              acc_ada_boost, acc_neural_net]})

models.sort_values(by='Score', ascending=False)
# using random forest for submission

random_forest.fit(X_train, Y_train)

Y_pred = random_forest.predict(X_test)



submission = pd.DataFrame({

        "PassengerId": test_df["PassengerId"],

        "Survived": Y_pred

    })

submission.to_csv('titanic_submission_1.csv', index=False)

#pd.set_option('display.max_rows', len(submission))

#submission
# Random Forest : scoring on training data



random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(X_train, Y_train)

acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)

acc_random_forest