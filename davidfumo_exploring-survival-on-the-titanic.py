# Imports



# pandas

import pandas as pd

from pandas import Series,DataFrame

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

sns.set_style('whitegrid')



# Machine Learning

from sklearn.ensemble import RandomForestClassifier

from sklearn.cross_validation import cross_val_score
# create the dataframes

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')



# preview data

train.head()
test.head()
# See the shape of our datasets

print(train.shape)

print(test.shape)
train.info()
# Missing columns data

train.isnull().sum()
test.isnull().sum()
# Look for the most common value for Embarked

train['Embarked'].value_counts()
# Embarked



# From this we can see there're a lot of age columns with missing values

# Cabin has a lot of missing values, let's assume it doesn't play 

# a big rule in our predictions

train = train.drop(['PassengerId','Name','Ticket','Cabin'], axis=1)

test    = test.drop(['Name','Ticket','Cabin'], axis=1)



# fill the two missing values with the most occurred value (S)

train['Embarked'] = train['Embarked'].fillna("S")



# Set up the matplotlib figure

fig, (axis1,axis2) = plt.subplots(1,2,figsize=(10,5))



# Draw barplot to show survivors for Embarked considering Sex

sns.factorplot(x="Embarked", y="Survived", hue="Sex", data=train,

                   size=6, kind="bar")



sns.countplot(x='Embarked', data=train, ax=axis1)

sns.countplot(x='Survived', hue="Embarked", data=train, order=[0,1], ax=axis2)
# Pclass



fig, (axis1,axis2) = plt.subplots(1,2, sharex=True,figsize=(10,5))



# Draw a nested barplot to show survivors for class 1, 2, 3

sns.countplot(x='Survived', hue='Pclass', data=train, ax=axis1)

axis1.set_ylabel('Frequency')



# Draw a nested barplot to show survivors for class and sex

sns.factorplot(x='Pclass', y='Survived', hue='Sex', data=train,

                   size=6, kind='bar', ax=axis2)

axis2.set_ylabel('survival probability')
# Fare



# CLean Fare in the test dataset

# Fill the missing value

test['Fare'] = test['Fare'].fillna(test['Fare'].median())



# use int instead of float

train['Fare'] = train['Fare'].astype(int)

test['Fare'] = test['Fare'].astype(int)



# create plot

train['Fare'].plot(kind='hist', figsize=(5,5),bins=100, xlim=(0,25))
# Age



train_avg_age = train['Age'].mean()

train_std_age = train['Age'].std()



test_avg_age = test['Age'].mean()

test_std_age = test['Age'].std()



# Generate random numbers between (mean - std) and (mean + std)

# to fill the empty age columns

# The generated age values will be in the

rand_train = np.random.randint(train_avg_age - train_std_age, train_avg_age + train_std_age,

                              train['Age'].isnull().sum())

rand_test = np.random.randint(test_avg_age - test_std_age, test_avg_age + test_std_age,

                              test['Age'].isnull().sum())



# fill "NaN" values in empty Age columns

train['Age'][np.isnan(train['Age'])] = rand_train

test['Age'][np.isnan(test['Age'])] = rand_test



# Now that we haven't missing values we can

# use Age as int instead float

train['Age'] = train['Age'].astype(int)

test['Age'] = test['Age'].astype(int)



# plot the distribuition of people by age

train['Age'].plot(kind='hist', bins=50)
# Sex 



fig, (axis1,axis2) = plt.subplots(1,2,sharex=True,figsize=(10,5))



# How many people survived Vs Died

#sns.countplot(x="Survived", data=train, palette="muted")

sns.countplot(x='Survived', data=train, order=[0,1], ax=axis1)

axis1.set_ylabel('Frequency')



# Survived people by their gender

#sns.countplot(x="Survived", hue='Sex', data=train, palette="muted")

sns.countplot(x='Survived', hue="Sex", data=train, order=[0,1], ax=axis2)

axis2.set_ylabel("Frequency")



axis1.set_xticklabels(['Survived', 'Died'], rotation=0)

axis2.set_xticklabels(['Survived', 'Died'], rotation=0)
# Family



# Family is made up of:

# sibsp = Number of Siblings/Spouses Aboard

# parch = Number of Parents/Children Aboard

train['Family'] = train['SibSp'] + train['Parch']



# visualize family size distribuition

train['Family'].plot(kind='hist', title='Family Size', figsize=(7,5),bins=100, xlim=(0,10))



train['Family'].loc[train['Family'] > 0] = 1

train['Family'].loc[train['Family'] == 0] = 0



test['Family'] =  test["Parch"] + test["SibSp"]

test['Family'].loc[test['Family'] > 0] = 1

test['Family'].loc[test['Family'] == 0] = 0



# drop Parch & SibSp from dataframes

train = train.drop(['SibSp','Parch'], axis=1)

test = test.drop(['SibSp','Parch'], axis=1)





# Does family play a role?

fig, (axis1,axis2) = plt.subplots(1,2, sharex=True,figsize=(10,5))





sns.countplot(x='Family', hue='Survived', data=train, order=[1, 0], ax=axis1)

axis1.set_xticklabels(['With Family', 'Without'])



# from people with family, how many of them survived?

sns.countplot(x='Survived', hue='Family', data=train[train['Family'] == 1], order=[1, 0], ax=axis2)

#axis2.set_xticklabels(['Survived', 'Died'])

axis2.set_title('Deaths VS survivors in People With Family')
# There're some people with families on board

# It seems like this predictor doesn't play a big role compared to others

# but we're going to use it to train our model

train[train['Family'] == 1].sum()
# Transforming categorical data to numeric data for our machine learning model

# Current categorical data: Sex, Embarked



# Sex

train.loc[train["Sex"] == "male", "Sex"] = 0 

train.loc[train["Sex"] == "female", "Sex"] = 1



test.loc[test["Sex"] == "male", "Sex"] = 0 

test.loc[test["Sex"] == "female", "Sex"] = 1



# Embarked

train.loc[train["Embarked"] == "S", "Embarked"] = 0

train.loc[train["Embarked"] == "C", "Embarked"] = 1

train.loc[train["Embarked"] == "Q", "Embarked"] = 2



test.loc[test["Embarked"] == "S", "Embarked"] = 0

test.loc[test["Embarked"] == "C", "Embarked"] = 1

test.loc[test["Embarked"] == "Q", "Embarked"] = 2



train.head()
# Now we're ready to build the machine learning model



# Predictors

feature_train = ['Pclass', 'Age', 'Sex', 'Fare', 'Embarked', 'Family']



X = train[feature_train] # instances to learn from

y = train['Survived'] # target/responses the model is trying to learn to predict





# Random Forest

# Initialize our algorithm with the default paramters

# n_estimators is the number of trees we want to make

# min_samples_split is the minimum number of rows we need to make a split

# min_samples_leaf is the minimum number of samples we can have at the place where a tree branch ends (the bottom points of the tree)

# forest = RandomForestClassifier(random_state=1, n_estimators=25, min_samples_split=2, min_samples_leaf=1)

forest = RandomForestClassifier(random_state=1, n_estimators=25)

scores = cross_val_score(forest, X, y, cv=10, scoring='accuracy')

print(scores)
# use average accuracy as an estimate of out-of-sample accuracy

print(scores.mean())
# search for an optimal value of n_estimators for Random Forest Model

# This code takes a bit of time to execute

"""

n_estimators = list(range(50, 100))

n_scores = []



for n in n_estimators:

    forest = RandomForestClassifier(random_state=1, n_estimators=n)

    scores = cross_val_score(forest, X, y, cv=10, scoring='accuracy')

    n_scores.append(scores.mean())



print(n_scores)

"""
# plot the value of n (x-axis) VS the cross validation accuracy (y-axis)

#plt.plot(n_estimators, n_scores)

#plt.xlabel('Value of N for Random Forest Classifier')

#plt.ylabel('Cross-Validated Accuracy')
# Make predictions using the test set.

forest = RandomForestClassifier(random_state=1, n_estimators=100)

forest.fit(X, y)

predictions = forest.predict(test[feature_train])
# Create a new dataframe with only the columns Kaggle wants from the dataset.

submission = DataFrame({

        "PassengerId": test["PassengerId"],

        "Survived": predictions

    })

submission.to_csv('titanic_submission_rf.csv', index=False)