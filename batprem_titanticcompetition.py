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
import os

os.listdir('../input/titanic') #Check whether the path to dataset is correct
train_df = pd.read_csv('../input/titanic/train.csv')

test_df = pd.read_csv('../input/titanic/test.csv')

combine = [train_df, test_df]
import re

def getTicketType(ticket):

    try:

        word1 = re.findall("[a-zA-Z]+", ticket)[0]

    except IndexError:

        word1 = 'Other'

    return word1

getTicketType("STON/O2. 3101282")
train_df['TicketType'] = train_df['Ticket'].apply(getTicketType)

set(train_df['TicketType'].values)

train_df.head(5)
observedField = 'TicketType'

train_df[[observedField, 'Survived']].groupby([observedField], as_index=False).mean().sort_values(by='Survived', ascending=False)
observedField = 'Embarked'

train_df[[observedField, 'Survived']].groupby([observedField], as_index=False).mean().sort_values(by='Survived', ascending=False)
observedField = 'Cabin'

train_df[[observedField, 'Survived']].groupby([observedField], as_index=False).mean().sort_values(by='Survived', ascending=False)
#feature = 'Pclass'

def get_conditional_prob_of_survive(feature):

    '''

    Function for returning a resulting dataframe

    '''

    return (train_df[[feature, 'Survived']].groupby([feature], as_index=False).

             mean().

             sort_values(by='Survived', ascending=False))

get_conditional_prob_of_survive("Pclass")
get_conditional_prob_of_survive('Sex')
#train_df[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)

get_conditional_prob_of_survive('Parch')
g = sns.FacetGrid(train_df, col='Survived')

g.map(plt.hist, 'Age', bins=20)

# grid = sns.FacetGrid(train_df, col='Pclass', hue='Survived')

grid = sns.FacetGrid(train_df, col='Survived', row='Pclass', size=2.2, aspect=1.6)

grid.map(plt.hist, 'Age', alpha=.5, bins=20)

grid.add_legend()
grid = sns.FacetGrid(train_df, row='Embarked', size=2.2, aspect=1.6)

# X: Pclass

# Y: Survived

# Lines: sex

# Just noting, it's the first time I use seaborn

grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex',palette='deep')



grid.add_legend()
# grid = sns.FacetGrid(train_df, col='Embarked', hue='Survived', palette={0: 'k', 1: 'w'})

grid = sns.FacetGrid(train_df, row='Embarked', col='Survived', size=2.2, aspect=1.6)

grid.map(sns.barplot, 'Sex', 'Fare', alpha=.5, ci=None)

grid.add_legend()
print("Before", train_df.shape, test_df.shape, combine[0].shape, combine[1].shape)



train_df = train_df.drop(['Ticket', 'Cabin'], axis=1) #Drop unused colimes

test_df = test_df.drop(['Ticket', 'Cabin'], axis=1)

combine = [train_df, test_df]



"After", train_df.shape, test_df.shape, combine[0].shape, combine[1].shape

for dataset in combine:

    # Change famale to 1 male to 0

    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)



train_df.head()
train_df[train_df['Age'].isnull()]
# grid = sns.FacetGrid(train_df, col='Pclass', hue='Gender')

grid = sns.FacetGrid(train_df, row='Pclass', col='Sex', size=2.2, aspect=1.6)

grid.map(plt.hist, 'Age', alpha=.5, bins=20)

grid.add_legend()
guess_ages = np.zeros((2,3))
# For train data

# Guess value (Using median)

for sex in range(0,2):

    for pclass in range(1,4):

        guess_df = train_df[(train_df['Sex'] == sex) & \

                                  (train_df['Pclass'] == pclass)]['Age'].dropna()

        age_guess = guess_df.median() #Guess age

        guess_ages[sex,pclass-1] = int( age_guess/0.5 + 0.5 ) * 0.5



# Reassign value

for sex in range(0, 2):

    for pclass in range(1,4):

        train_df.loc[(

                        (train_df.Age.isnull())&

                        (train_df.Sex == sex)&

                        (train_df.Pclass == pclass)

                    ),

                    'Age'] = guess_ages[sex,pclass-1]



train_df['Age'] = train_df['Age'].astype(int)

train_df
# Do the same with testing

# Guess value (Using median)

for sex in range(0,2):

    for pclass in range(1,4):

        guess_df = test_df[(train_df['Sex'] == sex) & \

                                  (test_df['Pclass'] == pclass)]['Age'].dropna()

        age_guess = guess_df.median() #Guess age

        guess_ages[sex,pclass-1] = int( age_guess/0.5 + 0.5 ) * 0.5



# Reassign value

for sex in range(0, 2):

    for pclass in range(1,4):

        test_df.loc[(

                        (test_df.Age.isnull())&

                        (test_df.Sex == sex)&

                        (test_df.Pclass == pclass)

                    ),

                    'Age'] = guess_ages[sex,pclass-1]



test_df['Age'] = test_df['Age'].astype(int)

test_df
train_df['AgeBand'] = pd.cut(train_df['Age'], #Columns to be proceeded 

                             5 #Number of bands

                            )

train_df[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True)
for dataset in combine:    

    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0

    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1

    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2

    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3

    dataset.loc[ dataset['Age'] > 64, 'Age']

train_df.head()
train_df = train_df.drop(['AgeBand'], axis=1)

combine = [train_df, test_df]

train_df.head()
for dataset in combine:

    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1



train_df[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived', ascending=False)
for dataset in combine:

    dataset['IsAlone'] = 0

    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1



train_df[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean()
train_df = train_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)

test_df = test_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)

combine = [train_df, test_df]



train_df.head()
for dataset in combine:

    dataset['Age*Class'] = dataset.Age * dataset.Pclass



train_df.loc[:, ['Age*Class', 'Age', 'Pclass']].head(10)
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
train_df['FareBand'] = pd.qcut(train_df['Fare'], 4)

train_df[['FareBand', 'Survived']].groupby(['FareBand'],

                                           as_index=False).mean().sort_values(by='FareBand', ascending=True)
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
test_df = test_df.drop(['Name'], axis = 1)
train_df = train_df.drop(['TicketType','Name'], axis=1)
X_train = train_df.drop(["Survived","PassengerId"], axis=1)

Y_train = train_df["Survived"]

X_test  = test_df.drop("PassengerId", axis=1).copy()

X_train.shape, Y_train.shape, X_test.shape
 #Pclass	Sex	Age	Fare	Embarked	IsAlone	Age*Class

X_train
random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(X_train, Y_train)

Y_pred = random_forest.predict(X_test)

random_forest.score(X_train, Y_train)

acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)

acc_random_forest
from numpy import loadtxt

from keras.models import Sequential

from keras.layers import Dense



model = Sequential()

model.add(Dense(20, input_dim=7, activation='relu'))

model.add(Dense(8, activation='relu'))

model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train.values, Y_train.values, epochs=150, batch_size=10)
Y_preds_ = model.predict(X_test.values)
submission = pd.DataFrame({

        "PassengerId": test_df["PassengerId"],

        "Survived": Y_preds_.T[0]

    })
submission['Survived'] = submission['Survived'].apply(lambda x: int(x > 0.5))
submission.to_csv('submission.csv', index=False)