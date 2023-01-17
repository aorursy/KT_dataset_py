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

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train_data = pd.read_csv("/kaggle/input/titanic/train.csv")

train_data.tail()
test_data = pd.read_csv("/kaggle/input/titanic/test.csv")

test_data.head()

combine = [train_data, test_data]
train_data.info()

print('_'*40)

test_data.info()
train_data.describe()
train_data.describe(include=['O'])
train_data[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train_data[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train_data[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train_data[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)
g = sns.FacetGrid(train_data, col='Survived')

g.map(plt.hist, 'Age', bins=20)
grid = sns.FacetGrid(train_data, row='Embarked', col='Survived', size=2.2, aspect=1.6)

grid.map(sns.barplot, 'Sex', 'Fare', alpha=.5, ci=None)

grid.add_legend()
print("Before", train_data.shape, test_data.shape, combine[0].shape, combine[1].shape)



train_data = train_data.drop(['Ticket', 'Cabin'], axis=1)

test_data = test_data.drop(['Ticket', 'Cabin'], axis=1)

combine = [train_data, test_data]



"After", train_data.shape, test_data.shape, combine[0].shape, combine[1].shape
train_data.head()
for dataset in combine:

    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)



pd.crosstab(train_data['Title'], train_data['Sex'])
for dataset in combine:

    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',\

 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')



    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

    

train_data[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

for dataset in combine:

    dataset['Title'] = dataset['Title'].map(title_mapping)

    dataset['Title'] = dataset['Title'].fillna(0)



train_data.head()
train_data = train_data.drop(['Name', 'PassengerId'], axis=1)

test_data = test_data.drop(['Name'], axis=1)

combine = [train_data, test_data]

train_data.shape, test_data.shape
for dataset in combine:

    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)



train_data.head()
# grid = sns.FacetGrid(train_data, col='Pclass', hue='Gender')

grid = sns.FacetGrid(train_data, row='Pclass', col='Sex', size=2.2, aspect=1.6)

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



train_data.head()
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



train_data.head()
train_data['AgeBand'] = pd.cut(train_data['Age'], 5)

train_data[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True)
for dataset in combine:    

    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0

    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1

    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2

    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3

    dataset.loc[ dataset['Age'] > 64, 'Age']

train_data.head()
train_data = train_data.drop(['AgeBand'], axis=1)

combine = [train_data, test_data]

train_data.head()
for dataset in combine:

    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1



train_data[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived', ascending=False)
for dataset in combine:

    dataset['IsAlone'] = 0

    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1



train_data[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean()
train_data = train_data.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)

test_data = test_data.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)

combine = [train_data, test_data]



train_data.head()
for dataset in combine:

    dataset['Age*Class'] = dataset.Age * dataset.Pclass



train_data.loc[:, ['Age*Class', 'Age', 'Pclass']].head(10)
freq_port = train_data.Embarked.dropna().mode()[0]

freq_port



for dataset in combine:

    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)

    

train_data[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False)
for dataset in combine:

    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)



train_data.head()
test_data['Fare'].fillna(test_data['Fare'].dropna().median(), inplace=True)

test_data.head()
train_data['FareBand'] = pd.qcut(train_data['Fare'], 4)

train_data[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand', ascending=True)
for dataset in combine:

    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0

    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1

    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2

    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3

    dataset['Fare'] = dataset['Fare'].astype(int)



train_data = train_data.drop(['FareBand'], axis=1)

combine = [train_data, test_data]

    

train_data.head(10)
test_data.head(10)
X_train = train_data.drop("Survived", axis=1)

Y_train = train_data["Survived"]

X_test  = test_data.drop("PassengerId", axis=1).copy()

X_train.shape, Y_train.shape, X_test.shape



X_cv = X_train.tail(100)

Y_cv = Y_train.tail(100)



X_cv.shape, Y_cv.shape


# first neural network with keras tutorial

from numpy import loadtxt

from keras.models import Sequential

from keras.layers import Dense

from keras import optimizers





# define the keras model

model = Sequential()

model.add(Dense(16, input_dim=8, activation='relu'))

model.add(Dense(9, activation='relu'))

model.add(Dense(1, activation='sigmoid'))



optimizers.Adam(learning_rate=0.03, beta_1=0.9, beta_2=0.999, amsgrad=False)

# compile the keras model

model.compile(loss='binary_crossentropy', optimizer='nadam', metrics=['accuracy'])

# fit the keras model on the dataset

model.fit(X_train, Y_train, epochs=150, batch_size=10)

# evaluate the keras model



_, accuracy = model.evaluate(X_train, Y_train)

print('Accuracy: %.2f' % (accuracy*100))

predictions=model.predict_classes(X_test)
output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions.flatten()})

output.head()
output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")