import sys

import matplotlib



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



import sklearn

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier
print(sys.version)               # Version 3.6.3

print(np.__version__)            # Version 1.15.0

print(pd.__version__)            # Version 0.23.4

print(matplotlib.__version__)    # Version 2.2.2

print(sklearn.__version__)       # Version 0.19.2
base_path = "../input/"

train_data = pd.read_csv(base_path + "train.csv")   # 28 KB

test_data = pd.read_csv(base_path + "test.csv")     # 60 KB

combine = [train_data, test_data]
print(train_data.columns.values)
print(train_data.head())
print(train_data.tail())
print(train_data.info())
print(train_data.describe())
train_data[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train_data[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train_data[['SibSp', 'Survived']].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train_data[['Parch', 'Survived']].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train_data[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False)
print("Before", train_data.shape, test_data.shape, combine[0].shape, combine[1].shape)



train_data = train_data.drop(['Ticket', 'Cabin'], axis=1)

test_data = test_data.drop(['Ticket', 'Cabin'], axis=1)

combine = [train_data, test_data]



print("After", train_data.shape, test_data.shape, combine[0].shape, combine[1].shape)
for dataset in combine:

    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

    

pd.crosstab(train_data['Title'], train_data['Sex'])
for dataset in combine:

    dataset['Title'] = dataset['Title'].replace(['Capt', 'Col', 'Countess', 'Don', 'Dr', 'Jonkheer', 'Lady', 'Major', 'Rev', 'Sir'], 'Rare')



    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

    

train_data[['Title', 'Survived']].groupby(['Title'], as_index=False).mean().sort_values(by='Survived', ascending=False)
title_mapping = {"Mr": 1, "Mrs": 2, "Master": 3, "Miss": 4, "Rare": 5}

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
train_data.isna().sum()
guess_ages = np.zeros((2,3))

guess_ages
### NO IDEA what is going on here. Copied from another notebook.



for dataset in combine:

    for i in range(0,2):

        for j in range(0,3):

            guess_df = dataset[(dataset['Sex'] == i) & (dataset['Pclass'] == j+1)]['Age'].dropna()



            age_guess = guess_df.median()

            

            guess_ages[i,j] = int( age_guess/0.5 + 0.5 ) * 0.5

            

    for i in range(0,2):

        for j in range(0,3):

            dataset.loc[ (dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j+1), 'Age' ] = guess_ages[i,j]

            

    dataset['Age'] = dataset['Age'].astype(int)

    

train_data.head()
train_data['AgeBand'] = pd.cut(train_data['Age'], 5)

train_data[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True)
for dataset in combine:

    dataset.loc[ dataset['Age'] <= 16, 'Age' ] = 0

    dataset.loc[ (dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age' ] = 1

    dataset.loc[ (dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age' ] = 2

    dataset.loc[ (dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age' ] = 3

    dataset.loc[ dataset['Age'] > 64, 'Age' ] = 5

    

train_data.head()
train_data.drop(['AgeBand'], axis=1, inplace=True)

combine = [train_data, test_data]

train_data.head()
for dataset in combine:

    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

    

train_data[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived', ascending=False)
for dataset in combine:

    dataset['IsAlone'] = 0

    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

    

train_data[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean()
train_data = train_data.drop(['SibSp', 'Parch', 'FamilySize'], axis=1)

test_data = test_data.drop(['SibSp', 'Parch', 'FamilySize'], axis=1)

combine = [train_data, test_data]



train_data.head()
freq_port = train_data.Embarked.dropna().mode()[0]

freq_port
for dataset in combine:

    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)

    

train_data[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False)
for dataset in combine:

    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

    

train_data.head()
test_data.isna().sum()
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
X_train = train_data.drop(['Survived'], axis=1)

Y_train = train_data['Survived']

X_test  = test_data.drop(['PassengerId'], axis=1).copy()



X_train.shape, Y_train.shape, X_test.shape
logreg = LogisticRegression()

logreg.fit(X_train, Y_train)

Y_pred = logreg.predict(X_test)

acc_log = round(logreg.score(X_train, Y_train) * 100, 2)

acc_log
coeff_df = pd.DataFrame(train_data.columns.delete(0))

coeff_df.columns = ['Feature']

coeff_df['Correlation'] = pd.Series(logreg.coef_[0])



coeff_df.sort_values(by='Correlation', ascending=False)
decision_tree = DecisionTreeClassifier()

decision_tree.fit(X_train, Y_train)

Y_pred = decision_tree.predict(X_test)

acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)

acc_decision_tree
random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(X_train, Y_train)

Y_pred = random_forest.predict(X_test)

random_forest.score(X_train, Y_train)

acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)

acc_random_forest
models = pd.DataFrame({

    'Model': ['Logistic Regression', 

              'Decision Tree', 

              'Random Forest'],

    'Score': [acc_log,

              acc_decision_tree,

              acc_random_forest]})

models.sort_values(by='Score', ascending=False)
submission = pd.DataFrame({

    'PassengerId': test_data['PassengerId'],

    'Survived': Y_pred

})

submission.to_csv('../input/submission.csv', index=False)