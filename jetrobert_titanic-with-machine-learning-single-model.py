import os

import pandas as pd

import numpy as np

import random as rnd



# visualization

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



# machine learning

from sklearn.model_selection import GridSearchCV, train_test_split, StratifiedKFold

from sklearn.metrics import accuracy_score, classification_report

from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler

from sklearn.linear_model import LogisticRegression

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.svm import SVC, LinearSVC

from sklearn.naive_bayes import GaussianNB

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier





#ignore warnings

import warnings

warnings.filterwarnings('ignore')
for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train_data = pd.read_csv("/kaggle/input/titanic/train.csv")

test_data = pd.read_csv("/kaggle/input/titanic/test.csv")
test_data.head()

train_data.head()
print(train_data.shape)

print(test_data.shape)
# feature names

print(train_data.columns.values)
train_data.info()
train_data.describe()
train_data[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train_data[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train_data[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train_data[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
women = train_data.loc[train_data.Sex == 'female']["Survived"]

rate_women = sum(women)/len(women)



print("% of women who survived:", rate_women)
men = train_data.loc[train_data.Sex == 'male']["Survived"]

rate_men = sum(men)/len(men)



print("% of men who survived:", rate_men)
sns.factorplot(x='Survived', col='Sex', kind='count', data=train_data)
g = sns.FacetGrid(train_data, col='Survived')

g.map(plt.hist, 'Age', bins=20)
# grid = sns.FacetGrid(train_df, col='Pclass', hue='Survived')

grid = sns.FacetGrid(train_data, col='Survived', row='Pclass', size=2.2, aspect=1.6)

grid.map(plt.hist, 'Age', alpha=.5, bins=20)

grid.add_legend()
# grid = sns.FacetGrid(train_df, col='Embarked')

grid = sns.FacetGrid(train_data, row='Embarked', size=2.2, aspect=1.6)

grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')

grid.add_legend()
# grid = sns.FacetGrid(train_df, col='Embarked', hue='Survived', palette={0: 'k', 1: 'w'})

grid = sns.FacetGrid(train_data, row='Embarked', col='Survived', size=2.2, aspect=1.6)

grid.map(sns.barplot, 'Sex', 'Fare', alpha=.5, ci=None)

grid.add_legend()
train_data.isnull().sum()
test_data.isnull().sum()
med_fare = test_data.groupby(['Pclass', 'Parch', 'SibSp']).Fare.median()[3][0][0]

# Filling the missing value in Fare with the median Fare of 3rd class alone passenger

test_data['Fare'] = test_data['Fare'].fillna(med_fare)
train_data['Age'] = train_data.groupby(['Sex', 'Pclass'])['Age'].apply(lambda x: x.fillna(x.median()))

test_data['Age'] = test_data.groupby(['Sex', 'Pclass'])['Age'].apply(lambda x: x.fillna(x.median()))
train_data['Embarked'] = train_data['Embarked'].fillna('S')

test_data['Embarked'] = test_data['Embarked'].fillna('S')
train_data['Deck'] = train_data['Cabin'].apply(lambda s: s[0] if pd.notnull(s) else 'M')

test_data['Deck'] = test_data['Cabin'].apply(lambda s: s[0] if pd.notnull(s) else 'M')



train_data['Deck'] = train_data['Deck'].replace(['A', 'B', 'C'], 'ABC')

train_data['Deck'] = train_data['Deck'].replace(['D', 'E'], 'DE')

train_data['Deck'] = train_data['Deck'].replace(['F', 'G'], 'FG')



test_data['Deck'] = test_data['Deck'].replace(['A', 'B', 'C'], 'ABC')

test_data['Deck'] = test_data['Deck'].replace(['D', 'E'], 'DE')

test_data['Deck'] = test_data['Deck'].replace(['F', 'G'], 'FG')



test_data['Deck'].value_counts()
# combine = [train_data, test_data]

# print("Before", train_data.shape, test_data.shape, combine[0].shape, combine[1].shape)



train_data = train_data.drop(['Cabin'], axis=1)

test_data = test_data.drop(['Cabin'], axis=1)

combine = [train_data, test_data]



print("After", train_data.shape, test_data.shape, combine[0].shape, combine[1].shape)
for dataset in combine:

    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)



pd.crosstab(train_data['Title'], train_data['Sex'])
for dataset in combine:

    dataset['Title'] = dataset['Title'].replace(['Miss', 'Mrs','Ms', 'Mlle', 'Lady', 'Mme', 'the Countess', 'Dona'], 

                                                'Miss/Mrs/Ms')

    dataset['Title'] = dataset['Title'].replace(['Dr', 'Col', 'Major', 'Jonkheer', 'Capt', 'Sir', 'Don', 'Rev'], 

                                                'Dr/Military/Noble/Clergy')

#     dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',\

#                                                  'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')



#     dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')

#     dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')

#     dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

    

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
# grid = sns.FacetGrid(train_df, col='Pclass', hue='Gender')

grid = sns.FacetGrid(train_data, row='Pclass', col='Sex', size=2.2, aspect=1.6)

grid.map(plt.hist, 'Age', alpha=.5, bins=20)

grid.add_legend()
guess_ages = np.zeros((2,3))

guess_ages
for dataset in combine:

    for i in range(0, 2):

        for j in range(0, 3):

            guess_data = dataset[(dataset['Sex'] == i) & \

                                  (dataset['Pclass'] == j+1)]['Age'].dropna()



            # age_mean = guess_df.mean()

            # age_std = guess_df.std()

            # age_guess = rnd.uniform(age_mean - age_std, age_mean + age_std)



            age_guess = guess_data.median()



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

    dataset.loc[dataset['Age'] <= 16, 'Age'] = 0

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



train_data[['FamilySize', 'Survived']].groupby(['FamilySize'], 

                                               as_index=False).mean().sort_values(by='Survived', ascending=False)
for dataset in combine:

    dataset['IsAlone'] = 0

    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1



train_data[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean()
# train_data = train_data.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)

# test_data = test_data.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)

# combine = [train_data, test_data]



# train_data.head()
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
test_data.head(10)
train_data['Ticket'] = LabelEncoder().fit_transform(train_data['Ticket'])

train_data['Deck'] = LabelEncoder().fit_transform(train_data['Deck'])



test_data['Ticket'] = LabelEncoder().fit_transform(test_data['Ticket'])

test_data['Deck'] = LabelEncoder().fit_transform(test_data['Deck'])
X_train = train_data.drop("Survived", axis=1)

Y_train = train_data["Survived"]

X_test  = test_data.copy()

X_test = X_test.drop(['PassengerId'], axis=1)

X_train.shape, Y_train.shape, X_test.shape



X_train = StandardScaler().fit_transform(X_train)

X_test = StandardScaler().fit_transform(X_test)
X_train.shape
train_x, val_x, train_y, val_y = train_test_split(X_train, Y_train, random_state = 0)

kfold = StratifiedKFold(n_splits=10)

val_y
# Logistic Regression



logreg = LogisticRegression()

logreg.fit(train_x, train_y)

pred_y = logreg.predict(val_x)

acc = accuracy_score(val_y, pred_y)

acc
coeff_df = pd.DataFrame(train_data.columns.delete(0))

coeff_df.columns = ['Feature']

coeff_df["Correlation"] = pd.Series(logreg.coef_[0])

coeff_df.sort_values(by='Correlation', ascending=False)
logreg = LogisticRegression()

n_folds = 10

clf = GridSearchCV(logreg, param_grid = {}, cv=n_folds, refit=True)

clf.fit(train_x, train_y)

pred_y = clf.predict(val_x)

acc = accuracy_score(val_y, pred_y)

cm = classification_report(val_y, pred_y)

print(acc)

print(cm)
# Support Vector Machines



svc = SVC()



n_folds = 10

clf = GridSearchCV(svc, param_grid = {}, cv=n_folds, refit=True)

clf.fit(train_x, train_y)

pred_y = clf.predict(val_x)

acc = accuracy_score(val_y, pred_y)

cm = classification_report(val_y, pred_y)

print(acc)

print(cm)
# KNN



knn = KNeighborsClassifier(algorithm='auto', 

                           leaf_size=30,

                           metric='minkowski',                                           

                           metric_params=None, 

                           n_neighbors=7, 

                           p=2,

                           n_jobs=4,

                           weights='uniform')

                                            

n_folds = 10

clf = GridSearchCV(knn, param_grid = {}, cv=n_folds, refit=True)

clf.fit(train_x, train_y)

pred_y = clf.predict(val_x)

acc = accuracy_score(val_y, pred_y)

cm = classification_report(val_y, pred_y)

print(acc)

print(cm)
# Gaussian Naive Bayes



gaussian = GaussianNB()

n_folds = 10

clf = GridSearchCV(gaussian, param_grid = {}, cv=n_folds, refit=True)

clf.fit(train_x, train_y)

pred_y = clf.predict(val_x)

acc = accuracy_score(val_y, pred_y)

cm = classification_report(val_y, pred_y)

print(acc)

print(cm)
# Perceptron



perceptron = Perceptron()

n_folds = 10

clf = GridSearchCV(perceptron, param_grid = {}, cv=n_folds, refit=True)

clf.fit(train_x, train_y)

pred_y = clf.predict(val_x)

acc = accuracy_score(val_y, pred_y)

cm = classification_report(val_y, pred_y)

print(acc)

print(cm)
# Stochastic Gradient Descent



sgd = SGDClassifier()

n_folds = 10

clf = GridSearchCV(sgd, param_grid = {}, cv=n_folds, refit=True)

clf.fit(train_x, train_y)

pred_y = clf.predict(val_x)

acc = accuracy_score(val_y, pred_y)

cm = classification_report(val_y, pred_y)

print(acc)

print(cm)
# Decision Tree



dt = DecisionTreeClassifier(random_state=0)

n_folds = 10

clf = GridSearchCV(dt, param_grid = {}, cv=n_folds, refit=True)

clf.fit(train_x, train_y)

pred_y = clf.predict(val_x)

acc = accuracy_score(val_y, pred_y)

cm = classification_report(val_y, pred_y)

print(acc)

print(cm)
Y_pred = clf.predict(X_test)

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': Y_pred})

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")