# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



#from subprocess import check_output

#print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/train.csv")

test_df = pd.read_csv("../input/test.csv")



#df = pd.read_csv("titanic_train.csv")

#test_df = pd.read_csv("titanic_test.csv")
df.info()
df.head(5)
df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)
df[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
df[['SibSp', 'Survived']].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)
df[['Parch', 'Survived']].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)
g = sns.FacetGrid(df, col='Survived')

g.map(plt.hist, 'Age', bins=20)
grid = sns.FacetGrid(df, col='Survived', row='Pclass', size=2.2, aspect=1.6)

grid.map(plt.hist, 'Age', alpha=.5, bins=20)

grid.add_legend();
grid = sns.FacetGrid(df, row='Embarked', size=2.2, aspect=1.6)

grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')

grid.add_legend()
grid = sns.FacetGrid(df, row='Embarked', col='Survived', size=2.2, aspect=1.6)

grid.map(sns.barplot, 'Sex', 'Fare', alpha = .5, ci=None)

grid.add_legend()
print("Before:", df.shape, test_df.shape)



df = df.drop(['Cabin', 'Ticket'], axis=1)

test_df = test_df.drop(['Cabin', 'Ticket'], axis=1)



print("After:", df.shape, test_df.shape)
df['Title'] = df.Name.str.extract('([A-Za-z]+)\.', expand=False)

test_df['Title'] = test_df.Name.str.extract('([A-Za-z]+)\.', expand=False)
pd.crosstab(df['Title'], df['Sex'])
df['Title'] = df['Title'].replace(['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

df['Title'] = df['Title'].replace(['Mlle', 'Ms'], 'Miss')

df['Title'] = df['Title'].replace('Mme', 'Mrs')



test_df['Title'] = test_df['Title'].replace(['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

test_df['Title'] = test_df['Title'].replace(['Mlle', 'Ms'], 'Miss')

test_df['Title'] = test_df['Title'].replace('Mme', 'Mrs')



df[['Title', 'Survived']].groupby('Title', as_index=False).mean()
title_mapping = {'Mr': 1, 'Miss': 2, 'Mrs': 3, 'Master': 4, 'Rare': 5}



df['Title'] = df['Title'].map(title_mapping)

df['Title'] = df['Title'].fillna(0)

test_df['Title'] = test_df['Title'].map(title_mapping)

test_df['Title'] = test_df['Title'].fillna(0)



df.head()
# save PassengerId

passenger_id = test_df['PassengerId']



df = df.drop(['PassengerId', 'Name'], axis=1)

test_df = test_df.drop(['PassengerId', 'Name'], axis=1)



df.shape, test_df.shape
df.head()
sex_mapping = {'female': 1, 'male': 0}



df['Sex'] = df['Sex'].map(sex_mapping).astype(int)

test_df['Sex'] = test_df['Sex'].map(sex_mapping).astype(int)
df.head()
pd.crosstab(df['Pclass'], df['Sex'])
grid = sns.FacetGrid(df, row='Pclass', col='Sex', size=2.2, aspect=1.6)

grid.map(plt.hist, 'Age', alpha=.5, bins=20)

grid.add_legend()
guess_ages_train = np.zeros((2, 3))

guess_ages_test = np.zeros((2, 3))
for i in range(0, 2):

    for j in range(0, 3):

        # train df

        guess = df[(df['Sex'] == i) & (df['Pclass'] == j+1)]['Age'].dropna()

        

        age_guess = guess.median()

        

        # round

        guess_ages_train[i, j] = int(age_guess / 0.5 + 0.5) * 0.5

        

        df.loc[(df['Age'].isnull()) & (df['Sex'] == i) & (df['Pclass'] == j+1), 'Age'] = guess_ages_train[i, j]

        

        # test df

        guess = test_df[(test_df['Sex'] == i) & (test_df['Pclass'] == j+1)]['Age'].dropna()

        

        age_guess = guess.median()

        

        # round

        guess_ages_test[i, j] = int(age_guess / 0.5 + 0.5) * 0.5

        

        test_df.loc[(test_df['Age'].isnull()) & (test_df['Sex'] == i) & (test_df['Pclass'] == j+1), 'Age'] = guess_ages_test[i, j]

        

df['Age'] = df['Age'].astype(int)      

test_df['Age'] = test_df['Age'].astype(int)
df.head(20)
df['AgeBand'] = pd.cut(df['Age'], 5)

df[['AgeBand', 'Survived']].groupby('AgeBand', as_index=False).mean()
df.loc[df['Age'] <= 16, 'Age'] = 0

df.loc[(df['Age'] > 16) & (df['Age'] <= 32), 'Age'] = 1

df.loc[(df['Age'] > 32) & (df['Age'] <= 48), 'Age'] = 2

df.loc[(df['Age'] > 48) & (df['Age'] <= 64), 'Age'] = 3

df.loc[df['Age'] > 64, 'Age'] = 4



test_df.loc[test_df['Age'] <= 16, 'Age'] = 0

test_df.loc[(test_df['Age'] > 16) & (test_df['Age'] <= 32), 'Age'] = 1

test_df.loc[(test_df['Age'] > 32) & (test_df['Age'] <= 48), 'Age'] = 2

test_df.loc[(test_df['Age'] > 48) & (test_df['Age'] <= 64), 'Age'] = 3

test_df.loc[test_df['Age'] > 64, 'Age'] = 4



df.head()
df = df.drop(['AgeBand'], axis=1)



df.head()
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

test_df['FamilySize'] = test_df['SibSp'] + test_df['Parch'] + 1



df[['FamilySize', 'Survived']].groupby('FamilySize', as_index=False).mean().sort_values(by='Survived', ascending=False)
df['IsAlone'] = 0

test_df['IsAlone'] = 0



df.loc[df['FamilySize'] == 1, 'IsAlone'] = 1

test_df.loc[test_df['FamilySize'] == 1, 'IsAlone'] = 1
df.head()
#df = df.drop(['SibSp', 'Parch', 'FamilySize'], axis=1)

#test_df = test_df.drop(['SibSp', 'Parch', 'FamilySize'], axis=1)
df['Age*Class'] = df['Age'] * df['Pclass']

test_df['Age*Class'] = test_df['Age'] * df['Pclass']
df.loc[:, ['Age*Class', 'Age', 'Pclass']].head(10)
df.info()
freq_port = df['Embarked'].dropna().mode()[0]

freq_port
df['Embarked'] = df['Embarked'].fillna(freq_port)

test_df['Embarked'] = test_df['Embarked'].fillna(freq_port)



df[['Embarked', 'Survived']].groupby('Embarked', as_index=False).mean().sort_values(by='Survived', ascending=False)
df['Embarked'] = df['Embarked'].map( {'S': 0, 'Q': 1, 'C': 2} ).astype(int)

df.head()
test_df['Embarked'] = test_df['Embarked'].map( {'S': 0, 'Q': 1, 'C': 2} ).astype(int)

test_df.head()
test_df.info()
test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True)

test_df.info()
test_df.head()
df['FareBand'] = pd.qcut(df['Fare'], 4)

df[['FareBand', 'Survived']].groupby('FareBand', as_index=False).mean().sort_values(by='Survived', ascending=True)
df.loc[df['Fare'] <= 7.91, 'Fare'] = 0

df.loc[(df['Fare'] > 7.91) & (df['Fare'] <= 14.454), 'Fare'] = 1

df.loc[(df['Fare'] > 14.454) & (df['Fare'] <= 31), 'Fare'] = 2

df.loc[df['Fare'] > 31, 'Fare'] = 3

df['Fare'] = df['Fare'].astype(int)



test_df.loc[test_df['Fare'] <= 7.91, 'Fare'] = 0

test_df.loc[(test_df['Fare'] > 7.91) & (test_df['Fare'] <= 14.454), 'Fare'] = 1

test_df.loc[(test_df['Fare'] > 14.454) & (test_df['Fare'] <= 31), 'Fare'] = 2

test_df.loc[test_df['Fare'] > 31, 'Fare'] = 3

test_df['Fare'] = test_df['Fare'].astype(int)



df = df.drop('FareBand', axis=1)



df.head()
test_df.head()
# machine learning

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier
X_train = df.drop('Survived', axis=1)

Y_train = df['Survived']

X_test = test_df

X_train.shape, Y_train.shape, X_test.shape
logreg = LogisticRegression()

logreg.fit(X_train, Y_train)

Y_pred_logreg = logreg.predict(X_test)

acc_log = round(logreg.score(X_train, Y_train), 4)

acc_log
coeff_df = pd.DataFrame(df.columns.delete(0))

coeff_df.columns = ['Feature']

coeff_df['Coefficient'] = pd.Series(logreg.coef_[0])



coeff_df.sort_values(by='Coefficient', ascending=False)
svc = SVC()

svc.fit(X_train, Y_train)

Y_pred_svc = svc.predict(X_test)

acc_svc = round(svc.score(X_train, Y_train), 4)

acc_svc
knn = KNeighborsClassifier(n_neighbors=3)

knn.fit(X_train, Y_train)

Y_pred_knn = knn.predict(X_test)

acc_knn = round(knn.score(X_train, Y_train), 4)

acc_knn
gaussian = GaussianNB()

gaussian.fit(X_train, Y_train)

Y_pred_gaussian = gaussian.predict(X_test)

acc_gaussian = round(gaussian.score(X_train, Y_train), 4)

acc_gaussian
perceptron = Perceptron()

perceptron.fit(X_train, Y_train)

Y_pred_perceptron = perceptron.predict(X_test)

acc_perceptron = round(perceptron.score(X_train, Y_train), 4)

acc_perceptron
linear_svc = LinearSVC()

linear_svc.fit(X_train, Y_train)

Y_pred_linear_svc = linear_svc.predict(X_test)

acc_linear_svc = round(linear_svc.score(X_train, Y_train), 4)

acc_linear_svc
sgd = SGDClassifier()

sgd.fit(X_train, Y_train)

Y_pred_sgd = sgd.predict(X_test)

acc_sgd = round(sgd.score(X_train, Y_train), 4)

acc_sgd
decision_tree = DecisionTreeClassifier()

decision_tree.fit(X_train, Y_train)

Y_pred_decision_tree = decision_tree.predict(X_test)

acc_decision_tree = round(decision_tree.score(X_train, Y_train), 4)

acc_decision_tree
random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(X_train, Y_train)

Y_pred_random_forest = random_forest.predict(X_test)

acc_random_forest = round(random_forest.score(X_train, Y_train), 4)

acc_random_forest
models = pd.DataFrame({

        'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 

                  'Random Forest', 'Naive Bayes', 'Perceptron', 

                  'Stochastic Gradient Decent', 'Linear SVC', 

                  'Decision Tree'],

        'Score': [acc_svc, acc_knn, acc_log, 

                  acc_random_forest, acc_gaussian, acc_perceptron, 

                  acc_sgd, acc_linear_svc, acc_decision_tree]

    })



models.sort_values(by='Score', ascending=False)
submission = pd.DataFrame({

        'PassengerId': passenger_id,

        'Survived': Y_pred_random_forest

    })



submission.to_csv('../output/submission.csv', index=False)