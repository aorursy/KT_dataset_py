# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_df = pd.read_csv("../input/titanic/train.csv")

test_df = pd.read_csv("../input/titanic/test.csv")

combine = [train_df, test_df]
train_df.head()
train_df.describe()
train_df.describe(include=["O"]) # select categorical columns
train_df[["Pclass", "Survived"]].groupby(["Pclass"], as_index=False).mean().sort_values(by="Survived", ascending=False)
train_df[["Sex", "Survived"]].groupby(["Sex"], as_index=False).mean().sort_values(by="Survived", ascending=False)
train_df[["SibSp", "Survived"]].groupby(["SibSp"], as_index=False).mean().sort_values(by="Survived", ascending=False)
train_df[["Parch", "Survived"]].groupby(["Parch"], as_index=False).mean().sort_values(by="Survived", ascending=False)
import seaborn as sns

import matplotlib.pyplot as plt
g = sns.FacetGrid(train_df, col='Survived')

g.map(plt.hist, 'Age', bins=20)
grid = sns.FacetGrid(train_df, col='Survived', row='Pclass', height=2.2, aspect=1.6)

grid.map(plt.hist, 'Age', bins=20)

grid.add_legend()
grid = sns.FacetGrid(train_df, row='Embarked', height=2.2, aspect=1.6)

grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')

grid.add_legend()
grid = sns.FacetGrid(train_df, row='Embarked', col='Survived', height=2.2, aspect=1.6)

grid.map(sns.barplot, 'Sex', 'Fare', ci=None)

grid.add_legend()
train_df = train_df.drop(['Ticket', 'Cabin'], axis=1)

test_df = test_df.drop(['Ticket', 'Cabin'], axis=1)

combine = [train_df, test_df]
for dataset in combine:

    dataset['Title'] = dataset.Name.str.extract('(\w+)\.', expand=False)



pd.crosstab(train_df['Title'], train_df['Sex'])
rare_features = ['Lady', 'Countess','Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona']



for dataset in combine:

    dataset['Title'] = dataset['Title'].replace(rare_features, 'Rare')



    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

    

train_df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

for dataset in combine:

    dataset['Title'] = dataset['Title'].map(title_mapping)

    dataset['Title'] = dataset['Title'].fillna(0)



train_df.head()
train_df = train_df.drop(['Name', 'PassengerId'], axis=1)

test_df = test_df.drop(['Name'], axis=1)

combine = [train_df, test_df]

train_df.shape, test_df.shape
for dataset in combine:

    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)



train_df.head()
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

train_df[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand', ascending=True)
for dataset in combine:

    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0

    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1

    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2

    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3

    dataset['Fare'] = dataset['Fare'].astype(int)



train_df = train_df.drop(['FareBand'], axis=1)

combine = [train_df, test_df]

    

train_df.head()
grid = sns.FacetGrid(train_df, row='Pclass', col='Sex', height=2.2, aspect=1.6)

grid.map(plt.hist, 'Age', bins=20)

grid.add_legend()
guess_ages = np.zeros((2,3))

guess_ages
for dataset in combine:

    for i in range(0, 2):

        for j in range(0, 3):

            guess_df = dataset[(dataset['Sex'] == i) & \

                                  (dataset['Pclass'] == j+1)]['Age'].dropna()



            age_guess = guess_df.median()



            # Convert random age float to nearest .5 age

            guess_ages[i,j] = int( age_guess/0.5 + 0.5 ) * 0.5

            

    for i in range(0, 2):

        for j in range(0, 3):

            dataset.loc[ (dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j+1),\

                    'Age'] = guess_ages[i, j]



    dataset['Age'] = dataset['Age'].astype(int)



train_df.head()
train_df['AgeBand'] = pd.cut(train_df['Age'], 5)

train_df[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True)
for dataset in combine:    

    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0

    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1

    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2

    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3

    dataset.loc[ dataset['Age'] > 64, 'Age']



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
train_df.head(10)
X_train = train_df.drop("Survived", axis=1)

y_train = train_df["Survived"]

X_test = test_df.drop("PassengerId", axis=1).copy()

X_train.shape, y_train.shape, X_test.shape
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import cross_val_score





logreg = LogisticRegression()

logreg.fit(X_train, y_train)

acc_log = round(cross_val_score(logreg, X_train, y_train, cv=5).mean() * 100, 2)

acc_log
coeff_df = pd.DataFrame(train_df.columns.delete(0))

coeff_df.columns = ['Feature']

coeff_df["Correlation"] = pd.Series(logreg.coef_[0])



coeff_df.sort_values(by='Correlation', ascending=False)
from sklearn.model_selection import GridSearchCV





tuned_logreg = LogisticRegression()

param_grid = {'max_iter' : [2000],

              'penalty' : ['l1', 'l2'],

              'C' : np.logspace(-4, 4, 20),

              'solver' : ['liblinear']}



grid_search = GridSearchCV(tuned_logreg, param_grid=param_grid, cv=5, verbose=True, n_jobs=-1)

best_logreg = grid_search.fit(X_train,y_train)

acc_best_logreg = round(best_logreg.best_score_ * 100, 2)

acc_best_logreg
from sklearn.svm import SVC





svc = SVC()

svc.fit(X_train, y_train)

acc_svc = round(cross_val_score(svc, X_train, y_train, cv=5).mean() * 100, 2)

acc_svc
tuned_svc = SVC()

param_grid = [{'kernel': ['rbf'], 'gamma': [.1,.5,1,2,5,10], 'C': [.1, 1, 10, 100, 1000]},

              {'kernel': ['linear'], 'C': [.1, 1, 10, 100, 1000]},

              {'kernel': ['poly'], 'degree' : [2,3,4,5], 'C': [.1, 1, 10, 100, 1000]}]



grid_search = GridSearchCV(tuned_svc, param_grid=param_grid, cv=5, verbose=True, n_jobs=-1)

best_svc = grid_search.fit(X_train, y_train)

acc_best_svc = round(best_svc.best_score_ * 100, 2)

acc_best_svc
from sklearn.svm import LinearSVC





lin_svc = LinearSVC()

lin_svc.fit(X_train, y_train)

acc_lin_svc = round(cross_val_score(lin_svc, X_train, y_train, cv=5).mean() * 100, 2)

acc_lin_svc
tuned_lin_svc = LinearSVC()

param_grid = {'C' : [.1, 1, 10, 100, 1000]}



grid_search = GridSearchCV(tuned_lin_svc, param_grid=param_grid, cv=5, verbose=True, n_jobs=-1)

best_lin_svc = grid_search.fit(X_train, y_train)

acc_best_lin_svc = round(best_lin_svc.best_score_ * 100, 2)

acc_best_lin_svc
from sklearn.neighbors import KNeighborsClassifier





knn = KNeighborsClassifier()

knn.fit(X_train, y_train)

acc_knn = round(cross_val_score(knn, X_train, y_train, cv=5).mean() * 100, 2)

acc_knn
tuned_knn = KNeighborsClassifier()

param_grid = {'n_neighbors' : [3,5,7,9],

              'weights' : ['uniform', 'distance'],

              'algorithm' : ['auto', 'ball_tree','kd_tree'],

              'p' : [1,2]}



grid_search = GridSearchCV(tuned_knn, param_grid = param_grid, cv = 5, verbose = True, n_jobs = -1)

best_knn = grid_search.fit(X_train, y_train)

acc_best_knn = round(best_knn.best_score_ * 100, 2)

acc_best_knn
from sklearn.ensemble import RandomForestClassifier





random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(X_train, y_train)

acc_random_forest = round(cross_val_score(random_forest, X_train, y_train).mean() * 100, 2)

acc_random_forest
tuned_rf = RandomForestClassifier(random_state = 1)

param_grid =  {'n_estimators': [400,450,500,550],

               'criterion':['gini','entropy'],

                            'bootstrap': [True],

                            'max_depth': [15, 20, 25],

                            'max_features': ['auto','sqrt', 10],

                            'min_samples_leaf': [2,3],

                            'min_samples_split': [2,3]}

                                  

grid_search = GridSearchCV(tuned_rf, param_grid = param_grid, cv=5, verbose=True, n_jobs=-1)

best_rf = grid_search.fit(X_train, y_train)

acc_best_rf = round(best_rf.best_score_ * 100, 2)

acc_best_rf
models = pd.DataFrame({

    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 'Random Forest', 'Linear SVC'],

    'Score': [acc_svc, acc_knn, acc_log, acc_random_forest, acc_lin_svc],

    'Tuned Score': [acc_best_svc, acc_best_knn, acc_best_logreg, acc_best_rf, acc_best_lin_svc]})



models.sort_values(by='Tuned Score', ascending=False)
best_pred = best_svc.predict(X_test)
submission = pd.DataFrame({

    "PassengerId": test_df["PassengerId"],

    "Survived": best_pred

})

submission.to_csv("submission.csv", index=False)