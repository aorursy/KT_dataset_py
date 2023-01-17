#Helps in data analysis

import numpy as np

import pandas as pd



#Helps in data visualisation and plotting

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
gender_submission = pd.read_csv("../input/titanic/gender_submission.csv")

test = pd.read_csv("../input/titanic/test.csv")

train = pd.read_csv("../input/titanic/train.csv")



combine = [train, test]
train.head()
train.shape,test.shape
#correlation matrix

corrmat = train.corr()

f, ax = plt.subplots(figsize=(15, 10))

sns.heatmap(corrmat, vmax=.8, square=True);
train.drop(['PassengerId'], axis=1, inplace=True)
train.describe(include=['O'])



#include=['O'] means we ask to describe only those parts of the training dataset that contain Python objects.
train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train[['SibSp', 'Survived']].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False)
g = sns.FacetGrid(train, col='Survived')

g.map(plt.hist, 'Age', bins=20)
# grid = sns.FacetGrid(train_df, col='Pclass', hue='Survived')

grid = sns.FacetGrid(train, col='Survived', row='Pclass', height=2.2, aspect=1.6)

grid.map(plt.hist, 'Age', alpha=.5, bins=20)

grid.add_legend();
grid = sns.FacetGrid(train, row='Embarked', height=2.2, aspect=1.6)

grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', order=None, hue_order=None, palette='deep',)

grid.add_legend()
# grid = sns.FacetGrid(train_df, col='Embarked', hue='Survived', palette={0: 'k', 1: 'w'})

grid = sns.FacetGrid(train, row='Embarked', col='Survived', height=2.2, aspect=1.6)

grid.map(sns.barplot, 'Sex', 'Fare', order=None, alpha=.5, ci=None)

grid.add_legend()
g = sns.FacetGrid(train, col='Pclass')

g.map(plt.hist, 'Fare', bins=5)
#missing data

total = train.isnull().sum().sort_values(ascending=False)

percent = (train.isnull().sum()/train.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.head(10)
#missing data

total_t = test.isnull().sum().sort_values(ascending=False)

percent_t = (test.isnull().sum()/test.isnull().count()).sort_values(ascending=False)

missing_data_t = pd.concat([total_t, percent_t], axis=1, keys=['Total_Test', 'Percent_Test'])

missing_data_t.head(10)
print("Before", train.shape, test.shape, combine[0].shape, combine[1].shape)



train = train.drop(['Ticket', 'Cabin'], axis=1)

test = test.drop(['Ticket', 'Cabin'], axis=1)

combine = [train, test]



print("After", train.shape, test.shape, combine[0].shape, combine[1].shape)
train.head()
train['Name_length'] = train['Name'].apply(len)

test['Name_length'] = test['Name'].apply(len)
train['Name_lengthCategory'] = pd.cut(train['Name_length'], 4)

train[['Name_lengthCategory', 'Survived']].groupby(['Name_lengthCategory'], as_index=False).mean().sort_values(by='Name_lengthCategory', ascending=True)
for dataset in combine:    

    dataset.loc[(dataset['Name_length'] > 11) & (dataset['Name_length'] <= 29), 'Name_length'] = 0

    dataset.loc[(dataset['Name_length'] > 29) & (dataset['Name_length'] <=47 ), 'Name_length'] = 1

    dataset.loc[(dataset['Name_length'] > 47) & (dataset['Name_length'] <= 64), 'Name_length'] = 2

    dataset.loc[(dataset['Name_length'] > 64), 'Name_length'] = 3

    

train.head()
train = train.drop(['Name_lengthCategory'], axis=1)

combine = [train, test]

train.head()
for dataset in combine:

    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)



pd.crosstab(train['Title'], train['Sex'])
for dataset in combine:

    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')



    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

    

train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()
for dataset in combine:

    title_mapping = {"Master": 1, "Miss": 2, "Mr": 3,  "Mrs": 4,  "Rare": 5}

    dataset['Title'] = dataset['Title'].map(title_mapping)

    dataset['Title'] = dataset['Title'].fillna(0)



train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()
#for dataset in combine:

mean_age_of_small_men=train.loc[train['Title'] ==1, 'Age'].mean()

mean_age_of_small_women=train.loc[train['Title'] ==2, 'Age'].mean()

mean_age_of_men=dataset.loc[dataset['Title'] ==3, 'Age'].mean()

mean_age_of_women=dataset.loc[dataset['Title'] ==4, 'Age'].mean()

mean_age_of_rare=dataset.loc[dataset['Title'] ==5, 'Age'].mean()





mean_age_of_small_men,mean_age_of_small_women,mean_age_of_men,mean_age_of_women,mean_age_of_rare
train.describe()
train = train.drop(['Name'], axis=1)

test = test.drop(['Name'], axis=1)

combine = [train, test]

train.shape, test.shape
for dataset in combine:

    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)



train.head()
grid = sns.FacetGrid(train, row='Pclass', col='Sex', height=2.2, aspect=1.6)

grid.map(plt.hist, 'Age', alpha=.5, bins=20)

grid.add_legend()
guess_ages = np.zeros((2,3))

guess_ages
for dataset in combine:

    for i in range(0, 2):

        for j in range(0, 3):

            guess = dataset[(dataset['Sex'] == i) & \

                                  (dataset['Pclass'] == j+1)]['Age'].dropna()



            # age_mean = guess.mean()

            # age_std = guess.std()

            # age_guess = rnd.uniform(age_mean - age_std, age_mean + age_std)



            age_guess = guess.median()



            # Convert random age float to nearest .5 age

            guess_ages[i,j] = int( age_guess/0.5 + 0.5 ) * 0.5

            

    for i in range(0, 2):

        for j in range(0, 3):

            dataset.loc[ (dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j+1),\

                    'Age'] = guess_ages[i,j]



    dataset['Age'] = dataset['Age'].astype(int)



train.head()
train['AgeCategory'] = pd.cut(train['Age'], 5)

train[['AgeCategory', 'Survived']].groupby(['AgeCategory'], as_index=False).mean().sort_values(by='AgeCategory', ascending=True)
for dataset in combine:    

    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0

    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1

    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2

    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3

    dataset.loc[ dataset['Age'] > 64, 'Age']

train.head()
train = train.drop(['AgeCategory'], axis=1)

combine = [train, test]

train.head()
for dataset in combine:

    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1



train[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived', ascending=False)
for dataset in combine:

    dataset['IsAlone'] = 0

    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1



train[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean()
train = train.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)

test = test.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)

combine = [train, test]



train.head()
for dataset in combine:

    dataset['Age*Class'] = dataset.Age * dataset.Pclass



train.loc[:, ['Age*Class', 'Age', 'Pclass']].head(10)
for dataset in combine:

    dataset['Embarked'] = dataset['Embarked'].fillna(train.Embarked.dropna().mode()[0])



train[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False)
for dataset in combine:

    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)



train.head()
test['Fare'].fillna(test['Fare'].dropna().median(), inplace=True)

test.head()
train['FareCategory'] = pd.qcut(train['Fare'], 4)

print (train[['FareCategory', 'Survived']].groupby(['FareCategory'], as_index=False).mean())
for dataset in combine:

    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0

    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1

    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2

    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3

    dataset['Fare'] = dataset['Fare'].astype(int)



train = train.drop(['FareCategory'], axis=1)

combine = [train, test]

    

train.head(10)
colormap = plt.cm.RdBu

plt.figure(figsize=(14,12))

plt.title('Pearson Correlation of Features', y=1.05, size=15)

sns.heatmap(train.astype(float).corr(),linewidths=0.1,vmax=1.0, 

            square=True, cmap=colormap, linecolor='white', annot=True)
X_train = train.drop("Survived", axis=1)

Y_train = train["Survived"]

X_test  = test.drop("PassengerId", axis=1).copy()

X_train.shape, Y_train.shape, X_test.shape
#models



from sklearn.model_selection import StratifiedShuffleSplit

from sklearn.metrics import accuracy_score, log_loss

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.linear_model import Perceptron

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

from sklearn.linear_model import LogisticRegression

from sklearn.linear_model import SGDClassifier



train.head(10)
# Logistic Regression



logreg = LogisticRegression()

logreg.fit(X_train, Y_train)

Y_pred = logreg.predict(X_test)

acc_log = round(logreg.score(X_train, Y_train) * 100, 2)

acc_log
#SVC Support Vector Classsifier



svc = SVC()

svc.fit(X_train, Y_train)

Y_pred = svc.predict(X_test)

acc_svc = round(svc.score(X_train, Y_train) * 100, 2)

acc_svc
#K Nearest-Neighbours Classifier



knn = KNeighborsClassifier(n_neighbors = 3)

knn.fit(X_train, Y_train)

Y_pred = knn.predict(X_test)

acc_knn = round(knn.score(X_train, Y_train) * 100, 2)

acc_knn
# Gaussian Naive Bayes



gaussian = GaussianNB()

gaussian.fit(X_train, Y_train)

Y_pred = gaussian.predict(X_test)

acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)

acc_gaussian
# Perceptron



perceptron = Perceptron()

perceptron.fit(X_train, Y_train)

Y_pred = perceptron.predict(X_test)

acc_perceptron = round(perceptron.score(X_train, Y_train) * 100, 2)

acc_perceptron
# Stochastic Gradient Descent



sgd = SGDClassifier()

sgd.fit(X_train, Y_train)

Y_pred = sgd.predict(X_test)

acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)

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
models = pd.DataFrame({

    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 

              'Random Forest', 'Naive Bayes', 'Perceptron', 

              'Stochastic Gradient Decent','Decision Tree'],

    'Score': [acc_svc, acc_knn, acc_log, 

              acc_random_forest, acc_gaussian, acc_perceptron, 

              acc_sgd, acc_decision_tree]})

models.sort_values(by='Score', ascending=False)
submission = pd.DataFrame({

        "PassengerId": test["PassengerId"],"Survived": Y_pred})

# submission.to_csv('../output/submission.csv', index=False)

submission.to_csv('titanic.csv', index=False)