# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
#ML imports

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier
train_df = pd.read_csv('/kaggle/input/titanic/train.csv')

test_df = pd.read_csv('/kaggle/input/titanic/test.csv')

combine = [train_df, test_df]
train_df.describe()
train_df.describe(include=['O'])
train_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train_df[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train_df[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)
#boxplots for all variables

# def boxplot(x, y, **kwargs):

#     sns.boxplot(x=x, y=y)

#     x=plt.xticks(rotation=90)



# def fillMissingCatColumns(data,categorical):

#     for c in categorical:

#         data[c] = data[c].astype('category')

#         if data[c].isnull().any():

#             data[c] = data[c].cat.add_categories(['MISSING'])

#             data[c] = data[c].fillna('MISSING')

    

# def getboxPlots(data,var,categorical):

#     fillMissingCatColumns(data,categorical)

#     f = pd.melt(data, id_vars=var, value_vars=categorical)

#     g = sns.FacetGrid(f, col="variable",  col_wrap=2, sharex=False, sharey=False)

#     g = g.map(boxplot, "value", var)

    



# data = train_df.copy()

# categorical = [f for f in data.columns if data.dtypes[f] == 'object']    

# getboxPlots(data,'Survived',categorical)
#Plotting using seaborn to see the relationship of dependent variables

g = sns.FacetGrid(train_df, col='Survived')

g.map(plt.hist, 'Age', bins=20)
grid = sns.FacetGrid(train_df, col='Survived', row='Pclass', size=2.2, aspect=1.6)

grid.map(plt.hist, 'Age', alpha=.5, bins=20)

grid.add_legend()
grid = sns.FacetGrid(train_df, row='Embarked', size=2.2, aspect=1.6)

grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')

grid.add_legend()
grid = sns.FacetGrid(train_df, row='Embarked', col='Survived', size=2.2, aspect=1.6)

grid.map(sns.barplot, 'Sex', 'Fare', alpha=.5, ci=None)

grid.add_legend()
#Plotting using matplotlib

plt.subplot2grid((5,5),(0,0), colspan = 2, rowspan = 2)

train_df['Survived'].value_counts(normalize=True).plot(kind='bar', alpha = 1)

plt.title('% Survived')



plt.subplot2grid((5,5), (0,3), colspan = 2, rowspan = 2)

train_df.Survived[train_df['Sex'] == 'male'].value_counts(normalize = True).plot(kind='bar', alpha = 1)

plt.title('% Men Survived')



plt.subplot2grid((5,5), (3,0), colspan = 2, rowspan = 2)

train_df.Survived[train_df['Sex']=='female'].value_counts(normalize=True).plot(kind='bar', alpha = 1, color ='#FA0193')

plt.title('% Women Survived')



plt.subplot2grid((5,5), (3,3), colspan = 2, rowspan = 2)

train_df.Sex[train_df['Survived'] == 1].value_counts(normalize= True).plot(kind='bar', alpha = 1, color = ['#FA0193', '#000000'])

plt.title('% Gender Survived')
#Cleaning the data

train_df = train_df.drop(['Ticket', 'Cabin'], axis=1)

test_df = test_df.drop(['Ticket', 'Cabin'], axis=1)

combine = [train_df, test_df]
#Extracting title from name

for dataset in combine:

    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)



pd.crosstab(train_df['Title'], train_df['Sex'])
#Changing title from categorical to ordinal variable

for dataset in combine:

    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',\

 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')



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
for dataset in combine:

    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)
grid = sns.FacetGrid(train_df, row='Pclass', col='Sex', size=2.2, aspect=1.6)

grid.map(plt.hist, 'Age', alpha=.5, bins=20)

grid.add_legend()
guess_ages = np.zeros((2,3))

guess_ages
#Use age and pclass to guess the age of the null values

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
#Creating agebands

train_df['AgeBand'] = pd.cut(train_df['Age'], 5)

train_df[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True)
#Convert agebands to ordinal variables

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
#Making use of famil size as a predictor

for dataset in combine:

    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1



train_df[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived', ascending=False)
#Create a column based on family size

for dataset in combine:

    dataset['IsAlone'] = 0

    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1



train_df[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean()
train_df = train_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)

test_df = test_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)

combine = [train_df, test_df]



train_df.head()
#Create feature combining age and pclass

for dataset in combine:

    dataset['Age*Class'] = dataset.Age * dataset.Pclass



train_df.loc[:, ['Age*Class', 'Age', 'Pclass']].head(10)
#Cleaning null data

for dataset in combine:

    dataset['Embarked'] = dataset['Embarked'].fillna('S')

    

train_df[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False)
for dataset in combine:

    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True)

test_df.head()
#creating fare bands just like age bands

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

    

train_df.head(10)
#Model, predict and solve

X_train = train_df.drop("Survived", axis=1)

Y_train = train_df["Survived"]

X_test  = test_df.drop("PassengerId", axis=1).copy()
#for polynomial preprocessing

from sklearn import preprocessing
poly = preprocessing.PolynomialFeatures(degree=2)

X_train_poly = poly.fit_transform(X_train)

X_test_poly = poly.fit_transform(X_test)
#Random Forest Classifier Algo

from sklearn.ensemble import RandomForestClassifier
#For choosing the best parameters, we can run the GridSearchCV across all parameters 

from sklearn.model_selection import GridSearchCV

params = {'n_estimators':np.arange(5, 110, 5), 'max_depth':np.arange(3, 10)}



grid_search_rfc = GridSearchCV(RandomForestClassifier(random_state=42), params, cv=5, scoring='neg_mean_squared_error')

grid_search_rfc.fit(X_train_poly, Y_train)

grid_search_rfc.best_params_
rf = RandomForestClassifier(random_state = 42, **grid_search_rfc.best_params_)

rf.fit(X_train_poly, Y_train)

rf.score(X_train_poly, Y_train)
#Model might overfit so we need to do cross validation of the model

#To check for overfitting, model is split and run cv times and mean accuracy is checked

from sklearn import model_selection

#Find the mean of scores from 50 samples

#with polynomial features

scores = model_selection.cross_val_score(rf, X_train_poly, Y_train, scoring = 'accuracy', cv=50)

scores.mean()
#Gradient boosting algorithm

import xgboost as xgb
#Getting best parameters for xgboost

xgb_model = xgb.XGBClassifier()

clf = GridSearchCV(xgb_model,

                   {'max_depth': [2,4,6],

                    'n_estimators': [50,100,200]}, verbose=1)

clf.fit(X_train_poly,Y_train)

print(clf.best_score_)

print(clf.best_params_)
#Test Train splitting for xgboost classifier

from sklearn.model_selection import train_test_split

X_train1, X_test1, y_train1, y_test1 = train_test_split(X_train, Y_train, random_state=0)

clf = xgb.XGBClassifier()

clf.fit(X_train1, y_train1, early_stopping_rounds=10, eval_metric="auc",

        eval_set=[(X_test1, y_test1)])

clf.score(X_train1, y_train1)
#Cross validation

scores = model_selection.cross_val_score(clf, X_train, Y_train, scoring = 'accuracy', cv=50)

scores.mean()
#Logistic regression model

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()

lr1 = lr.fit(X_train_poly, Y_train)

lr1.score(X_train_poly, Y_train)
#Cross validation

scores = model_selection.cross_val_score(lr, X_train_poly, Y_train, scoring = 'accuracy', cv=50)

scores

scores.mean()
Y_pred = clf.predict(X_test)
submission = pd.DataFrame({

    "PassengerId" : test_df["PassengerId"],

    "Survived" : Y_pred

})
submission.to_csv('gender_submission.csv', index=False)