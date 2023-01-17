# data analysis and wrangling

import pandas as pd

import numpy as np

import random as rnd



# visualization

import seaborn as sns

import matplotlib.pyplot as plt

import warnings; warnings.simplefilter('ignore')

%matplotlib inline



# machine learning

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.tree import DecisionTreeClassifier

from xgboost import XGBClassifier

from sklearn.preprocessing import OneHotEncoder, LabelEncoder

from sklearn import feature_selection

from sklearn import model_selection

from sklearn import metrics

from sklearn.metrics import classification_report
train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')

combine = [train_df, test_df]
train_df.head()
print(train_df.dtypes)

print('-'*50)

print(test_df.dtypes)
train_df.describe()
train_df.describe(include=['O'])
train_df.describe(include= 'all')
train_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train_df[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train_df[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train_df[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)
sns.set_style('darkgrid')

plt.figure(figsize=(100, 60))



g= sns.FacetGrid(train_df, col= 'Survived')

g.map(plt.hist, 'Age', bins= 20)
grid = sns.FacetGrid(train_df, col= 'Survived', row= 'Pclass', size= 2.2, aspect= 1.5)

grid.map(plt.hist, 'Age', alpha= 0.5, bins= 20)

grid.add_legend();
grid = sns.FacetGrid(train_df, row='Embarked', size=2.2, aspect=1.5)

grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')

grid.add_legend()
grid = sns.FacetGrid(train_df, row= 'Embarked', col= 'Survived', size= 2.2, aspect= 1.6)

grid.map(sns.barplot, 'Sex', 'Fare', alpha= 0.5, ci=None)

grid.add_legend()
print("Before", train_df.shape, test_df.shape, combine[0].shape, combine[1].shape)



train_df = train_df.drop(['Ticket', 'Cabin'], axis=1)

test_df = test_df.drop(['Ticket', 'Cabin'], axis=1)

combine = [train_df, test_df]



"After", train_df.shape, test_df.shape, combine[0].shape, combine[1].shape

train_df.head()

test_df.head()

test_df.shape
train_df.isnull().sum()
test_df.isnull().sum()
for dataset in combine:    

    #complete missing age with median

    dataset['Age'].fillna(dataset['Age'].median(), inplace = True)



    #complete embarked with mode

    dataset['Embarked'].fillna(dataset['Embarked'].mode()[0], inplace = True)



    #complete missing fare with median

    dataset['Fare'].fillna(dataset['Fare'].median(), inplace = True)

    

#delete the cabin feature/column and others previously stated to exclude in train dataset

drop_column = ['PassengerId']

train_df.drop(drop_column, axis=1, inplace = True)



print(train_df.isnull().sum())

print("-"*10)

print(test_df.isnull().sum())
###CREATE: Feature Engineering for train and test dataset

for dataset in combine:    

    #Discrete variables

    dataset['FamilySize'] = dataset ['SibSp'] + dataset['Parch'] + 1



    dataset['IsAlone'] = 1 #initialize to yes/1 is alone

    dataset['IsAlone'].loc[dataset['FamilySize'] > 1] = 0 # now update to no/0 if family size is greater than 1



    #quick and dirty code split title from name: http://www.pythonforbeginners.com/dictionary/python-split

    dataset['Title'] = dataset['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]





    #Continuous variable bins

    #Fare Bins/Buckets using qcut or frequency bins

    dataset['FareBin'] = pd.qcut(dataset['Fare'], 4)



    #Age Bins/Buckets using cut or value bins

    dataset['AgeBin'] = pd.cut(dataset['Age'].astype(int), 5)





    

#cleanup rare title names



stat_min = 10                                                        #while small is arbitrary, we'll use the common minimum in statistics 

title_names = (train_df['Title'].value_counts() < stat_min)          #this will create a true false series with title name as index





train_df['Title'] = train_df['Title'].apply(lambda x: 'Misc' if title_names.loc[x] == True else x)

print(train_df['Title'].value_counts())

print("-"*10)





#preview data again

train_df.info()

test_df.info()

train_df.sample(10)
label = LabelEncoder()

for dataset in combine:    

    dataset['Sex_Code'] = label.fit_transform(dataset['Sex'])

    dataset['Embarked_Code'] = label.fit_transform(dataset['Embarked'])

    dataset['Title_Code'] = label.fit_transform(dataset['Title'])

    dataset['AgeBin_Code'] = label.fit_transform(dataset['AgeBin'])

    dataset['FareBin_Code'] = label.fit_transform(dataset['FareBin'])
train_df.head()
test_df.head()
print("Before", train_df.shape, test_df.shape, combine[0].shape, combine[1].shape)



train_df = train_df.drop(['Name', 'Sex', 'SibSp', 'Parch', 'Fare', 'Embarked', 'FareBin', 'AgeBin', 'Title'], axis=1)

test_df = test_df.drop(['Name', 'Sex', 'SibSp', 'Parch', 'Fare', 'Embarked', 'FareBin', 'AgeBin', 'Title'], axis=1)

combine = [train_df, test_df]



"After", train_df.shape, test_df.shape, combine[0].shape, combine[1].shape

train_df.head()
test_df.head()
X_train = train_df.drop(['Survived'],axis = 1)

y_train = train_df['Survived']



X_test = test_df.drop(['PassengerId'], axis = 1)



logmodel = LogisticRegression()

logmodel.fit(X_train, y_train)
Y_pred = logmodel.predict(X_test)

acc_log = round(logmodel.score(X_train, y_train) * 100, 2)

print(acc_log)
scores1= model_selection.cross_val_score(logmodel, X_train, train_df['Survived'], cv=5, scoring="f1")

s1= scores1.mean()

s1
svc = SVC()

svc.fit(X_train, y_train)

Y_pred = svc.predict(X_test)

acc_svc = round(svc.score(X_train, y_train) * 100, 2)

acc_svc
scores2= model_selection.cross_val_score(svc, X_train, train_df['Survived'], cv=5, scoring="f1")

s2= scores2.mean()

s2
random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(X_train, y_train)

Y_pred = random_forest.predict(X_test)

random_forest.score(X_train, y_train)

acc_random_forest = round(random_forest.score(X_train, y_train) * 100, 2)

acc_random_forest
scores3= model_selection.cross_val_score(random_forest, X_train, train_df['Survived'], cv=5, scoring="f1")

s3= scores3.mean()

s3
knn = KNeighborsClassifier(n_neighbors = 3)

knn.fit(X_train, y_train)

Y_pred = knn.predict(X_test)

acc_knn = round(knn.score(X_train, y_train) * 100, 2)

acc_knn
scores4= model_selection.cross_val_score(knn, X_train, train_df['Survived'], cv=5, scoring="f1")

s4= scores4.mean()

s4
gaussian = GaussianNB()

gaussian.fit(X_train, y_train)

Y_pred = gaussian.predict(X_test)

acc_gaussian = round(gaussian.score(X_train, y_train) * 100, 2)

acc_gaussian

scores5= model_selection.cross_val_score(gaussian, X_train, train_df['Survived'], cv=5, scoring="f1")

s5= scores5.mean()

s5
decision_tree = DecisionTreeClassifier()

decision_tree.fit(X_train, y_train)

Y_pred = decision_tree.predict(X_test)

acc_decision_tree = round(decision_tree.score(X_train, y_train) * 100, 2)

acc_decision_tree
scores6= model_selection.cross_val_score(decision_tree, X_train, train_df['Survived'], cv=5, scoring="f1")

s6= scores6.mean()

s6


classifier = XGBClassifier()

classifier.fit(X_train, y_train)

Y_pred = classifier.predict(X_test)

acc_xgb = round(classifier.score(X_train, y_train) * 100, 2)

acc_xgb

scores7= model_selection.cross_val_score(classifier, X_train, train_df['Survived'], cv=5, scoring="f1")

s7= scores7.mean()

s7
models = pd.DataFrame({

    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 

              'Random Forest', 'Naive Bayes', 'Decision Tree', 'XGBoost'],

    'Acc_score': [acc_svc, acc_knn, acc_log, 

              acc_random_forest, acc_gaussian, acc_decision_tree, acc_xgb],

    'f1_score': [s1, s2, s3, s4, s5, s6, s7]})

models.sort_values(by='Acc_score', ascending=False)
submission = pd.DataFrame({

        "PassengerId": test_df["PassengerId"],

        "Survived": classifier.predict(X_test)

    })

submission.to_csv('Submission.csv', index=False)