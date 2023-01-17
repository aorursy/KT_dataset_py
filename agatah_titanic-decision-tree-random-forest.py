import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import re

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier, export_graphviz

import sklearn.tree as tree

from sklearn.metrics import roc_curve, auc, confusion_matrix

from sklearn import metrics

from subprocess import call

from IPython.display import Image



import warnings

warnings.filterwarnings('ignore')
#Reading file

titanic=pd.read_csv('../input/train.csv')
#Inspecting data

print(titanic.describe())

print(titanic.info())
#Filling nans in age column with mean age of people from the same sex and Pclass

grouped=titanic.groupby(['Sex', 'Pclass'])

def cleaning(group):

    return group.fillna(group.mean())

titanic['Age']=grouped['Age'].transform(cleaning)
#Filling nans values in embarked column with the most frequent value

print(titanic['Embarked'].value_counts())

titanic['Embarked']=titanic['Embarked'].fillna('S')
#Dropping unnecessary columns - PassengerID, ticket and cabin

titanic.drop(['PassengerId', 'Ticket', 'Cabin'], axis=1, inplace=True)
#Adding column with information about family size

titanic['FamilySize']= titanic['SibSp']+titanic['Parch']+1
#Adding column IsAlone: 0 - travelling with family, 1 - travelling alone

titanic['IsAlone']=1

titanic['IsAlone'].loc[titanic['FamilySize']>1]=0
#Extracting titles from name column and creating column Title

titanic['Title']=titanic['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)

titanic['Title'].replace({'Ms':'Miss', 'Mlle':'Miss', 'Mme':'Mrs'}, inplace=True)

titanic['Title'].replace(['Dr', 'Rev', 'Col', 'Major', 'Jonkheer', 'Countess', 'Lady', 'Don', 'Capt', 'Sir'], 'Special', inplace=True)
#Creating column with age group

titanic['GroupAge']=pd.cut(titanic['Age'], 4, labels=False)
#Creating column with fare group

titanic['GroupFare']=pd.qcut(titanic['Fare'],4, labels=False)
#Dropping unnecessary columns

titanic.drop(['FamilySize', 'Name', 'Age', 'SibSp', 'Parch', 'Fare'], axis=1, inplace=True)
#Creating dummy variables for categorical columns

titanic=pd.get_dummies(titanic, columns=['Pclass', 'Sex', 'Embarked', 'Title'], drop_first=True)
#Inspecting data

print(titanic.head())

print(titanic.describe())

print(titanic.info())
#Extract feature columns and target column

X=titanic.drop('Survived', axis=1)

y=titanic['Survived']
#split dataset to train and test sample

X_train, X_test, y_train, y_test=train_test_split(X,y, test_size=0.2, random_state=42)
#function which makes prediction on given model and return auc score

def prediction_auc(X, y_true):

    y_pred=decisiontree.predict(X)

    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_true, y_pred)

    result = auc(false_positive_rate, true_positive_rate)

    return result



#plotting results for train and test data depending on teste variable

def plotting(x, x_name, list_auc_train, list_auc_test):

    plt.plot(x, list_auc_train, label='train')    

    plt.plot(x, list_auc_test, label='test')

    plt.legend()

    plt.xlabel(x_name)

    plt.ylabel('AUC score')

    plt.show()



#function which return image of tree's structure

def visualizing_tree(tree, name):

    export_graphviz(tree, out_file='tree.dot',feature_names = X_train.columns.tolist(), class_names=['Not Survived', 'Survived'], rounded = True, proportion = False, precision = 2, filled = True)

    call(['dot', '-Tpng', 'tree.dot', '-o', name, '-Gdpi=600'])

    return Image(filename = name)
max_depths=range(1,31) #creating list of values to check



list_auc_train=[] #list to store results for train dataset 

list_auc_test=[] #list to store results for test dataset



#creating model and check result for each value of max depth

for max_depth in max_depths:

    decisiontree=DecisionTreeClassifier(max_depth=max_depth)

    decisiontree.fit(X_train,y_train)

    

    auc_train=prediction_auc(X_train,y_train)

    list_auc_train.append(auc_train)

    

    auc_test=prediction_auc(X_test,y_test)

    list_auc_test.append(auc_test)



#plotting results    

plotting(max_depths, 'max depth', list_auc_train, list_auc_test)
#buliding tree with max depth=4

decisiontree_depth_4=DecisionTreeClassifier(max_depth=4)

decisiontree_depth_4.fit(X_train,y_train)

y_pred=decisiontree_depth_4.predict(X_test)

false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)

result = auc(false_positive_rate, true_positive_rate)

print('AUC score: ', result)
#visualizing tree structure

visualizing_tree(decisiontree_depth_4, 'tree_depth_4.png')
min_samples_splits=np.linspace(0.05,0.4,8) #creating list of values to check

list_auc_train=[] #list to store results for train dataset 

list_auc_test=[] #list to store results for test dataset



#creating model and check result for each value of min samples split

for min_samples_split in min_samples_splits:

    decisiontree=DecisionTreeClassifier(min_samples_split=min_samples_split)

    decisiontree.fit(X_train,y_train)

    auc_train=prediction_auc(X_train,y_train)

    list_auc_train.append(auc_train)

    

    auc_test=prediction_auc(X_test,y_test)

    list_auc_test.append(auc_test)



#plotting results    

plotting(min_samples_splits, 'min samples split', list_auc_train, list_auc_test)
#building tree with min samples split=0.1 and visualizing its structure

decisiontree_split_01=DecisionTreeClassifier(min_samples_split=0.1)

decisiontree_split_01.fit(X_train,y_train)

y_pred=decisiontree_split_01.predict(X_test)

false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)

result = auc(false_positive_rate, true_positive_rate)

print('AUC score: ', result)

visualizing_tree(decisiontree_split_01, 'tree_split_01.png')
min_samples_leafs = np.linspace(0.1, 0.5, 5, endpoint=True) #creating list of values to check

list_auc_train=[] #list to store results for train dataset 

list_auc_test=[] #list to store results for test dataset



#creating model and check result for each value of min samples leaf

for min_samples_leaf in min_samples_leafs:

    decisiontree=DecisionTreeClassifier(min_samples_leaf=min_samples_leaf)

    decisiontree.fit(X_train,y_train)

    auc_train=prediction_auc(X_train,y_train)

    list_auc_train.append(auc_train)

    

    auc_test=prediction_auc(X_test,y_test)

    list_auc_test.append(auc_test)



#plotting results    

plotting(min_samples_leafs, 'min samples leaf', list_auc_train, list_auc_test)
#building tree with min samples leaf=0.2 and visualizing its structure

decisiontree_leaf_02=DecisionTreeClassifier(min_samples_leaf=0.2)

decisiontree_leaf_02.fit(X_train,y_train)

y_pred=decisiontree_leaf_02.predict(X_test)

false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)

result = auc(false_positive_rate, true_positive_rate)

print('AUC score: ', result)

visualizing_tree(decisiontree_leaf_02, 'tree_leaf_02.png')
#Random Forest

randomforest=RandomForestClassifier() #initializing model

ne = np.arange(1,20) 

param_grid = {'n_estimators' : ne} #parameters grid



rf_cv = GridSearchCV(randomforest, param_grid=param_grid, cv=5) #performing gridsearchcv

rf_cv.fit(X, y)

print('Best estimator', rf_cv.best_params_)
#Random Forest

randomforest=RandomForestClassifier() #initializing model

ne = np.arange(1,20) 

param_grid = {'n_estimators' : ne} #parameters grid



rf_cv = GridSearchCV(randomforest, param_grid=param_grid, cv=5) #performing gridsearchcv

rf_cv.fit(X, y)

print('Best estimator', rf_cv.best_params_)