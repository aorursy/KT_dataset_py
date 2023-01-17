import re

import os

import numpy as np

import pandas as pd

import csv

import matplotlib.pyplot as plt

import seaborn as sns

import sys

import time

import random

from subprocess import check_call

from IPython.display import Image as PImage



from sklearn.model_selection import StratifiedShuffleSplit

from sklearn.metrics import accuracy_score, log_loss, roc_curve, auc

import sklearn.metrics

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.neural_network import MLPClassifier

from sklearn.ensemble import RandomForestClassifier

#from sklearn.cross_validation import train_test_split

from sklearn import tree

%matplotlib inline



# usefull function for later use

def surviveByGroup(data, group):

	return data[['Survived',group]].groupby([group], as_index=False).agg(['mean', 'count']) .sort_values([('Survived','mean')], ascending=False)
data_train = pd.read_csv('../input/train.csv')

data_test  = pd.read_csv('../input/test.csv')

# preview the data

data_train.head()
def processName(name):

	title_dict = {'Mrs':'Mrs', 'Mr':'Mr', 'Miss':'Miss',

				  'Ms':'RareFem','Mlle':'RareFem','Countess':'RareFem','Lady':'RareFem', 'Dona' : 'RareFem',

				  'Master':'Master',

				  'Dr':'Dr',

				  'Major':'RareA', 'Col':'RareA', 'Don':'RareA', 'Sir':'RareA',

				  'Rev':'RareZ', 'Jonkheer':'RareZ',  'Capt':'RareZ' 

				}

	for title in title_dict.keys():

		if name.find(title) != -1:

			return title_dict[title]

	return 'None'



data_train['Title'] = data_train['Name'].map(processName)

data_train['Title'] = data_train['Title'].fillna('')

surviveByGroup(data_train, 'Title')
surviveByGroup(data_train, 'Sex')
fig, ax = plt.subplots(figsize=(13,5))



data_age = data_train['Age'].dropna()

sns.distplot(data_age, hist=True, rug=True, kde=False,ax=ax)

# we need to manage empty values

data_train['Age'].fillna(data_train['Age'].median(),inplace=True)
data_train['FamilySize'] = data_train['SibSp'] + data_train['Parch'] + 1.0

fig, ax = plt.subplots(figsize=(13,5))

sns.barplot(x=data_train['FamilySize'], y=data_train['Survived'], ax=ax)
fare_data = data_train[['Survived','Fare','FamilySize','Pclass']].dropna()



# compute the fare per person

fare_data['Single_Fare'] = fare_data['Fare']/fare_data['FamilySize']

# plot data

fig, (axis1,axis2) = plt.subplots(1,2,figsize=(13,5))

sns.barplot(y="Fare", x="Survived", hue="Pclass", data=fare_data, ax=axis1)

sns.barplot(y="Single_Fare", x="Survived", hue="Pclass", data=fare_data, ax=axis2)

# replace missing values by the median as first approximation for the next computations

data_train['Fare'].fillna(data_train['Fare'].median(),inplace = True)
fig, ax = plt.subplots(1,figsize=(13,5))

sns.barplot(y="Fare", x="FamilySize", hue="Pclass", data=fare_data,ax=ax)
sns.countplot(hue="Survived", x='Pclass', data=data_train)

surviveByGroup(data_train, 'Pclass')
sns.barplot(y="Survived", x='Embarked', hue='Sex', data=data_train)
def processCabin(str_t):

	if str_t == None:

		return ''

	m = re.match(r'([a-zA-Z])([0-9]+)$', str_t)

	if m:

		return str(m.group(1))

	return ''

data_train['Cabin'] = data_train['Cabin'].fillna('')

data_train['Deck'] = data_train['Cabin'].map(processCabin)

data_train['HasCabin'] = (data_train.Deck != '')

sns.countplot(hue="Survived", x='HasCabin', data=data_train)

sns.factorplot("Survived", col="Deck", col_wrap=8, data=data_train[data_train.Deck != ''], kind="count", size=1.7, aspect=.9)
def processTicket(ticket):

	if isinstance(ticket, int):

		return ticket

	ticket = ticket.replace(".", "").replace("/", "")

	m = re.match(r'([0-9]+)$', ticket)

	if m:

		return str(int(m.group(1))//1000)

	m2 = re.match(r'([a-zA-Z]+) *([0-9]+)$', ticket)

	if m2:

		result =  m2.group(1)+' '+str(int(m2.group(2))//1000)

		return result.lower()

	return ''

data_train['GTicket'] = data_train['Ticket'].map(processTicket)

d2 = surviveByGroup(data_train, 'GTicket')

d2[d2[('Survived','count')] > 10]
# we reverse the class just to have a positive correlation to obtain : higher number => higher survival

data_train['ISex'] = data_train['Sex'].map( {'female': 1, 'male': 0} ).astype(int)

data_train['ReversePclass'] = data_train['Pclass'].map( {1:2,2:1,3:0} ).astype(int)

data_train['GoodEmbark'] = data_train['Embarked'].map( {'C':2,'S':1,'Q':0,None:1} ).astype(int)

# compute the correlation

corr = data_train[['Survived','ISex', 'ReversePclass', "HasCabin", 'Fare', 'GoodEmbark']].corr()

fig , ax = plt.subplots( figsize =( 12 , 10 ) )

_ = sns.heatmap( corr,  cmap = 'OrRd',

    square=True,  cbar_kws={ 'shrink' : .9 }, 

    ax=ax, annot = True,  annot_kws = { 'fontsize' : 14 }

)

plt.show()
data_train['HasCabin'] = data_train['HasCabin'].map( {None:0, False:0, True:1} ).astype(int)

title_mapping = {'':0, 'None':0, 'Mrs':1, 'Miss':2, 'Mr':3,  'Dr':4, 'Master':5, 'RareA':6, 'RareZ':7, 'RareFem':8, }

data_train['CTitle'] = data_train['Title'].map(title_mapping).astype(int)



data_clean = data_train[['ISex', 'ReversePclass', "HasCabin", 'Fare', 'GoodEmbark', 'FamilySize', 'CTitle', 'Age']]



data_clean.head(5)
X_train = data_clean

Y_train = data_train["Survived"]

solver = RandomForestClassifier(criterion='gini', min_samples_split=20, min_samples_leaf=10, max_depth=4, 

                                n_estimators=1000,n_jobs=-1,max_features='auto')

solver.fit(X_train, Y_train);
pd.concat((pd.DataFrame(X_train.iloc[:, :].columns, columns = ['variable']), 

            pd.DataFrame(solver.feature_importances_, columns = ['importance'])), 

            axis = 1).sort_values(by='importance', ascending = False)[:]
accuracy = round(solver.score(X_train, Y_train) * 100, 2)

print(str(accuracy)+" %")
with open('tree0.dot', 'w') as my_file:

	example_tree = solver.estimators_[0]

	my_file = tree.export_graphviz(example_tree, out_file = my_file, feature_names=list(X_train),

                                   filled=True, rounded = True, class_names=['die','survive']  )

check_call(['dot','-Tpng','tree0.dot','-Gdpi=49','-o','tree0.png'])

PImage("tree0.png")