# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier , GradientBoostingClassifier

from sklearn import tree



# Visualisation

import matplotlib as mpl

import matplotlib.pyplot as plt

import matplotlib.pylab as pylab

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/train.csv")

test    = pd.read_csv("../input/test.csv")
train.head()
train.describe()
corr = train.corr()

_ , ax = plt.subplots( figsize =( 12 , 10 ) )

cmap = sns.diverging_palette( 220 , 10 , as_cmap = True )

_ = sns.heatmap(corr, cmap = cmap, square=True, cbar_kws={ 'shrink' : .9 }, ax=ax, annot = True, annot_kws = {'fontsize' : 12 })
#type(train['Survived'][0])

train['Name']
type(train)
target = train['Survived'].values

train = train.drop(['Survived'], axis=1)

id_test = test['PassengerId']

train_size = train.shape[0]
titanic = pd.concat((train,test), axis=0, ignore_index=True)
titanic = titanic.drop(['Name','PassengerId','Ticket'], axis=1)
titanic.Age = titanic.Age.fillna(titanic.Age.mean())

titanic.Fare = titanic.Age.fillna(titanic.Fare.mean())

titanic.Cabin = titanic.Cabin.fillna( 'U' )
titanic.Cabin = titanic.Cabin.map( lambda c : c[0] )
print ("Nb null dans Age : "+str(titanic.Age.isnull().sum()))

print ("Nb null dans Parch : "+str(titanic.Parch.isnull().sum()))

print ("Nb null dans Pclass : "+str(titanic.Pclass.isnull().sum()))

print ("Nb null dans Fare : "+str(titanic.Fare.isnull().sum()))

print ("Nb null dans Sex : "+str(titanic.Sex.isnull().sum()))

print ("Nb null dans Cabin : "+str(titanic.Cabin.isnull().sum()))
features = ['Pclass','Sex','SibSp','Parch','Cabin', 'Embarked']
for f in features:

    titanic_dummy = pd.get_dummies(titanic[f], prefix = f)

    titanic = titanic.drop([f], axis = 1)

    titanic = pd.concat((titanic, titanic_dummy), axis = 1)
titanic
vals = titanic.values

X = vals[:train_size]

y = target

X_test = vals[train_size:]
X
model = GradientBoostingClassifier()

model.fit(X,y)

y_pred = model.predict(X_test)
from IPython.display import Image



dot_data = tree.export_graphviz(model, out_file='tree.dot', 

                         filled=True, rounded=True,  

                         special_characters=True)  

test = pd.DataFrame( { 'PassengerId': id_test , 'Survived': y_pred } )
test.to_csv( 'titanic_pred.csv' , index = False )