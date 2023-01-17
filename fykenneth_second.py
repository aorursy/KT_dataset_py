# Ignore warnings

import warnings

warnings.filterwarnings('ignore')



# Handle table-like data and matrices

import numpy as np

import pandas as pd



# Modelling Algorithms

from sklearn.tree import DecisionTreeClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier , GradientBoostingClassifier



# Modelling Helpers

from sklearn.preprocessing import Imputer , Normalizer , scale

from sklearn.cross_validation import train_test_split , StratifiedKFold

from sklearn.feature_selection import RFECV



# Visualisation

import matplotlib as mpl

import matplotlib.pyplot as plt

import matplotlib.pylab as pylab

import seaborn as sns



# Configure visualisations

%matplotlib inline

mpl.style.use( 'ggplot' )

sns.set_style( 'white' )

pylab.rcParams[ 'figure.figsize' ] = 8 , 6
train = pd.read_csv("../input/train.csv")

test    = pd.read_csv("../input/test.csv")
train.head()
from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

model = LogisticRegression()
train.Survived
sex = pd.Series( np.where( train.Sex == 'male' , 1 , 0 ) , name = 'Sex' )

sex_2 = pd.Series( np.where( test.Sex == 'male' , 1 , 0 ) , name = 'Sex' )
# Create dataset

imputed = pd.DataFrame()

imputed2 = pd.DataFrame()

# Fill missing values of Age with the average of Age (mean)

imputed[ 'Age' ] = train.Age.fillna( train.Age.mean() )

imputed2[ 'Age' ] = test.Age.fillna( test.Age.mean() )

# Fill missing values of Fare with the average of Fare (mean)

imputed[ 'Fare' ] = train.Fare.fillna( train.Fare.mean())

imputed2[ 'Fare' ] = test.Fare.fillna( test.Fare.mean())

imputed.head()
cabin = pd.DataFrame()

cabin2 = pd.DataFrame()

# replacing missing cabins with U (for Uknown)

cabin[ 'Cabin' ] = train.Cabin.fillna( 'U' )

cabin2[ 'Cabin' ] = test.Cabin.fillna( 'U' )

# mapping each Cabin value with the cabin letter

cabin[ 'Cabin' ] = cabin[ 'Cabin' ].map( lambda c : c[0] )

cabin2[ 'Cabin' ] = cabin[ 'Cabin' ].map( lambda c : c[0] )



# dummy encoding ...

cabin = pd.get_dummies( cabin['Cabin'] , prefix = 'Cabin' )

cabin2 = pd.get_dummies( cabin2['Cabin'] , prefix = 'Cabin' )

cabin.head()
full_X = pd.concat( [ sex ] , axis=1 )

print (full_X.shape, train.Survived.shape)
train.Survived.head()
model.fit(full_X, train.Survived)
#testing area

k_range = list(range(1,31))

param_grid = dict(n_neighbors = k_range)

print (param_grid)
#grid method

from sklearn.grid_search import GridSearchCV

grid = GridSearchCV(model, param_grid, cv = 10, scoring = 'accuracy')

grid.fit(full_X,train.Survived)
grid.grid_scores_


test_x = pd.concat( [sex_2 ] , axis=1 )

test_x.head()
def plot_model_var_imp( model , X , y ):

    imp = pd.DataFrame( 

        model.feature_importances_  , 

        columns = [ 'Importance' ] , 

        index = X.columns 

    )

    imp = imp.sort_values( [ 'Importance' ] , ascending = True )

    imp[ : 10 ].plot( kind = 'barh' )

    print (model.score( X , y ))

plot_model_var_imp(model, full_X,train.Survived)
y_pred = model.predict(full_X)
len(y_pred)
from sklearn import metrics

print (metrics.accuracy_score(train.Survived,y_pred))
y_pred2 = model.predict(test_x)
y_pred2.shape
passenger_id = test[:].PassengerId

testreal = pd.DataFrame( { 'PassengerId': passenger_id , 'Survived': y_pred2 } )
testreal.shape
testreal.head()
testreal.to_csv( 'titanic_pred2.csv' , index = False )