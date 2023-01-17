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



import matplotlib.pyplot as plt



# Modelling Helpers

from sklearn.preprocessing import Imputer , Normalizer , scale

from sklearn.cross_validation import train_test_split , StratifiedKFold

from sklearn.feature_selection import RFECV



# Visualisation

import matplotlib as mpl

import matplotlib.pyplot as plt

import matplotlib.pylab as pylab

import seaborn as sns
# load data 

col_names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']

mydf = pd.read_csv("../input/pima-indians-diabetes.csv", names=col_names)
#Dimensions of Your Data

#shape

print (mydf.shape)
# peek

print(mydf.head(10))
# data types

print(mydf.dtypes)
#description

print(mydf.describe())
#class distribution

class_counts = mydf.groupby('class').size()

print(class_counts)
# Pearson correlations

pd.set_option('display.width', 100)

pd.set_option('precision', 3) 

correlations = mydf.corr(method='pearson') 

print(correlations)
#histograms

mydf.hist()

plt.show()
#density

mydf.plot(kind='density', subplots=True, layout=(3,3), sharex=False) 

plt.show()
#box and whisker plot

mydf.plot(kind='box', subplots=True, layout=(3,3), sharex=False, sharey=False) 

plt.show()
768*0.7
x,y = np.hsplit(mydf,[8])
x.head()
y.head()
train_valid_X = x[ 0:537 ]

train_valid_y = y

test_X = x[ 537: ]

train_X , valid_X , train_y , valid_y = train_test_split( x , y , train_size = .7 )



print (x.shape , train_X.shape , valid_X.shape , train_y.shape , valid_y.shape , test_X.shape)
model = RandomForestClassifier(n_estimators=100)

# Score the model

score = model.fit( train_X , train_y )

print (model.score( train_X , train_y ) , model.score( valid_X , valid_y ))
model = SVC()

# Score the model

score = model.fit( train_X , train_y )

print (model.score( train_X , train_y ) , model.score( valid_X , valid_y ))
model = GradientBoostingClassifier()

# Score the model

score = model.fit( train_X , train_y )

print (model.score( train_X , train_y ) , model.score( valid_X , valid_y ))
model = KNeighborsClassifier(n_neighbors = 3)

# Score the model

score = model.fit( train_X , train_y )

print (model.score( train_X , train_y ) , model.score( valid_X , valid_y ))
model = GaussianNB()

# Score the model

score = model.fit( train_X , train_y )

print (model.score( train_X , train_y ) , model.score( valid_X , valid_y ))
model = LogisticRegression()

# Score the model

score = model.fit( train_X , train_y )

print (model.score( train_X , train_y ) , model.score( valid_X , valid_y ))
model = DecisionTreeClassifier()

# Score the model

score = model.fit( train_X , train_y )

print (model.score( train_X , train_y ) , model.score( valid_X , valid_y ))