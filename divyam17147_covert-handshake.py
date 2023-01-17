# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

# Tabular Data
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

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

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))


# Any results you write to the current directory are saved as output.
def plot_histograms(df, variables, n_rows, n_cols):
    fig = plt.figure( figsize = ( 16 , 12 ) )
    for i, var_name in enumerate( variables ):
        ax=fig.add_subplot( n_rows , n_cols , i+1 )
        df[ var_name ].hist( bins=10 , ax=ax )
        ax.set_title( 'Skew: ' + str( round( float( df[ var_name ].skew() ) , ) ) ) # + ' ' + var_name ) #var_name+" Distribution")
        ax.set_xticklabels( [] , visible=False )
        ax.set_yticklabels( [] , visible=False )
    fig.tight_layout()  # Improves appearance a bit.
    plt.show()
def plot_distribution( df , var , target , **kwargs ):
    row = kwargs.get( 'row' , None )
    col = kwargs.get( 'col' , None )
    facet = sns.FacetGrid( df , hue=target , aspect=4 , row = row , col = col )
    facet.map( sns.kdeplot , var , shade= True )
    facet.set( xlim=( 0 , df[ var ].max() ) )
    facet.add_legend()
def plot_categories( df , cat , target , **kwargs ):
    row = kwargs.get( 'row' , None )
    col = kwargs.get( 'col' , None )
    facet = sns.FacetGrid( df , row = row , col = col )
    facet.map( sns.barplot , cat , target )
    facet.add_legend()
def plot_correlation_map( df ):
    corr = df.corr()
    _ , ax = plt.subplots( figsize =( 720 , 600 ) )
    cmap = sns.diverging_palette( 220 , 10 , as_cmap = True )
    _ = sns.heatmap(
        corr, 
        cmap = cmap,
        square=True, 
        cbar_kws={ 'shrink' : .9 }, 
        ax=ax, 
        annot = True, 
        annot_kws = { 'fontsize' : 12 }
    )
def describe_more(df):
    var = []; l = []; t = []
    for x in df:
        var.append( x )
        l.append( len( pd.value_counts( df[ x ] ) ) )
        t.append( df[ x ].dtypes )
    levels = pd.DataFrame( { 'Variable' : var , 'Levels' : l , 'Datatype' : t } )
    levels.sort_values( by = 'Levels' , inplace = True )
    return levels
def plot_variable_importance( X , y ):
    tree = DecisionTreeClassifier( random_state = 99 )
    tree.fit( X , y )
    plot_model_var_imp( tree , X , y )
def plot_model_var_imp(model, X, y):
    imp = pd.DataFrame( model.feature_importances_ , columns = ['Importance'], index = X.columns)
    imp = imp.sort_values( [ 'Importance' ] , ascending = True )
    imp[:10].plot( kind = 'barh' )
    print (model.score( X , y ))
# Get data
train_file = "../input/train.csv"
test_file = "../input/test.csv"

train_set = pd.read_csv(train_file)
test_set = pd.read_csv(test_file)
train_set.head()
train_set.describe()
# MSSubClass gives housing type for sale - Need specific houses features
print(train_set['MSSubClass'].head())
train_set.shape
train_set.corr()["SalePrice"]
train_set
# A lot of the data is NaN, so we must quantify those catergories
print(train_set.corr().shape)
print(train_set.shape)
# Assign dummy variables to check for correlation
mszone_clone = pd.get_dummies(train_set.MSZoning, prefix = "MSZoning")
mszone_clone["MSZoning_C (all)"].sum()
lot_clone = pd.DataFrame()
lot_clone["LotFrontage"] = train_set.LotFrontage.fillna(train_set.LotFrontage.mean())
lot_clone
street_clone = pd.get_dummies(train_set.Street, prefix = "Street")
print(train_set["Street"].unique())
print(street_clone.head())
"""
print(train_set["MSZoning"].unique())
mszone_clone = np.array([])
for i in train_set["MSZoning"]:
    if
"""
