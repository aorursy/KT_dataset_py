# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
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
def plot_histograms( df , variables , n_rows , n_cols ):

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

    corr = titanic.corr()

    _ , ax = plt.subplots( figsize =( 12 , 10 ) )

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



def describe_more( df ):

    var = [] ; l = [] ; t = []

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

    

def plot_model_var_imp( model , X , y ):

    imp = pd.DataFrame( 

        model.feature_importances_  , 

        columns = [ 'Importance' ] , 

        index = X.columns 

    )

    imp = imp.sort_values( [ 'Importance' ] , ascending = True )

    imp[ : 10 ].plot( kind = 'barh' )

    print (model.score( X , y ))

    
# get titanic & test csv files as a DataFrame

train = pd.read_csv("../input/train.csv")

test    = pd.read_csv("../input/test.csv")



full = train.append( test , ignore_index = True )

titanic = full[ :891 ]



del train , test



print ('Datasets:' , 'full:' , full.shape , 'titanic:' , titanic.shape)
# Run the code to see the variables, then read the variable description below to understand them.

titanic.head()
titanic.describe()
plot_correlation_map( titanic )
# Plot distributions of Age of passangers who survived or did not survive

plot_distribution( titanic , var = 'Age' , target = 'Survived' , row = 'Sex' )
# Excersise 1

# Plot distributions of Fare of passangers who survived or did not survive

plot_distribution(titanic, var='Fare', target= 'Survived')
#So far the age distributions for female with respect to survical is between 15-40yrs old whereas 

#age distributions for male with respect to survival is between 20-40yrs old. As with respect to female not surviving 

#age distribution is between 10-25yrs old and male 15-25yrs old. Then lower fare shows higher rate of not surviving
# Plot survival rate by Embarked

plot_categories( titanic , cat = 'Embarked' , target = 'Survived' )
#Looks like port of embarkation Cherbourg lost most passengers whereas southhampton lost least passengers 
# Excersise 2

# Plot survival rate by Sex

plot_categories(titanic, cat= 'Sex', target = 'Survived')
#Wow! looks like female survived by ~70% compare to make only ~20%
# Excersise 3

# Plot survival rate by Pclass

plot_categories(titanic, cat='Pclass', target='Survived')
#The above plot shows the survival rate was higher for passengers on 1st class vs 2nd or 3rd.

#Of the 1st class passengers, ~60% survived where as ~45% in the 2nd class and only ~25% on the 3rd class
# Excersise 4

# Plot survival rate by SibSp

plot_categories(titanic, cat='SibSp', target='Survived')
#The above plot shows passengers with 1 sibling or spouse has higher rate of survival then passengers 

#with no spouse or siblings. Also, the as the number of siblings increased, the survival rate decreased
# Excersise 5

# Plot survival rate by Parch

plot_categories(titanic, cat='Parch', target='Survived')
#The above plot shows passengers with 1 to 3 children, the survival rate was 60%

#passengers with no children was lower survival ~38%. However, passengers with 5 children, the survival 

#rate is lower ~20%