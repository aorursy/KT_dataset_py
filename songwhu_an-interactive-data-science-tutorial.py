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
train.head()
# Run the code to see the variables, then read the variable description below to understand them.

titanic.head()
titanic.describe()
plot_correlation_map( titanic )
# Plot distributions of Age of passangers who survived or did not survive

plot_distribution( titanic , var = 'Age' , target = 'Survived' , row = 'Sex' )
# Excersise 1

# Plot distributions of Fare of passangers who survived or did not survive

plot_distribution(titanic, var='Fare', target='Survived', row='Sex')
# Plot survival rate by Embarked

plot_categories( titanic , cat = 'Embarked' , target = 'Survived' )
# Excersise 2

# Plot survival rate by Sex

plot_categories(titanic, cat='Sex', target='Survived')
# Excersise 3

# Plot survival rate by Pclass

plot_categories(titanic, cat='Pclass', target='Survived')
# Excersise 4

# Plot survival rate by SibSp

plot_categories(titanic, cat='SibSp', target='Survived')
# Excersise 5

# Plot survival rate by Parch

plot_categories(titanic, cat='Parch', target='Survived')
# Transform Sex into binary values 0 and 1

sex = pd.Series( np.where( full.Sex == 'male' , 1 , 0 ) , name = 'Sex' )
# Create a new variable for every unique value of Embarked

embarked = pd.get_dummies( full.Embarked , prefix='Embarked' )

embarked.head()
# Create a new variable for every unique value of Embarked

pclass = pd.get_dummies( full.Pclass , prefix='Pclass' )

pclass.head()
# Create dataset

imputed = pd.DataFrame()



# Fill missing values of Age with the average of Age (mean)

imputed[ 'Age' ] = full.Age.fillna( full.Age.mean() )



# Fill missing values of Fare with the average of Fare (mean)

imputed[ 'Fare' ] = full.Fare.fillna( full.Fare.mean() )



imputed.head()
title = pd.DataFrame()

# we extract the title from each name

title[ 'Title' ] = full[ 'Name' ].map( lambda name: name.split( ',' )[1].split( '.' )[0].strip() )



# a map of more aggregated titles

Title_Dictionary = {

                    "Capt":       "Officer",

                    "Col":        "Officer",

                    "Major":      "Officer",

                    "Jonkheer":   "Royalty",

                    "Don":        "Royalty",

                    "Sir" :       "Royalty",

                    "Dr":         "Officer",

                    "Rev":        "Officer",

                    "the Countess":"Royalty",

                    "Dona":       "Royalty",

                    "Mme":        "Mrs",

                    "Mlle":       "Miss",

                    "Ms":         "Mrs",

                    "Mr" :        "Mr",

                    "Mrs" :       "Mrs",

                    "Miss" :      "Miss",

                    "Master" :    "Master",

                    "Lady" :      "Royalty"



                    }



# we map each title

title[ 'Title' ] = title.Title.map( Title_Dictionary )

title = pd.get_dummies( title.Title )

#title = pd.concat( [ title , titles_dummies ] , axis = 1 )



title.head()
cabin = pd.DataFrame()



# replacing missing cabins with U (for Uknown)

cabin[ 'Cabin' ] = full.Cabin.fillna( 'U' )



# mapping each Cabin value with the cabin letter

cabin[ 'Cabin' ] = cabin[ 'Cabin' ].map( lambda c : c[0] )



# dummy encoding ...

cabin = pd.get_dummies( cabin['Cabin'] , prefix = 'Cabin' )



cabin.head()
# a function that extracts each prefix of the ticket, returns 'XXX' if no prefix (i.e the ticket is a digit)

def cleanTicket( ticket ):

    ticket = ticket.replace( '.' , '' )

    ticket = ticket.replace( '/' , '' )

    ticket = ticket.split()

    ticket = map( lambda t : t.strip() , ticket )

    ticket = list(filter( lambda t : not t.isdigit() , ticket ))

    if len( ticket ) > 0:

        return ticket[0]

    else: 

        return 'XXX'



ticket = pd.DataFrame()



# Extracting dummy variables from tickets:

ticket[ 'Ticket' ] = full[ 'Ticket' ].map( cleanTicket )

ticket = pd.get_dummies( ticket[ 'Ticket' ] , prefix = 'Ticket' )



ticket.shape

ticket.head()
family = pd.DataFrame()



# introducing a new feature : the size of families (including the passenger)

family[ 'FamilySize' ] = full[ 'Parch' ] + full[ 'SibSp' ] + 1



# introducing other features based on the family size

family[ 'Family_Single' ] = family[ 'FamilySize' ].map( lambda s : 1 if s == 1 else 0 )

family[ 'Family_Small' ]  = family[ 'FamilySize' ].map( lambda s : 1 if 2 <= s <= 4 else 0 )

family[ 'Family_Large' ]  = family[ 'FamilySize' ].map( lambda s : 1 if 5 <= s else 0 )



family.head()
# Select which features/variables to include in the dataset from the list below:

# imputed , embarked , pclass , sex , family , cabin , ticket



full_X = pd.concat( [ imputed , embarked , pclass, sex, family,cabin , ticket ] , axis=1 )

full_X.head()
# Create all datasets that are necessary to train, validate and test models

train_valid_X = full_X[ 0:891 ]

train_valid_y = titanic.Survived

test_X = full_X[ 891: ]

train_X , valid_X , train_y , valid_y = train_test_split( train_valid_X , train_valid_y , train_size = .7 )



print (full_X.shape , train_X.shape , valid_X.shape , train_y.shape , valid_y.shape , test_X.shape)
train_valid_X.head()
train_valid_y.head()
plot_variable_importance(train_X, train_y)
rf_model = RandomForestClassifier(n_estimators=100)
svm_model = SVC()
gb_model = GradientBoostingClassifier()
knn_model = KNeighborsClassifier(n_neighbors = 3)
gnb_model = GaussianNB()
lg_model = LogisticRegression()
rf_model.fit( train_X , train_y )

svm_model.fit( train_X , train_y )

gb_model.fit( train_X , train_y )

knn_model.fit( train_X , train_y )

gnb_model.fit( train_X , train_y )

lg_model.fit( train_X , train_y )
# Score the model

#print(lg_model.score( train_X , train_y ) , lg_model.score( valid_X , valid_y ))

#print ('rf: ', rf_model.score( train_X , train_y ) , rf_model.score( valid_X , valid_y ))

rf_train_score = rf_model.score( train_X , train_y )

rf_valid_score = rf_model.score( valid_X , valid_y )

svm_train_score = svm_model.score( train_X , train_y )

svm_valid_score = svm_model.score( valid_X , valid_y )

gb_train_score = gb_model.score( train_X , train_y )

gb_valid_score = gb_model.score( valid_X , valid_y )

knn_train_score = knn_model.score( train_X , train_y )

knn_valid_score = knn_model.score( valid_X , valid_y )

gnb_train_score = gnb_model.score( train_X , train_y )

gnb_valid_score = gnb_model.score( valid_X , valid_y )

lg_train_score = lg_model.score( train_X , train_y )

lg_valid_score = lg_model.score( valid_X , valid_y )

models = pd.DataFrame({

    'Model': ['rf','svm','gb','knn','gnb','lg'],

    'train_score':[rf_train_score, svm_train_score, gb_train_score, knn_train_score, gnb_train_score, lg_train_score],

    'valid_score':[rf_valid_score, svm_valid_score, gb_valid_score, knn_valid_score, gnb_valid_score, lg_valid_score]

})

models.sort_values(by='valid_score', ascending=False)
rf_rfecv = RFECV( estimator = rf_model , step = 1 , cv = StratifiedKFold( train_y , 2 ) , scoring = 'accuracy' )

rf_rfecv.fit( train_X , train_y )

gb_rfecv = RFECV( estimator = gb_model , step = 1 , cv = StratifiedKFold( train_y , 2 ) , scoring = 'accuracy' )

gb_rfecv.fit( train_X , train_y )

lg_rfecv = RFECV( estimator = lg_model , step = 1 , cv = StratifiedKFold( train_y , 2 ) , scoring = 'accuracy' )

lg_rfecv.fit( train_X , train_y )

rf_train_score = rf_rfecv.score( train_X , train_y )

rf_valid_score = rf_rfecv.score( valid_X , valid_y )

gb_train_score = gb_rfecv.score( train_X , train_y )

gb_valid_score = gb_rfecv.score( valid_X , valid_y )

lg_train_score = lg_rfecv.score( train_X , train_y )

lg_valid_score = lg_rfecv.score( valid_X , valid_y )

rf_optional = rf_rfecv.n_features_

gb_optional = gb_rfecv.n_features_

lg_optional = lg_rfecv.n_features_

models = pd.DataFrame({

    'Model': ['rf','gb','lg'],

    'train_score':[rf_train_score, gb_train_score, lg_train_score],

    'valid_score':[rf_valid_score, gb_valid_score, lg_valid_score],

    'optimal':[rf_optional, gb_optional, lg_optional]

})

models.sort_values(by='valid_score', ascending=False)

#print (rfecv.score( train_X , train_y ) , rfecv.score( valid_X , valid_y ))

#print( "Optimal number of features : %d" % rfecv.n_features_ )
train_valid_X_transform = rf_rfecv.transform(train_valid_X)
test_X_transform=rf_rfecv.transform(test_X)
#test_Y = model.predict( test_X )

rf_model.fit(train_valid_X_transform,train_valid_y)

test_Y = rf_model.predict(test_X_transform).astype(int)

passenger_id = full[891:].PassengerId

test = pd.DataFrame( { 'PassengerId': passenger_id , 'Survived': test_Y } )

test.shape

test.head()
test.to_csv( 'titanic_pred.csv' , index = False )