# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

# RegEx
import re as re

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
    corr = train_set.corr()
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
train = pd.read_csv('../input/train.csv', header = 0, dtype={'Age': np.float64})
test  = pd.read_csv('../input/test.csv' , header = 0, dtype={'Age': np.float64})

full = train.append (test , ignore_index = True )

# remove what we don't need
# del train , test
print ('Datasets:' , 'full:' , full.shape , 'train:', train.shape, 'test:', test.shape)
print (full.info())

# Run the code to see the variables, then read the variable description below to understand them.
full.head()
train.describe()
# Plot distributions of Age of passangers who survived or did not survive
plot_distribution( train , var = 'Age' , target = 'Survived' , row = 'Sex' )
plot_distribution( train , var = 'Fare' , target = 'Survived' )
# Plot survival rate by Embarked
plot_categories( train , cat = 'Embarked' , target = 'Survived' )
plot_categories( train , cat = 'Sex' , target = 'Survived' )
plot_categories( train , cat = 'Pclass' , target = 'Survived' )
plot_categories( train , cat = 'SibSp' , target = 'Survived' )
plot_categories( train , cat = 'Parch' , target = 'Survived' )
# create Family Size and IsAlone
newFeatures = pd.DataFrame()
newFeatures[ 'FamilySize' ] = full[ 'Parch' ] + full[ 'SibSp' ] + 1
newFeatures[ 'IsAlone' ] = newFeatures[ 'FamilySize' ].map( lambda s : 1 if s == 1 else 0 )
# create Title
def get_title(name):
    title_search = re.search(' ([A-Za-z]+)\.', name)
    # If the title exists, extract and return it.
    if title_search:
        return title_search.group(1)
    return ""
newFeatures ['Title'] = full['Name'].apply(get_title)
newFeatures.head()

# normalize some of the titles, removing some of the rare ones
newFeatures['Title'] = newFeatures['Title'].replace(['Lady', 'Countess','Capt', 'Col', \
                                                     'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

newFeatures['Title'] = newFeatures['Title'].replace('Mlle', 'Miss')
newFeatures['Title'] = newFeatures['Title'].replace('Ms', 'Miss')
newFeatures['Title'] = newFeatures['Title'].replace('Mme', 'Mrs')
# impute Embark and Fare
imputed = pd.DataFrame()
# Fill in missing Embark values, using the most common data, 'S'
imputed['Embarked'] = full.Embarked.fillna('S')
# Fill in missing Fare values, using the median
imputed['Fare'] = full.Fare.fillna( full.Fare.median() )
# Fill in missing Age values, using a random value between (mean - std) and (mean + std)
age_avg = full['Age'].mean()
age_std = full['Age'].std()
age_null_count = full['Age'].isnull().sum()
imputed['Age'] = full.Age.fillna(np.random.randint(age_avg - age_std, age_avg + age_std))
#display dfs
imputed.head()

# group Fare and Age
full ['CategoricalFare'] = pd.qcut(full['Fare'], 4, labels=[0,1,2,3]).astype(int)
full ['CategoricalAge'] = pd.cut(full['Age'], 5, labels=[0,1,2,3,4]).astype(int)
full.head()

# Mapping Sex
full['Sex'] = full['Sex'].map( {'female': 0, 'male': 1} )
full.head()
#Map Title
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
newFeatures['Title'] = newFeatures['Title'].map(title_mapping)
newFeatures['Title'] = newFeatures['Title'].fillna(0)
newFeatures.head()
#Map Embarked
imputed['Embarked'] = imputed['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} )
imputed.head()
# drop the features we don't need
drop_elements = ['Name', 'Ticket', 'Cabin', 'SibSp',\
                 'Parch', 'Embarked', 'Fare', 'Age']
full = full.drop(drop_elements, axis = 1)
full.head()
# select features we want
full_X = pd.concat([full, newFeatures, imputed] , axis=1 )
full_X.head()
# Create all datasets that are necessary to train, validate and test models
train_valid_X = full_X[ 0:891 ].drop('Survived', axis = 1)
train_valid_y = train.Survived
test_X = full_X[ 891: ]
train_X , valid_X , train_y , valid_y = train_test_split( train_valid_X , train_valid_y , train_size = .7 )

print (full_X.shape , train_X.shape , valid_X.shape , train_y.shape , valid_y.shape , test_X.shape)
plot_variable_importance(train_X, train_y)
#select a model
model = RandomForestClassifier(n_estimators=100)
# train the model
model.fit( train_X , train_y )
# Score the Model
models = np.array (['Random Forest',model.score( train_X , train_y ) , model.score( valid_X , valid_y )])
print (models)
#Decision Tree Models can actually select optimal features in the model
plot_model_var_imp(model, train_X, train_y)
# automatically select the optimal number of features and visualize this.
rfecv = RFECV( estimator = model , step = 1 , cv = StratifiedKFold( train_y , 2 ) , scoring = 'accuracy' )
rfecv.fit( train_X , train_y )

print (rfecv.score( train_X , train_y ) , rfecv.score( valid_X , valid_y ))
print( "Optimal number of features : %d" % rfecv.n_features_ )

# Plot number of features VS. cross-validation scores
plt.figure()
plt.xlabel( "Number of features selected" )
plt.ylabel( "Cross validation score (nb of correct classifications)" )
plt.plot( range( 1 , len( rfecv.grid_scores_ ) + 1 ) , rfecv.grid_scores_ )
plt.show()
#select a model
model = SVC()
# train the model
model.fit( train_X , train_y )
# Score the Model
models = np.vstack([models, np.array(['SVM' , model.score( train_X , train_y ) , model.score( valid_X , valid_y )])])
print (models)
#select a model
model = GradientBoostingClassifier()
# train the model
model.fit( train_X , train_y )
# Score the Model
models = np.vstack([models, np.array(['Gradient Boosting' , model.score( train_X , train_y ) , model.score( valid_X , valid_y )])])
print (models)
#select a model
model = KNeighborsClassifier(n_neighbors = 3)
# train the model
model.fit( train_X , train_y )
# Score the Model
models = np.vstack([models, np.array(['KNN' , model.score( train_X , train_y ) , model.score( valid_X , valid_y )])])
print (models)
#select a model
model = GaussianNB()
# train the model
model.fit( train_X , train_y )
# Score the Model
models = np.vstack([models, np.array(['GaussianNB' , model.score( train_X , train_y ) , model.score( valid_X , valid_y )])])
print (models)
#select a model
model = LogisticRegression()
# train the model
model.fit( train_X , train_y )
# Score the Model
models = np.vstack([models, np.array(['LogisticRegression' , model.score( train_X , train_y ) , model.score( valid_X , valid_y )])])
print (models)
test_X = test_X.drop('Survived', axis=1)
test_X.head()
# run the best model, Random Forest - scored 62.6%
model = RandomForestClassifier(n_estimators=100)
# run the best model, Gradient Boosting - scored 62.6, 73.205
#model = GradientBoostingClassifier()
# train the model
model.fit( train_X , train_y )

# run the model on the test data set
test_Y = model.predict( test_X )
passenger_id = full[891:].PassengerId
test = pd.DataFrame( { 'PassengerId': passenger_id , 'Survived': test_Y } )
test.shape
test.head()
test.to_csv( 'titanic_pred.csv' , index = False )