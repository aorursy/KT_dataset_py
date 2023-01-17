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



# Evaluation

from sklearn.model_selection import cross_val_score



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

    corr = df.corr()

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

plot_distribution( titanic , var = 'Fare' , target = 'Survived' )
# Plot survival rate by Embarked

plot_categories( titanic , cat = 'Embarked' , target = 'Survived' )
# Excersise 2

# Plot survival rate by Sex

plot_categories( titanic , cat = 'Sex' , target = 'Survived' )
# Excersise 3

# Plot survival rate by Pclass

plot_categories( titanic , cat = 'Pclass' , target = 'Survived' )
# Excersise 4

# Plot survival rate by SibSp

plot_categories( titanic , cat = 'SibSp' , target = 'Survived' )
# Excersise 5

# Plot survival rate by Parch

plot_categories( titanic , cat = 'Parch' , target = 'Survived' )
# explore res
# Transform Sex into binary values 0 and 1

sex = pd.Series( np.where( full.Sex == 'male' , 1 , 0 ) , name = 'Sex' )

# Create a new variable for every unique value of Embarked

embarked = pd.get_dummies( full.Embarked , prefix='Embarked' )

embarked.head()
# Create a new variable for every unique value of Embarked

pclass = pd.get_dummies( full.Pclass , prefix='Pclass' )

pclass.head()
total = titanic.isnull().sum().sort_values(ascending=False)

percent = (titanic.isnull().sum()/titanic.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.head(20)
# Create dataset

imputed = pd.DataFrame()



# Fill missing values of Age with the average of Age (mean)

imputed[ 'Age' ] = full.Age.fillna( full.Age.median() )



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
age_bins = [0,20,30,40,50,999]

age_labels = ['20-','20-30','30-40','40-50','50+']



ranges = pd.cut(imputed['Age'], bins=age_bins, labels=age_labels, include_lowest=True)



ages = pd.get_dummies(ranges)

ages.head()
fares = pd.get_dummies(pd.qcut(imputed['Fare'], 4))

fares.head()
# Select which features/variables to include in the dataset from the list below:

# imputed , embarked , pclass , sex , family , cabin , 



# test: 0.78468 train: 0.8317

#full_X = pd.concat( [imputed, embarked , sex, pclass, family] , axis=1 ) # 



# 0.8250

#full_X = pd.concat( [embarked , sex, pclass, family, ages, fares] , axis=1 )



# 0.8384

#full_X = pd.concat( [sex, family, pclass, ages, fares] , axis=1 )



# 0.8406

full_X = pd.concat( [imputed, sex, family, pclass] , axis=1 )



#full_X = pd.concat( [imputed , sex, cabin, pclass, family ] , axis=1 )



full_X.head()

# Create all datasets that are necessary to train, validate and test models

train_valid_X = full_X[ 0:891 ]

train_valid_y = titanic.Survived

test_X = full_X[ 891: ]

train_X , valid_X , train_y , valid_y = train_test_split( train_valid_X , train_valid_y , train_size = .7 )
plot_variable_importance(train_X, train_y)
model_randomForest = RandomForestClassifier(n_estimators=100)

model_svc = SVC()

model_gradiantBoosting = GradientBoostingClassifier()

model_knn = KNeighborsClassifier(n_neighbors = 3)

model_gaussian = GaussianNB()

model_logistic = LogisticRegression()

model_dtree = DecisionTreeClassifier(criterion = 'gini', random_state = 0)
model_randomForest.fit( train_X , train_y )

model_svc.fit(train_X, train_y)

model_logistic.fit(train_X, train_y)

model_gaussian.fit(train_X, train_y)

model_knn.fit(train_X, train_y)

model_gradiantBoosting.fit(train_X, train_y)

model_dtree.fit(train_X, train_y)

from sklearn.ensemble import VotingClassifier



ensemble_voting=VotingClassifier(estimators=[('KNN',model_knn),

                                              ('RFor', model_randomForest),

                                              ('LR',model_logistic),

                                              ('DT',model_dtree),

                                              ('NB',model_gaussian)

                                             ], 

                       voting='soft').fit(train_X,train_y)
acc_randomForest = cross_val_score(estimator = model_randomForest, X = train_valid_X, y = titanic.Survived, cv = 10)

randomForest_acc_mean = acc_randomForest.mean()

randomForest_std = acc_randomForest.std()



acc_logistic = cross_val_score(estimator = model_logistic, X = train_valid_X, y = titanic.Survived, cv = 10)

logistic_acc_mean = acc_logistic.mean()

logistic_std= acc_logistic.std()



acc_gaussian = cross_val_score(estimator = model_gaussian, X = train_valid_X, y = titanic.Survived, cv = 10)

gaussian_acc_mean = acc_gaussian.mean()

gaussian_std = acc_gaussian.std()



acc_knn = cross_val_score(estimator = model_knn, X = train_valid_X, y = titanic.Survived, cv = 10)

knn_acc_mean = acc_knn.mean()

knn_std = acc_knn.std()



acc_gradiantBoosting = cross_val_score(estimator = model_gradiantBoosting, X = train_valid_X, y = titanic.Survived, cv = 10)

gradiantBoosting_acc_mean = acc_gradiantBoosting.mean()

gradiantBoosting_std = acc_gradiantBoosting.std()



acc_dtree = cross_val_score(estimator = model_dtree, X = train_valid_X, y = titanic.Survived, cv = 10)

dtree_acc_mean = acc_dtree.mean()

dtree_std = acc_dtree.std()



acc_svc = cross_val_score(estimator = model_svc, X = train_valid_X, y = titanic.Survived, cv = 10)

svc_acc_mean = acc_svc.mean()

svc_std = acc_svc.std()



acc_voting = cross_val_score(estimator = ensemble_voting, X = train_valid_X, y = titanic.Survived, cv = 10)

voting_acc_mean = acc_voting.mean()

voting_std = acc_voting.std()
x_labels = ('Accuracy','Deviation')

y_labels = ('Logistic Regression','K-Nearest Neighbors','Kernel SVM','Naive Bayes'

            ,'Decision Tree','Random Forest','XGBoost', 'Ensemble Voting')

score_array = np.array([[logistic_acc_mean, logistic_std],

                        [knn_acc_mean, knn_std],

                        [svc_acc_mean, svc_std],

                        [gaussian_acc_mean, gaussian_std],

                        [dtree_acc_mean, dtree_std],

                        [randomForest_acc_mean, randomForest_std],

                        [gradiantBoosting_acc_mean, gradiantBoosting_std],

                        [voting_acc_mean, voting_std]])  

fig = plt.figure(1)

fig.subplots_adjust(left=0.2,top=0.8, wspace=1)

ax = plt.subplot2grid((4,3), (0,0), colspan=2, rowspan=2)

score_table = ax.table(cellText=score_array,

                       rowLabels=y_labels,

                       colLabels=x_labels,

                       loc='upper center')

score_table.set_fontsize(14)

ax.axis("off") # Hide plot axis

fig.set_size_inches(w=18, h=10)

plt.show()
# Score the model

print ('KNN: ',model_knn.score( train_X , train_y ) , model_knn.score( valid_X , valid_y ))

print ('Random Forest: ', model_randomForest.score( train_X , train_y ) , model_randomForest.score( valid_X , valid_y ))

print ('Gradiant Boosting: ',model_gradiantBoosting.score( train_X , train_y ) , model_gradiantBoosting.score( valid_X , valid_y ))

print ('Logistic: ', model_logistic.score( train_X , train_y ) , model_logistic.score( valid_X , valid_y ))

print ('Gaussian', model_gaussian.score( train_X , train_y ) , model_gaussian.score( valid_X , valid_y ))

print ('Decision tree', model_dtree.score( train_X , train_y ) , model_dtree.score( valid_X , valid_y ))

print ('Ensemble - voting: ',ensemble_voting.score( train_X , train_y ) , ensemble_voting.score( valid_X , valid_y ))

model = model_gradiantBoosting
#plot_model_var_imp(model, train_X, train_y)
rfecv = RFECV( estimator = model , step = 1 , cv = StratifiedKFold( train_y , 10 ) , scoring = 'accuracy' )

rfecv.fit( train_X , train_y )



#

print (rfecv.score( train_X , train_y ) , rfecv.score( valid_X , valid_y ))

print( "Optimal number of features : %d" % rfecv.n_features_ )



# Plot number of features VS. cross-validation scores

#plt.figure()

#plt.xlabel( "Number of features selected" )

#plt.ylabel( "Cross validation score (nb of correct classifications)" )

#plt.plot( range( 1 , len( rfecv.grid_scores_ ) + 1 ) , rfecv.grid_scores_ )

#plt.show()
test_Y = model.predict( test_X ).astype(int)

passenger_id = full[891:].PassengerId

test = pd.DataFrame( { 'PassengerId': passenger_id , 'Survived': test_Y } )

test.shape

test.head()

test.to_csv( 'titanic_pred.csv' , index = False )