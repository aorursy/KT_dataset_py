# Handle table-like data and matrices
import numpy as np
import pandas as pd

# Modelling Algorithms
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier , GradientBoostingClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn import ensemble, linear_model, neighbors, svm, tree, neural_network
from sklearn import svm,model_selection, naive_bayes,tree, ensemble, discriminant_analysis, gaussian_process



# Modelling Helpers
from sklearn.preprocessing import Imputer , Normalizer , scale ,PolynomialFeatures
from sklearn.cross_validation import train_test_split , cross_val_score, StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error,confusion_matrix, precision_score, recall_score, auc,roc_curve


# Visualisation
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab

# Configure visualisations
%matplotlib inline
mpl.style.use( 'ggplot' )
sns.set_style( 'white' )
pylab.rcParams[ 'figure.figsize' ] = 8 , 6

# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

# Read csv input files 
train    = pd.read_csv("../input/train.csv")
test     = pd.read_csv("../input/test.csv")
All = train.append( test , ignore_index = True )
titanic_ds = All[ :891 ]
del train , test
print ('Datasets:' , 'All:' , All.shape , 'titanic_ds:' , titanic_ds.shape)

# Visualization 
titanic_ds.head()
#Variable Description
#Survived: Survived 1 or not Survived 0
#Pclass: Passenger's class
#Name: Passenger's name
#Sex: Passenger's sex
#Age: Passenger's age
#SibSp: Number of siblings/spouses aboard
#Parch: Number of parents/children aboard
#Ticket: Ticket number
#Fare: Fare
#Cabin: Cabin
#Embarked: Port of embarkation

#look at some key information about the variables
titanic_ds.describe()
f,ax = plt.subplots(figsize=(12, 12))
sns.heatmap(titanic_ds.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
# Let's further explore the relationship between the features and survival of passengers.. We start by looking at the relationship between age and survival.

facet = sns.FacetGrid( titanic_ds , hue='Survived' , aspect=4 , row = 'Sex')
facet.map( sns.kdeplot , 'Age' , shade= True )
facet.set( xlim=( 0 , titanic_ds[ 'Age' ].max() ) )
facet.add_legend()
facet = sns.FacetGrid( titanic_ds)# , row = row , col = col )
facet.map( sns.barplot , 'Embarked' , 'Survived'  )
facet.add_legend()
facet = sns.FacetGrid( titanic_ds)# , row = row , col = col )
facet.map( sns.barplot , 'Sex' , 'Survived'  )
facet.add_legend()
facet = sns.FacetGrid( titanic_ds)# , row = row , col = col )
facet.map( sns.barplot , 'Pclass' , 'Survived'  )
facet.add_legend()
facet = sns.FacetGrid( titanic_ds)# , row = row , col = col )
facet.map( sns.barplot , 'SibSp' , 'Survived'  )
facet.add_legend()
facet = sns.FacetGrid( titanic_ds)# , row = row , col = col )
facet.map( sns.barplot , 'Parch' , 'Survived'  )
facet.add_legend()
#Data Preparation

# Transform Sex into binary values 0 and 1
sex = pd.Series( np.where( All.Sex == 'male' , 1 , 0 ) , name = 'Sex' )
# Create a new variable for every unique value of Embarked
embarked = pd.get_dummies( All.Embarked , prefix='Embarked' )
embarked.head()
# Create a new variable for every unique value of Embarked
pclass = pd.get_dummies( All.Pclass , prefix='Pclass' )
pclass.head()
#Fill missing values in variables

# Create dataset
imputed = pd.DataFrame()

# Fill missing values of Age with the average of Age (mean)
imputed[ 'Age' ] = All.Age.fillna( All.Age.mean() )

# Fill missing values of Fare with the average of Fare (mean)
imputed[ 'Fare' ] = All.Fare.fillna( All.Fare.mean() )

imputed.head()
#Feature Engineering – Creating new variables
#Credit: http://ahmedbesbes.com/how-to-score-08134-in-titanic-kaggle-challenge.html

#Extract titles from passenger names

title = pd.DataFrame()
# we extract the title from each name
title[ 'Title' ] = All[ 'Name' ].map( lambda name: name.split( ',' )[1].split( '.' )[0].strip() )

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

#Extract Cabin category information from the Cabin number

cabin = pd.DataFrame()

# replacing missing cabins with U (for Uknown)
cabin[ 'Cabin' ] = All.Cabin.fillna( 'U' )

# mapping each Cabin value with the cabin letter
cabin[ 'Cabin' ] = cabin[ 'Cabin' ].map( lambda c : c[0] )

# dummy encoding ...
cabin = pd.get_dummies( cabin['Cabin'] , prefix = 'Cabin' )

cabin.head()
#Extract ticket class from ticket number

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
ticket[ 'Ticket' ] = All[ 'Ticket' ].map( cleanTicket )
ticket = pd.get_dummies( ticket[ 'Ticket' ] , prefix = 'Ticket' )

ticket.shape
ticket.head()
#Create family size and category for family size

family = pd.DataFrame()

# introducing a new feature : the size of families (including the passenger)
family[ 'FamilySize' ] = All[ 'Parch' ] + All[ 'SibSp' ] + 1

# introducing other features based on the family size
family[ 'isAlone' ] = family[ 'FamilySize' ].map( lambda s : 1 if s == 1 else 0 )
family[ 'Smallfamily' ]  = family[ 'FamilySize' ].map( lambda s : 1 if 2 <= s <= 4 else 0 )
family[ 'Largefamily' ]  = family[ 'FamilySize' ].map( lambda s : 1 if 5 <= s else 0 )

family.head()
#Assemble final datasets for modelling

# Select which features/variables to include in the dataset from the list below:
# imputed , embarked , pclass , sex , family , cabin , ticket

All_X = pd.concat( [ imputed , embarked , cabin , sex ] , axis=1 )
All_X.head()
#Create datasets
# Create all datasets that are necessary to train, validate and test models
train_val_X = All_X[ 0:891 ]
train_val_y = titanic_ds.Survived
test_X = All_X[ 891: ]
train_X , val_X , train_y , val_y = train_test_split( train_val_X , train_val_y , train_size = .75 )

print (All_X.shape , train_X.shape , val_X.shape , train_y.shape , val_y.shape , test_X.shape)
#Feature importance

tree = DecisionTreeClassifier( random_state = 99 )
tree.fit( train_X , train_y )
imp = pd.DataFrame( 
    tree.feature_importances_  , 
    columns = [ 'Importance' ] , 
    index = train_X.columns 
)
imp = imp.sort_values( [ 'Importance' ] , ascending = True )
imp[ : 10 ].plot( kind = 'barh' )
print (tree.score( train_X , train_y ))
# Modeling
MLA = [
    #Ensemble Methods
    ensemble.AdaBoostClassifier(),
    ensemble.BaggingClassifier(),
    ensemble.ExtraTreesClassifier(),
    ensemble.GradientBoostingClassifier(),
    ensemble.RandomForestClassifier(),

    #Gaussian Processes
    gaussian_process.GaussianProcessClassifier(),
    
    #GLM
    linear_model.LogisticRegressionCV(),
    linear_model.PassiveAggressiveClassifier(),
    linear_model. RidgeClassifierCV(),
    linear_model.SGDClassifier(),
    linear_model.Perceptron(),
    
    #Navies Bayes
    naive_bayes.BernoulliNB(),
    naive_bayes.GaussianNB(),
    
    #Nearest Neighbor
    neighbors.KNeighborsClassifier(),
    
    #SVM
    svm.SVC(probability=True),
    svm.NuSVC(probability=True),
    svm.LinearSVC()
    
    #Trees    
    #tree.DecisionTreeClassifier(),
    #tree.ExtraTreeClassifier()
    
    ]


#train_X , val_X , train_y , val_y
#x_train, x_test, y_train, y_test


MLA_columns = []
MLA_compare = pd.DataFrame(columns = MLA_columns)


index = 0
for alg in MLA:
    
    
    predicted = alg.fit(train_X, train_y).predict(val_X)
    fp, tp, th = roc_curve(val_y, predicted)
    MLA_name = alg.__class__.__name__
    MLA_compare.loc[index,'MLA Name'] = MLA_name
    MLA_compare.loc[index, 'MLA Train Accuracy'] = round(alg.score(train_X, train_y), 4)
    MLA_compare.loc[index, 'MLA Test Accuracy'] = round(alg.score(val_X, val_y), 4)
    MLA_compare.loc[index, 'MLA Precission'] = precision_score(val_y, predicted)
    MLA_compare.loc[index, 'MLA Recall'] = recall_score(val_y, predicted)
    MLA_compare.loc[index, 'MLA AUC'] = auc(fp, tp)
    index+=1
    
MLA_compare.sort_values(by = ['MLA Test Accuracy'], ascending = False, inplace = True)    
MLA_compare
plt.subplots(figsize=(15,6))
sns.barplot(x="MLA Name", y="MLA Test Accuracy",data=MLA_compare)
plt.xticks(rotation=90)
plt.title('MLA Test Accuracy Comparison')
plt.show()

# We can plot all these figures for more clear visualizatıon

#MLA Train Accuracy Comparison
#MLA Test Accuracy Comparison
#MLA Precission Comparison
#MLA Recall Comparison
#MLA AUC Comparison
#train_X , val_X , train_y , val_y
#x_train, x_test, y_train, y_test

index = 1
for alg in MLA:  
    
    predicted = alg.fit(train_X, train_y).predict(val_X)
    fp, tp, th = roc_curve(val_y, predicted)
    roc_auc_mla = auc(fp, tp)
    MLA_name = alg.__class__.__name__
    plt.plot(fp, tp, lw=2, alpha=0.3, label='ROC %s (AUC = %0.2f)'  % (MLA_name, roc_auc_mla))
   
    index+=1

plt.title('Comparison - ROC Curve ')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.plot([0,1],[0,1],'r--')
plt.xlim([0,1])
plt.ylim([0,1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')    
plt.show()

index = 1
for alg in MLA:     
    alg.fit(train_X, train_y)    
    print('Alg name: ', alg.__class__.__name__)
    print('BEFORE tuning Parameters: ', alg.get_params())
    print("BEFORE tuning Training w/bin set score: {:.2f}". format(alg.score(train_X, train_y))) 
    print("BEFORE tuning Test w/bin set score: {:.2f}". format(alg.score(val_X, val_y)))
    print('-'*10)
    index+=1
#tune parameters
param_grid = {#'bootstrap': [True, False],
              'class_weight': ['balanced' , None],
              #'max_depth': [1, 2,3,4, None],
              #'max_features': ['log2', 'auto'],
              #'max_leaf_nodes': [0,1,2,3,4, None],
              #'min_impurity_decrease': [True, False, None],
              #'min_impurity_split': [True, False],
              #'min_samples_leaf': [1, 2,3,4,5],
              #'min_samples_split': [1,2,3,4,5],
              #'min_weight_fraction_leaf': [0.0,1.0,2.0,3.0,4.0,5.0], 
              #'n_estimators': [10,15,25,35,45], 
              'n_jobs':  [1,2,3,4,5], 
              #'oob_score': [True, False], 
              'random_state': [0,1, 2,3,4, None], 
              #'verbose': [0,1, 2,3,4, 5], 
              'warm_start': [True, False]
             }
# So, what this GridSearchCV function do is finding the best combination of parameters value that is set above.

tune_model = model_selection.GridSearchCV(linear_model.PassiveAggressiveClassifier(), param_grid=param_grid, scoring = 'roc_auc') #linear_model.PassiveAggressiveClassifier()
tune_model.fit (train_X, train_y)

print('Tuning parameters for the Algorithm: PassiveAggressiveClassifier' )
print('AFTER tuning Parameters: ', tune_model.best_params_)
print("AFTER tuning Training w/bin set score: {:.2f}". format(tune_model.score(train_X, train_y))) 
print("AFTER tuning Test w/bin set score: {:.2f}". format(tune_model.score(val_X, val_y)))
print('-'*20)  

param_grid2 = {'max_depth' : [None, 10,20],
              'max_features' : ['auto',None],
              'n_estimators' :[100,200,300],
              'random_state': [7]}

clf = RandomForestClassifier(n_estimators =100)
tune_model = model_selection.GridSearchCV(clf, param_grid=param_grid2, scoring = 'roc_auc') #linear_model.PassiveAggressiveClassifier()
tune_model.fit (train_X, train_y)

print('Tuning parameters for the Algorithm: RandomForestClassifier' )
print('AFTER tuning Parameters: ', tune_model.best_params_)
print("AFTER tuning Training w/bin set score: {:.2f}". format(tune_model.score(train_X, train_y))) 
print("AFTER tuning Test w/bin set score: {:.2f}". format(tune_model.score(val_X, val_y)))
print('-'*20)   


######################################
# Automagic to fınd the optımal number of features for some algorithm
    
rfecv = RFECV( estimator = RandomForestClassifier(n_estimators =100) , step = 1 , cv = StratifiedKFold( train_y , 2 ) , scoring = 'accuracy' )
rfecv.fit( train_X , train_y )

print (rfecv.score( train_X , train_y ) , rfecv.score( val_X , val_y ))
print( "Optimal number of features : %d" % rfecv.n_features_ )

#Plot number of features VS. cross-validation scores
plt.figure()
plt.xlabel( "Number of features selected" )
plt.ylabel( "Cross validation score (nb of correct classifications)" )
plt.plot( range( 1 , len( rfecv.grid_scores_ ) + 1 ) , rfecv.grid_scores_ )
plt.show()


clf = RandomForestClassifier(n_estimators =100)
##Train the selected model
clf.fit( train_X , train_y )

#Evaluation
# Score the model
print (clf.score( train_X , train_y ) , clf.score( val_X , val_y ))
#Deployment

test_Y = clf.predict( test_X )
passenger_id = All[891:].PassengerId
test = pd.DataFrame( { 'PassengerId': passenger_id , 'Survived': test_Y } )
test.shape
test.head()
test.to_csv( 'titanic_ds_prediction.csv' , index = False )

#