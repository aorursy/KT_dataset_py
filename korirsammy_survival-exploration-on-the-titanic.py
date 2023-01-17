# remove warnings

import warnings

warnings.filterwarnings('ignore')



#Import Important Libraries

import numpy as np # linear algebra



import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

pd.options.display.max_columns = 100



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

pd.options.display.max_rows = 100

# Input data files are available in the "../input/" directory.

# The script below lists all  the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
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




titanic.head()
titanic.describe()
sum(pd.isnull(titanic['Age']))
titanic['Age'].fillna(titanic['Age'].median(), inplace=True)
plot_correlation_map( titanic )


# Plot distributions of Age of passangers who survived or did not survive

plot_distribution( titanic , var = 'Age' , target = 'Survived' , row = 'Sex' )
survived_sex = titanic[titanic['Survived']==1]['Sex'].value_counts()

dead_sex = titanic[titanic['Survived']==0]['Sex'].value_counts()

df = pd.DataFrame([survived_sex,dead_sex])

df.index = ['Survived','Dead']

df.plot(kind='bar',stacked=True, figsize=(15,8))
figure = plt.figure(figsize=(15,8))

plt.hist([titanic[titanic['Survived']==1]['Age'],titanic[titanic['Survived']==0]['Age']], stacked=True, color = ['g','r'],

         bins = 30,label = ['Survived','Dead'])

plt.xlabel('Age')

plt.ylabel('Number of passengers')

plt.legend()
figure = plt.figure(figsize=(15,8))

plt.hist([titanic[titanic['Survived']==1]['Fare'],titanic[titanic['Survived']==0]['Fare']], stacked=True, color = ['g','r'],

         bins = 30,label = ['Survived','Dead'])

plt.xlabel('Fare')

plt.ylabel('Number of passengers')

plt.legend()
plt.figure(figsize=(15,8))

ax = plt.subplot()

ax.scatter(titanic[titanic['Survived']==1]['Age'],titanic[titanic['Survived']==1]['Fare'],c='green',s=40)

ax.scatter(titanic[titanic['Survived']==0]['Age'],titanic[titanic['Survived']==0]['Fare'],c='red',s=40)

ax.set_xlabel('Age')

ax.set_ylabel('Fare')

ax.legend(('survived','dead'),scatterpoints=1,loc='upper right',fontsize=15,)
ax = plt.subplot()

ax.set_ylabel('Average fare')

titanic.groupby('Pclass').mean()['Fare'].plot(kind='bar',figsize=(15,8), ax = ax)
survived_embark = titanic[titanic['Survived']==1]['Embarked'].value_counts()

dead_embark = titanic[titanic['Survived']==0]['Embarked'].value_counts()

df = pd.DataFrame([survived_embark,dead_embark])

df.index = ['Survived','Dead']

df.plot(kind='bar',stacked=True, figsize=(15,8))
# Plot distributions of Fare of passangers who survived or did not survive

plot_categories( titanic , cat = 'Fare' , target = 'Survived' )
# Plot survival rate by Embarked

plot_categories( titanic , cat = 'Embarked' , target = 'Survived' )
# Plot survival rate by Sex

plot_categories( titanic , cat = 'Sex' , target = 'Survived' )
# Plot survival rate by Pclass

plot_categories( titanic , cat = 'Pclass' , target = 'Survived' )


# Plot survival rate by SibSp

plot_categories( titanic , cat = 'SibSp' , target = 'Survived' )
# Plot survival rate by Parch

plot_categories( titanic , cat = 'Parch' , target = 'Survived' )
# Define a function that check if a feature has been processed or not

def status(feature):

    print ('Processing',feature,': ok')
def get_combined_data():

    # reading train data

    train = pd.read_csv('../input/train.csv')

    

    # reading test data

    test = pd.read_csv('../input/test.csv')



    # extracting and then removing the targets from the training data 

    targets = train.Survived

    train.drop('Survived',1,inplace=True)

    



    # merging train data and test data for future feature engineering

    combined = train.append(test)

    combined.reset_index(inplace=True)

    combined.drop('index',inplace=True,axis=1)

    

    return combined

combined = get_combined_data()
combined.shape


# Function to parse and exract titles from passanger names

def get_titles():



    global combined

    

    # we extract the title from each name

    combined['Title'] = combined['Name'].map(lambda name:name.split(',')[1].split('.')[0].strip())

    

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

    combined['Title'] = combined.Title.map(Title_Dictionary)
# Check the new titles feature

get_titles()

combined.head()
grouped = combined.groupby(['Sex','Pclass','Title'])

grouped.median()
def process_age():

    

    global combined

    

    # a function that fills the missing values of the Age variable

    

    def fillAges(row):

        if row['Sex']=='female' and row['Pclass'] == 1:

            if row['Title'] == 'Miss':

                return 30

            elif row['Title'] == 'Mrs':

                return 45

            elif row['Title'] == 'Officer':

                return 49

            elif row['Title'] == 'Royalty':

                return 39



        elif row['Sex']=='female' and row['Pclass'] == 2:

            if row['Title'] == 'Miss':

                return 20

            elif row['Title'] == 'Mrs':

                return 30



        elif row['Sex']=='female' and row['Pclass'] == 3:

            if row['Title'] == 'Miss':

                return 18

            elif row['Title'] == 'Mrs':

                return 31



        elif row['Sex']=='male' and row['Pclass'] == 1:

            if row['Title'] == 'Master':

                return 6

            elif row['Title'] == 'Mr':

                return 41.5

            elif row['Title'] == 'Officer':

                return 52

            elif row['Title'] == 'Royalty':

                return 40



        elif row['Sex']=='male' and row['Pclass'] == 2:

            if row['Title'] == 'Master':

                return 2

            elif row['Title'] == 'Mr':

                return 30

            elif row['Title'] == 'Officer':

                return 41.5



        elif row['Sex']=='male' and row['Pclass'] == 3:

            if row['Title'] == 'Master':

                return 6

            elif row['Title'] == 'Mr':

                return 26

    

    combined.Age = combined.apply(lambda r : fillAges(r) if np.isnan(r['Age']) else r['Age'], axis=1)

    

    status('age')
process_age()
combined.info()
def process_names():

    

    global combined

    # we clean the Name variable

    combined.drop('Name',axis=1,inplace=True)

    

    # encoding in dummy variable

    titles_dummies = pd.get_dummies(combined['Title'],prefix='Title')

    combined = pd.concat([combined,titles_dummies],axis=1)

    

    # removing the title variable

    combined.drop('Title',axis=1,inplace=True)

    

    status('names')
process_names()
combined.head()
#This function  replaces one missing Fare value by the mean

def process_fares():

    

    global combined

    # there's one missing fare value - replacing it with the mean.

    combined.Fare.fillna(combined.Fare.mean(),inplace=True)

    

    status('fare')
process_fares()
# This functions replaces the two missing values of Embarked with the most frequent Embarked value.

def process_embarked():

    

    global combined

    # two missing embarked values - filling them with the most frequent one (S)

    combined.Embarked.fillna('S',inplace=True)

    

    # dummy encoding 

    embarked_dummies = pd.get_dummies(combined['Embarked'],prefix='Embarked')

    combined = pd.concat([combined,embarked_dummies],axis=1)

    combined.drop('Embarked',axis=1,inplace=True)

    

    status('embarked')
process_embarked()
# This function replaces NaN values with U (for Unknow). 

# It then maps each Cabin value to the first letter. 

#Then it encodes the cabin values using dummy encoding .

def process_cabin():

    

    global combined

    

    # replacing missing cabins with U (for Uknown)

    combined.Cabin.fillna('U',inplace=True)

    

    # mapping each Cabin value with the cabin letter

    combined['Cabin'] = combined['Cabin'].map(lambda c : c[0])

    

    # dummy encoding ...

    cabin_dummies = pd.get_dummies(combined['Cabin'],prefix='Cabin')

    

    combined = pd.concat([combined,cabin_dummies],axis=1)

    

    combined.drop('Cabin',axis=1,inplace=True)

    

    status('cabin')
process_cabin()
combined.info()
combined.head()
#This function maps the string values male and female to 1 and 0 respectively.

def process_sex():

    

    global combined

    # mapping string values to numerical one 

    combined['Sex'] = combined['Sex'].map({'male':1,'female':0})

    

    status('sex')
process_sex()
# This function encodes the values of Pclass (1,2,3) using a dummy encoding.

def process_pclass():

    

    global combined

    # encoding into 3 categories:

    pclass_dummies = pd.get_dummies(combined['Pclass'],prefix="Pclass")

    

    # adding dummy variables

    combined = pd.concat([combined,pclass_dummies],axis=1)

    

    # removing "Pclass"

    

    combined.drop('Pclass',axis=1,inplace=True)

    

    status('pclass')
process_pclass()
def process_ticket():

    

    global combined

    

    # a function that extracts each prefix of the ticket, returns 'XXX' if no prefix (i.e the ticket is a digit)

    def cleanTicket(ticket):

        ticket=''

        ticket = ticket.replace('.','')

        ticket = ticket.replace('/','')

        ticket = ticket.split()

        ticket = map(lambda t : t.strip() , ticket)

        ticket = list(filter(lambda t : not t.isdigit(), ticket))       

        if len(ticket) > 0:

            return ticket[0]

        else: 

            return 'XXX'

    



    # Extracting dummy variables from tickets:



    combined['Ticket'] = combined['Ticket'].map(cleanTicket)

    tickets_dummies = pd.get_dummies(combined['Ticket'],prefix='Ticket')

    combined = pd.concat([combined, tickets_dummies],axis=1)

    combined.drop('Ticket',inplace=True,axis=1)



    status('ticket')
process_ticket()
def process_family():

    

    global combined

    # introducing a new feature : the size of families (including the passenger)

    combined['FamilySize'] = combined['Parch'] + combined['SibSp'] + 1

    

    # introducing other features based on the family size

    combined['Singleton'] = combined['FamilySize'].map(lambda s : 1 if s == 1 else 0)

    combined['SmallFamily'] = combined['FamilySize'].map(lambda s : 1 if 2<=s<=4 else 0)

    combined['LargeFamily'] = combined['FamilySize'].map(lambda s : 1 if 5<=s else 0)
process_family()
combined.shape
combined.head()
def scale_all_features():

    

    global combined

    

    features = list(combined.columns)

    features.remove('PassengerId')

    combined[features] = combined[features].apply(lambda x: x/x.max(), axis=0)

    

    print ('Features scaled successfully !')
scale_all_features()
from sklearn.pipeline import make_pipeline

from sklearn.ensemble import RandomForestClassifier

from sklearn.feature_selection import SelectKBest

from sklearn.cross_validation import StratifiedKFold

from sklearn.grid_search import GridSearchCV

from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier

from sklearn.cross_validation import cross_val_score
def compute_score(clf, X, y,scoring='accuracy'):

    xval = cross_val_score(clf, X, y, cv = 5,scoring=scoring)

    return np.mean(xval)
def recover_train_test_target():

    global combined

    

    train0 = pd.read_csv('../input/train.csv')

    

    targets = train0.Survived

    train = combined.ix[0:890]

    test = combined.ix[891:]

    

    return train,test,targets
train,test,targets = recover_train_test_target()
from sklearn.ensemble import ExtraTreesClassifier

from sklearn.feature_selection import SelectFromModel

clf = ExtraTreesClassifier(n_estimators=200)

clf = clf.fit(train, targets)
features = pd.DataFrame()

features['feature'] = train.columns

features['importance'] = clf.feature_importances_
features.sort(['importance'],ascending=False)
model = SelectFromModel(clf, prefit=True)

train_new = model.transform(train)

train_new.shape


test_new = model.transform(test)

test_new.shape


forest = RandomForestClassifier(max_features='sqrt')



parameter_grid = {

                 'max_depth' : [4,5,6,7,8],

                 'n_estimators': [200,210,240,250],

                 'criterion': ['gini','entropy']

                 }



cross_validation = StratifiedKFold(targets, n_folds=5)



grid_search = GridSearchCV(forest,

                           param_grid=parameter_grid,

                           cv=cross_validation)



grid_search.fit(train_new, targets)



print('Best score: {}'.format(grid_search.best_score_))

print('Best parameters: {}'.format(grid_search.best_params_))
output = grid_search.predict(test_new).astype(int)

df_output = pd.DataFrame()

df_output['PassengerId'] = test['PassengerId']

df_output['Survived'] = output

df_output[['PassengerId','Survived']].to_csv('titanic_pred.csv',index=False)