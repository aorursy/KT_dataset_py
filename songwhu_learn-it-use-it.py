# data analysis and wrangling

import numpy as np

import pandas as pd



# visualization

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



# Modelling Helpers

from sklearn.preprocessing import Imputer , Normalizer , scale

from sklearn.cross_validation import train_test_split , StratifiedKFold

from sklearn.feature_selection import RFECV



# Modelling Algorithms

from sklearn.tree import DecisionTreeClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier , GradientBoostingClassifier

from sklearn.linear_model import SGDClassifier

from sklearn.linear_model import Perceptron

train_data = pd.read_csv('../input/train.csv')

test_data = pd.read_csv('../input/test.csv')

full_data = [train_data, test_data]
train_data.head()
test_data.head()
train_data.info()

print('_'*50)

test_data.info()
train_data.describe()
test_data.describe()
train_data.describe(include=['O'])
test_data.describe(include=['O'])
corr = train_data.corr()

_, ax = plt.subplots(figsize = (12, 10))

cmap = sns.diverging_palette(220, 10, as_cmap=True)

sns.heatmap(corr, cmap=cmap, square=True, cbar_kws={'shrink': .9}, ax = ax, annot=True,annot_kws={'fontsize' : 12})
train_data[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train_data[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived',ascending=False)
train_data[['SibSp', 'Survived']].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train_data[['Parch', 'Survived']].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)
g = sns.FacetGrid(train_data, col='Survived')

g.map(plt.hist, 'Age', bins=20)
g = sns.FacetGrid(train_data, col='Survived')

g.map(plt.hist, 'Fare', bins=3)
grid = sns.FacetGrid(train_data, col='Survived', row='Pclass', size=2.2, aspect=1.6)

grid.map(plt.hist, 'Age', alpha=.5, bins=20)

grid.add_legend()
grid = sns.FacetGrid(train_data, row='Embarked', size=2.2, aspect=1.6)

grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')

grid.add_legend()
def plot_distribution( df , var , target , **kwargs ):

    row = kwargs.get( 'row' , None )

    col = kwargs.get( 'col' , None )

    facet = sns.FacetGrid( df , hue=target , aspect=3 , row = row , col = col )

    facet.map( sns.kdeplot , var , shade= True )

    facet.set( xlim=( 0 , df[ var ].max() ) )

    facet.add_legend()

    

plot_distribution( train_data , var = 'Age' , target = 'Survived' , row = 'Sex' )    
plot_distribution( train_data , var = 'Fare' , target = 'Survived' , row = 'Sex' )    
grid = sns.FacetGrid(train_data, row='Embarked', col='Survived', size=2.2, aspect=1.6)

grid.map(sns.barplot, 'Sex', 'Fare', alpha=.5, ci=None)

grid.add_legend()
full_data = train_data.append(test_data, ignore_index=True)

train_data = full_data[:891]
full_data = full_data.drop(['Cabin'], axis = 1)
full_data['Sex'] = full_data['Sex'].map( {'female': 1, 'male': 0} ).astype(int)
guess_ages = np.zeros((2,3))

for i in range(0, 2):

    for j in range(0, 3):

        guess_df = full_data[(full_data['Sex'] == i) & \

                              (full_data['Pclass'] == j+1)]['Age'].dropna()

            

        age_guess = guess_df.median()



        # Convert random age float to nearest .5 age

        guess_ages[i,j] = int( age_guess/0.5 + 0.5 ) * 0.5

            

for i in range(0, 2):

    for j in range(0, 3):

        full_data.loc[ (full_data.Age.isnull()) & (full_data.Sex == i) & (full_data.Pclass == j+1),\

                    'Age'] = guess_ages[i,j]



full_data['Age'] = full_data['Age'].astype(int)

full_data.head()
freq_port = full_data.Embarked.dropna().mode()[0]

full_data.Embarked = full_data.Embarked.fillna(freq_port)

full_data.info()
full_data.Fare = full_data.Fare.fillna(full_data.Fare.dropna().mean())
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

full_data['Title'] = full_data.Name.map(lambda name: name.split( ',' )[1].split( '.' )[0].strip())

full_data.Title = full_data.Title.map(Title_Dictionary)

full_data = full_data.drop(['Name'], axis=1)
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



full_data.Ticket = full_data.Ticket.map(cleanTicket)
full_data.head()
title_dm = pd.get_dummies(full_data.Title, prefix = 'Title')

embarked_dm = pd.get_dummies(full_data.Embarked, prefix = 'Embarked')

ticket_dm = pd.get_dummies(full_data.Ticket, prefix = 'Ticket')

full_data = pd.concat([full_data, title_dm, embarked_dm, ticket_dm], axis=1)

full_data = full_data.drop(['Ticket', 'Embarked', 'Title'], axis=1)

full_data.head()
train_df = full_data[ :891]

train_df['AgeBand'] = pd.cut(train_df['Age'], 5)

train_df[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True)
full_data.loc[ full_data['Age'] <= 16, 'Age'] = 0

full_data.loc[(full_data['Age'] > 16) & (full_data['Age'] <= 32), 'Age'] = 1

full_data.loc[(full_data['Age'] > 32) & (full_data['Age'] <= 48), 'Age'] = 2

full_data.loc[(full_data['Age'] > 48) & (full_data['Age'] <= 64), 'Age'] = 3

full_data.loc[ full_data['Age'] > 64, 'Age']  = 4

age_dm = pd.get_dummies(full_data.Age, prefix='Age')

full_data = pd.concat([full_data, age_dm], axis=1)

full_data = full_data.drop(['Age'], axis=1)

full_data.head()
train_df = full_data[ :891 ]

train_df['FareBand'] = pd.qcut(train_df['Fare'], 4)

train_df[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand', ascending=True)
full_data.info()
full_data.loc[ full_data['Fare'] <= 7.91, 'Fare'] = 0

full_data.loc[(full_data['Fare'] > 7.91) & (full_data['Fare'] <= 14.454), 'Fare'] = 1

full_data.loc[(full_data['Fare'] > 14.454) & (full_data['Fare'] <= 31), 'Fare']   = 2

full_data.loc[ full_data['Fare'] > 31, 'Fare'] = 3

full_data['Fare'] = full_data['Fare'].astype(int)

fare_dm = pd.get_dummies(full_data.Fare, prefix='Fare')

full_data = pd.concat([full_data, fare_dm], axis = 1)

full_data = full_data.drop('Fare', axis=1)

full_data.head()
full_data['FamilySize'] = full_data.SibSp + full_data.Parch + 1

full_data['IsAlone'] = 0

full_data.loc[full_data.FamilySize == 1, 'IsAlone'] = 1

full_data = full_data.drop(['FamilySize'], axis=1)

full_data.head()
# Create all datasets that are necessary to train, validate and test models

train_valid_X = full_data[ 0:891 ].drop(['Survived'], axis= 1)

train_valid_y = full_data[ 0:891 ].Survived

test_X = full_data[ 891: ].drop(['Survived'], axis= 1)

train_X , valid_X , train_y , valid_y = train_test_split( train_valid_X , train_valid_y , train_size = .7 )



print (full_data.shape , train_X.shape , valid_X.shape , train_y.shape , valid_y.shape , test_X.shape)
# data analysis and wrangling

import numpy as np

import pandas as pd



# visualization

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



# Modelling Helpers

from sklearn.preprocessing import Imputer , Normalizer , scale

from sklearn.cross_validation import train_test_split , StratifiedKFold

from sklearn.feature_selection import RFECV



# Modelling Algorithms

from sklearn.tree import DecisionTreeClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier , GradientBoostingClassifier
train_data = pd.read_csv('../input/train.csv')

test_data = pd.read_csv('../input/test.csv')

full_data = [train_data, test_data]
train_data.head()
test_data.head()
train_data.info()

print('_'*50)

test_data.info()
train_data.describe()
test_data.describe()
train_data.describe(include=['O'])
test_data.describe(include=['O'])
corr = train_data.corr()

_, ax = plt.subplots(figsize = (12, 10))

cmap = sns.diverging_palette(220, 10, as_cmap=True)

sns.heatmap(corr, cmap=cmap, square=True, cbar_kws={'shrink': .9}, ax = ax, annot=True,annot_kws={'fontsize' : 12})
train_data[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train_data[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived',ascending=False)
train_data[['SibSp', 'Survived']].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train_data[['Parch', 'Survived']].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)
g = sns.FacetGrid(train_data, col='Survived')

g.map(plt.hist, 'Age', bins=20)
g = sns.FacetGrid(train_data, col='Survived')

g.map(plt.hist, 'Fare', bins=3)
grid = sns.FacetGrid(train_data, col='Survived', row='Pclass', size=2.2, aspect=1.6)

grid.map(plt.hist, 'Age', alpha=.5, bins=20)

grid.add_legend()
grid = sns.FacetGrid(train_data, row='Embarked', size=2.2, aspect=1.6)

grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')

grid.add_legend()
def plot_distribution( df , var , target , **kwargs ):

    row = kwargs.get( 'row' , None )

    col = kwargs.get( 'col' , None )

    facet = sns.FacetGrid( df , hue=target , aspect=3 , row = row , col = col )

    facet.map( sns.kdeplot , var , shade= True )

    facet.set( xlim=( 0 , df[ var ].max() ) )

    facet.add_legend()

    

plot_distribution( train_data , var = 'Age' , target = 'Survived' , row = 'Sex' )    
plot_distribution( train_data , var = 'Fare' , target = 'Survived' , row = 'Sex' )    
grid = sns.FacetGrid(train_data, row='Embarked', col='Survived', size=2.2, aspect=1.6)

grid.map(sns.barplot, 'Sex', 'Fare', alpha=.5, ci=None)

grid.add_legend()
full_data = train_data.append(test_data, ignore_index=True)

train_data = full_data[:891]
full_data = full_data.drop(['Cabin'], axis = 1)
full_data['Sex'] = full_data['Sex'].map( {'female': 1, 'male': 0} ).astype(int)
guess_ages = np.zeros((2,3))

for i in range(0, 2):

    for j in range(0, 3):

        guess_df = full_data[(full_data['Sex'] == i) & \

                              (full_data['Pclass'] == j+1)]['Age'].dropna()

            

        age_guess = guess_df.median()



        # Convert random age float to nearest .5 age

        guess_ages[i,j] = int( age_guess/0.5 + 0.5 ) * 0.5

            

for i in range(0, 2):

    for j in range(0, 3):

        full_data.loc[ (full_data.Age.isnull()) & (full_data.Sex == i) & (full_data.Pclass == j+1),\

                    'Age'] = guess_ages[i,j]



full_data['Age'] = full_data['Age'].astype(int)

full_data.head()
freq_port = full_data.Embarked.dropna().mode()[0]

full_data.Embarked = full_data.Embarked.fillna(freq_port)

full_data.info()
full_data.Fare = full_data.Fare.fillna(full_data.Fare.dropna().mean())
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

full_data['Title'] = full_data.Name.map(lambda name: name.split( ',' )[1].split( '.' )[0].strip())

full_data.Title = full_data.Title.map(Title_Dictionary)

full_data = full_data.drop(['Name'], axis=1)
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



full_data.Ticket = full_data.Ticket.map(cleanTicket)
full_data.head()
title_dm = pd.get_dummies(full_data.Title, prefix = 'Title')

embarked_dm = pd.get_dummies(full_data.Embarked, prefix = 'Embarked')

ticket_dm = pd.get_dummies(full_data.Ticket, prefix = 'Ticket')

full_data = pd.concat([full_data, title_dm, embarked_dm, ticket_dm], axis=1)

full_data = full_data.drop(['Ticket', 'Embarked', 'Title'], axis=1)

full_data.head()
train_df = full_data[ :891]

train_df['AgeBand'] = pd.cut(train_df['Age'], 5)

train_df[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True)
full_data.loc[ full_data['Age'] <= 16, 'Age'] = 0

full_data.loc[(full_data['Age'] > 16) & (full_data['Age'] <= 32), 'Age'] = 1

full_data.loc[(full_data['Age'] > 32) & (full_data['Age'] <= 48), 'Age'] = 2

full_data.loc[(full_data['Age'] > 48) & (full_data['Age'] <= 64), 'Age'] = 3

full_data.loc[ full_data['Age'] > 64, 'Age']  = 4

age_dm = pd.get_dummies(full_data.Age, prefix='Age')

full_data = pd.concat([full_data, age_dm], axis=1)

full_data = full_data.drop(['Age'], axis=1)

full_data.head()
train_df = full_data[ :891 ]

train_df['FareBand'] = pd.qcut(train_df['Fare'], 4)

train_df[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand', ascending=True)
full_data.info()
full_data.loc[ full_data['Fare'] <= 7.91, 'Fare'] = 0

full_data.loc[(full_data['Fare'] > 7.91) & (full_data['Fare'] <= 14.454), 'Fare'] = 1

full_data.loc[(full_data['Fare'] > 14.454) & (full_data['Fare'] <= 31), 'Fare']   = 2

full_data.loc[ full_data['Fare'] > 31, 'Fare'] = 3

full_data['Fare'] = full_data['Fare'].astype(int)

fare_dm = pd.get_dummies(full_data.Fare, prefix='Fare')

full_data = pd.concat([full_data, fare_dm], axis = 1)

full_data = full_data.drop('Fare', axis=1)

full_data.head()
full_data['FamilySize'] = full_data.SibSp + full_data.Parch + 1

full_data['IsAlone'] = 0

full_data.loc[full_data.FamilySize == 1, 'IsAlone'] = 1

full_data = full_data.drop(['FamilySize'], axis=1)

full_data.head()
# Create all datasets that are necessary to train, validate and test models

train_valid_X = full_data[ 0:891 ].drop(['Survived'], axis= 1)

train_valid_y = full_data[ 0:891 ].Survived

test_X = full_data[ 891: ].drop(['Survived'], axis= 1)

train_X , valid_X , train_y , valid_y = train_test_split( train_valid_X , train_valid_y , train_size = .7 )



print (full_data.shape , train_X.shape , valid_X.shape , train_y.shape , valid_y.shape , test_X.shape)
dt_model = DecisionTreeClassifier()

rf_model = RandomForestClassifier(n_estimators=100)

svc_model = SVC()

lsv_model = LinearSVC()

gb_model = GradientBoostingClassifier()

knn_model = KNeighborsClassifier(n_neighbors=3)

gnb_model = GaussianNB()

lr_model = LogisticRegression()

sgd_model = SGDClassifier()

perceptron_model = Perceptron()
dt_model.fit(train_X, train_y)

rf_model.fit(train_X, train_y)

svc_model.fit(train_X, train_y)

lsv_model.fit(train_X, train_y)

gb_model.fit(train_X, train_y)

knn_model.fit(train_X, train_y)

gnb_model.fit(train_X, train_y)

lr_model.fit(train_X, train_y)

sgd_model.fit(train_X, train_y)

perceptron_model.fit(train_X, train_y)
dt_train = dt_model.score( train_X , train_y )

dt_valid = dt_model.score( valid_X , valid_y )

rf_train = rf_model.score( train_X , train_y )

rf_valid = rf_model.score( valid_X , valid_y )

svc_train = svc_model.score( train_X , train_y )

svc_valid = svc_model.score( valid_X , valid_y )

lsv_train = lsv_model.score( train_X , train_y )

lsv_valid = lsv_model.score( valid_X , valid_y )

gb_train = gb_model.score( train_X , train_y )

gb_valid = gb_model.score( valid_X , valid_y )

knn_train = knn_model.score( train_X , train_y )

knn_valid = knn_model.score( valid_X , valid_y )

gnb_train = gnb_model.score( train_X , train_y )

gnb_valid = gnb_model.score( valid_X , valid_y )

lr_train = lr_model.score( train_X , train_y )

lr_valid = lr_model.score( valid_X , valid_y )

sgd_train = sgd_model.score( train_X , train_y )

sgd_valid = sgd_model.score( valid_X , valid_y )

perceptron_train = perceptron_model.score( train_X , train_y )

perceptron_valid = perceptron_model.score( valid_X , valid_y )
dt_train = dt_model.score( train_X , train_y )

dt_valid = dt_model.score( valid_X , valid_y )

rf_train = rf_model.score( train_X , train_y )

rf_valid = rf_model.score( valid_X , valid_y )

svc_train = svc_model.score( train_X , train_y )

svc_valid = svc_model.score( valid_X , valid_y )

lsv_train = lsv_model.score( train_X , train_y )

lsv_valid = lsv_model.score( valid_X , valid_y )

gb_train = gb_model.score( train_X , train_y )

gb_valid = gb_model.score( valid_X , valid_y )

knn_train = knn_model.score( train_X , train_y )

knn_valid = knn_model.score( valid_X , valid_y )

gnb_train = gnb_model.score( train_X , train_y )

gnb_valid = gnb_model.score( valid_X , valid_y )

lr_train = lr_model.score( train_X , train_y )

lr_valid = lr_model.score( valid_X , valid_y )

models = pd.DataFrame({

    'Model': ['dt', 'rf', 'svc', 

              'lsv', 'gb', 'knn', 

              'gnb', 'lr', 

              'sgd','perceptron'],

    'train_Score': [dt_train, rf_train, svc_train, lsv_train, 

                   gb_train, knn_train, gnb_train, lr_train, 

                   sgd_train, perceptron_train],

    'valid_Score': [dt_valid, rf_valid, svc_valid, lsv_valid, 

                   gb_valid, knn_valid, gnb_valid, lr_valid, 

                   sgd_valid, perceptron_valid]

})

models.sort_values(by='valid_Score', ascending=False)
rfecv = RFECV( estimator = rf_model , step = 1 , cv = StratifiedKFold( train_y , 2 ) , scoring = 'accuracy' )

rfecv.fit( train_X , train_y )



print (rfecv.score( train_X , train_y ) , rfecv.score( valid_X , valid_y ))

print( "Optimal number of features : %d" % rfecv.n_features_ )



# Plot number of features VS. cross-validation scores

plt.figure()

plt.xlabel( "Number of features selected" )

plt.ylabel( "Cross validation score (nb of correct classifications)" )

plt.plot( range( 1 , len( rfecv.grid_scores_ ) + 1 ) , rfecv.grid_scores_ )

plt.show()
rf_model.fit(train_valid_X, train_valid_y)

test_Y = rf_model.predict( test_X ).astype(int)

passenger_id = full_data[891:].PassengerId

test = pd.DataFrame( { 'PassengerId': passenger_id , 'Survived': test_Y } )

test.shape

test.head()

test.to_csv( 'titanic_pred.csv' , index = False )