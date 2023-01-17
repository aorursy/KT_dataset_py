##### INITALIZATION ####################################

#Ignore Warnings
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.preprocessing import Imputer, Normalizer, scale
from sklearn.cross_validation import train_test_split, StratifiedKFold
from sklearn.feature_selection import RFECV

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns

#Configure viusalizations
%matplotlib inline
mpl.style.use('ggplot')
sns.set_style('white')
pylab.rcParams[ 'figure.figsize' ] = 8, 6
train = pd.read_csv("../input/titanic/train.csv")
test = pd.read_csv("../input/titanic/test.csv")

full = train.append(test, ignore_index= True)
titanic = full[:891]

del train, test

print(' Datasets: ', 'full:' , full.shape , ' titanic:', titanic.shape)
########################################################
#####  DATA VISUALIZATION  #############################

titanic.head()

# Variable descriptions .............
# Survived: Survived (1) or died (0)
# Pclass: Passenger's class
# Name: Passenger's name
# Sex: Passenger's sex
# Age: Passenger's age
# SibSp: Number of siblings/spouses aboard
# Parch: Number of parents/children aboard
# Ticket: Ticket number
# Fare: Fare
# Cabin: Cabin
# Embarked: Port of embarkation


titanic.describe()
def plot_correlation_map(df):
    corr = titanic.corr()
    _, ax = plt.subplots( figsize = (12,10))
    cmap = sns.diverging_palette(220, 10, as_cmap = True)
    _ = sns.heatmap(corr, cmap = cmap, square = True, cbar_kws = {'shrink': .9} 
                    ,ax= ax, annot= True, annot_kws= { 'fontsize': 12 })
    
plot_correlation_map(titanic)
#High correlation between SibSp and Parch

def plot_distribution(df, var, target, **kwargs):
    row = kwargs.get('row', None)
    col = kwargs.get('col', None)
    facet = sns.FacetGrid(df, hue=target, aspect=4, row= row, col= col)
    facet.map(sns.kdeplot, var, shade= True)
    facet.set( xlim=(0, df[var].max()))
    facet.add_legend()
    
# Plot Distribution of age of passengers who surivied or did not survived.
plot_distribution(titanic, var= 'Age', target= 'Survived', row= 'Sex')
    
# Differences between surivial for different values is what will be used to seperate the target variable in the model.
# If the two lines were similar then this would not have been a good variable for our predictive model.
# Plot distribution of fare of passengers who survived or did not survived.
plot_distribution(titanic, var= 'Fare', target= 'Survived', row= 'Sex')
def plot_categories(df, cat, target, **kwargs):
    row = kwargs.get('row', None)
    col = kwargs.get('col', None)
    facet = sns.FacetGrid(df, row = row, col = col)
    facet.map(sns.barplot, cat, target)
    facet.add_legend()
    
# Plot of surivial rate by embarked category.
plot_categories(titanic, cat= 'Embarked', target= 'Survived')

# Plot of surivial rate by sex category.
plot_categories(titanic, cat= 'Sex', target= 'Survived')

# Plot of surivial rate by SibSp category.
plot_categories(titanic, cat= 'SibSp', target= 'Survived')

# Plot of surivial rate by Parch category.
plot_categories(titanic, cat= 'Parch', target= 'Survived')
########################################################
##### DATA PREPARATION #################################

sex = pd.Series( np.where( full.Sex == 'male', 1, 0) , name = 'Sex')
embarked = pd.get_dummies( full.Embarked , prefix = 'Embarked')
embarked.head()

pclass = pd.get_dummies( full.Pclass , prefix = 'PClass')
pclass.head()

# Replacing missing data with the average of the column
imputed = pd.DataFrame()
imputed['Age'] = full.Age.fillna( full.Age.mean() )
imputed['Fare'] = full.Fare.fillna( full.Fare.mean() )
imputed.head()


########################################################
##### FEATURE ENGINEERING ##############################

title = pd.DataFrame()
title['Title'] = full['Name'].map(lambda name: name.split(',')[1].split('.')[0].strip())

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

title['Title'] = title.Title.map(Title_Dictionary)
title = pd.get_dummies( title.Title)

title.head()

cabin = pd.DataFrame()

cabin['Cabin'] = full.Cabin.fillna('U')
cabin['Cabin'] = cabin['Cabin'].map(lambda c: c[0])
cabin = pd.get_dummies( cabin['Cabin'] , prefix = 'Cabin')
cabin.head()


def getTicketPrefix(ticket):
    ticket = ticket.replace('.', '' )
    ticket = ticket.replace('/', '' )
    ticket = ticket.split()
    ticket = map( lambda t: t.strip(), ticket)
    ticket = list(filter( lambda t: not t.isdigit() , ticket))
    if len(ticket) > 0:
        return ticket[0]
    else:
        return 'XXX'
    
ticket = pd.DataFrame()
ticket['Ticket'] = full['Ticket'].map(getTicketPrefix)
ticket = pd.get_dummies(ticket['Ticket'], prefix = 'Ticket')
ticket.head()

family = pd.DataFrame()
family['FamilySize'] = full['Parch'] + full['SibSp'] + 1
family['FamilySize_Single'] = family['FamilySize'].map(lambda t: 1 if t == 1 else 0)
family['FamilySize_Small'] = family['FamilySize'].map(lambda t: 1 if 2 <= t <= 4 else 0)
family['FamilySize_Large'] = family['FamilySize'].map(lambda t: 1 if 5 <= t else 0)

family.head()
# Variable Selection for Modelling
full_X = pd.concat([imputed, embarked, sex, cabin], axis = 1)
full_X.head()
train_valid_X = full_X[0:891]
train_valid_Y = titanic.Survived
test_X = full_X[891:]

train_X, valid_X, train_Y, valid_Y = train_test_split(train_valid_X, train_valid_Y, train_size = .7)

#print(full_X.shape , train_X.shape , valid_X.shape , train_Y.shape , valid_Y.shape , test_X.shape)

def plot_variable_importance(X, Y):
    tree = DecisionTreeClassifier(random_state = 99)
    tree.fit(X,Y)
    plot_model_var_imp(tree, X, Y)
    
    
def plot_model_var_imp(model, X, Y):
    imp = pd.DataFrame(
        model.feature_importances_  ,
        columns = ['Importance'] , 
        index = X.columns
    )
    imp = imp.sort_values(['Importance'] , ascending = True)
    imp[:10].plot(kind = 'barh')
    print(model.score(X, Y))
  
plot_variable_importance(train_X, train_Y)
########################################################
##### MODELLING ########################################

#Random Forest Classifier
model = RandomForestClassifier(n_estimators=100)

#SVM
#model = SVC()

#Gradient Boosting Classifier
#model = GradientBoostingClassifier()

#K-nearest neighbors
#model = KNeighborsClassifier(n_neighbors = 3)

#Gaussian Naive Bayes
#model = GaussianNB()

#logistic Regression
#model = LogisticRegression()

model.fit(train_X, train_Y)


########################################################
##### EVALUATION #######################################

print(model.score(train_X, train_Y), model.score(valid_X, valid_Y))
########################################################
##### DEPLOYMENT #######################################

test_Y = model.predict(test_X).astype(int)
passenger_id = full[891:].PassengerId
test = pd.DataFrame({ 'PassengerId': passenger_id, 'Survived': test_Y})
#test.shape
#test.head()
#test.to_csv('titanic_pred.csv', index=False)
