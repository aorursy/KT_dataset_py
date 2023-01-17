# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Ignore warnings

import warnings

warnings.filterwarnings('ignore')



# Handle table-like data and matrices

import numpy as np

import pandas as pd



# Modelling Algorithms

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier , GradientBoostingClassifier

from xgboost import XGBRegressor



# Modelling Helpers

from sklearn.preprocessing import Imputer , Normalizer , scale

from sklearn.model_selection import train_test_split , StratifiedKFold, GridSearchCV



# Encoders

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import OneHotEncoder



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



def plot_distribution( df , var , target , title, **kwargs ):

    row = kwargs.get( 'row' , None )

    col = kwargs.get( 'col' , None )

    facet = sns.FacetGrid( df , hue=target , aspect=4 , row = row , col = col )

    facet.map( sns.kdeplot , var , shade= True )

    facet.set( xlim=( 0 , df[ var ].max() ) )

    facet.add_legend()

    plt.subplots_adjust(top=0.8)

    plt.suptitle(title, fontsize = 16)



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

df_train = pd.read_csv("/kaggle/input/titanic/train.csv")

df_test = pd.read_csv("/kaggle/input/titanic/test.csv")



# append train and test datasets to easy data preparation

full = df_train.append( df_test , ignore_index = True )

df_train = full[ :891 ]

df_test = full[ 891: ]
df_train.head()
df_train.describe()
df_train.info()
# obtain "string" columns

object_cols = (df_train.dtypes == 'object')

object_cols = list(object_cols[object_cols].index)

print(object_cols)



# obtain numerical columns

num_cols = ((df_train.dtypes == 'int64') | (df_train.dtypes == 'float64'))

num_cols = list(num_cols[num_cols].index)

print(num_cols)
df_test.describe()
df_test.info()
plot_correlation_map(df_train)
# Plot distributions of Age of passangers who survived or did not survive

plot_distribution( df_train , var = 'Age' , target = 'Survived' , row = 'Sex' ,

                  title = 'Surviving vs Age and Sex')



# Plot distributions of Fare of passangers who survived or did not survive

plot_distribution( df_train , var = 'Fare' , target = 'Survived' , row = 'Sex',

                  title = 'Surviving vs Fare and Sex')
# Plot survival rate by Sex

plot_categories(df_train, cat = 'Sex', target = 'Survived' )



# Plot survival rate by Pclass

plot_categories( df_train , cat = 'Pclass' , target = 'Survived' )



# Plot survival rate by Parch

plot_categories( df_train , cat = 'Parch' , target = 'Survived' )



# Plot survival rate by SibSp

plot_categories( df_train , cat = 'SibSp' , target = 'Survived' )



# Plot survival rate by Embarked

plot_categories( df_train , cat = 'Embarked' , target = 'Survived' )
# Shape of training data (num_rows, num_columns)

print(df_train.shape)



# Number of missing values in each column of training data

missing_val_count_by_column = (df_train.isnull().sum())

print(missing_val_count_by_column[missing_val_count_by_column > 0])
# Shape of testing data (num_rows, num_columns)

print(df_test.shape)



# Number of missing values in each column of testing data

missing_val_count_by_column = (df_test.isnull().sum())

print(missing_val_count_by_column[missing_val_count_by_column > 0])
full['Age'] = full['Age'].fillna(full.Age.median())

full['Fare'] = full['Fare'].fillna(full.Age.median())
full.describe()
# Replacing missing cabins and embarked with Unknown_ ...

full['Cabin'] = full.Cabin.fillna('Unknown_Cabin')

full['Embarked'] = full.Embarked.fillna('Unknown_Embarked')
# All categorical columns

object_cols = [col for col in full.columns if full[col].dtype == "object"]



# Get number of unique entries in each column with categorical data

object_nunique = list(map(lambda col: full[col].nunique(), object_cols))

d = dict(zip(object_cols, object_nunique))



# Print number of unique entries by column, in ascending order

sorted(d.items(), key=lambda x: x[1])
# Apply label encoder to column Sex

label_encoder = LabelEncoder()

full['Sex'] = label_encoder.fit_transform(full['Sex'])
# Apply one-hot encoder to each column with categorical data

categorical = ['Embarked']

OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)

OH_embarked = pd.DataFrame(OH_encoder.fit_transform(full[categorical]))



# One-hot encoding removed index; put it back

OH_embarked.index = full.index

OH_embarked
full.drop(['Embarked'], axis=1, inplace=True)

full = pd.concat([full, OH_embarked], axis=1)

full
# separate X and y (target : survived)

y = full.Survived[:891]

y
full.drop(['Cabin', 'Name', 'PassengerId', 'Ticket', 'Survived', 'Parch', 'SibSp'], 

              axis=1, inplace=True)
X = full[:891]

X
# Break off validation set from training data

X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,

                                                      random_state=0)
print(f"Shapes of X_train {X_train.shape} and y_train {y_train.shape}")

print(f"Shapes of X_valid {X_valid.shape} and y_valid {y_valid.shape}")
model = RandomForestClassifier(random_state=0)



param_grid = {'n_estimators': range(100,1000,50)}



grid_search = GridSearchCV(model, param_grid , cv=10, iid=False)

grid_search.fit(X_train, y_train)



print(f"best parameters from grid search: {grid_search.best_params_} accuracy : {grid_search.score(X_valid, y_valid):.3f}")
model = SVC(random_state=0)



param_grid = {

    'kernel': ['linear', 'rbf'], 

    'C': [1, 10],

    'gamma': [0.001, 0.0001]

    }



grid_search = GridSearchCV(model, param_grid , cv=10, iid=False)

grid_search.fit(X_train, y_train)



print(f"best parameters from grid search: {grid_search.best_params_} accuracy : {grid_search.score(X_valid, y_valid):.3f}") 
model = GradientBoostingClassifier()



param_grid = {

    'n_estimators': [100,200,500,750,1000],

    'learning_rate': [0.05, 0.1, 0.5, 1]

}



grid_search = GridSearchCV(model, param_grid , cv=10, iid=False)

grid_search.fit(X_train, y_train)



print(f"best parameters from grid search: {grid_search.best_params_} accuracy : {grid_search.score(X_valid, y_valid):.3f}") 
model = GaussianNB()



param_grid = {

    }



grid_search = GridSearchCV(model, param_grid , cv=10, iid=False)

grid_search.fit(X_train, y_train)



print(f"best parameters from grid search: {grid_search.best_params_} accuracy : {grid_search.score(X_valid, y_valid):.3f}") 
model = LogisticRegression(random_state=0)



param_grid = {

    'C': [0.1, 1.0, 10, 100]

}



grid_search = GridSearchCV(model, param_grid, cv=10, iid=False)

grid_search.fit(X_train, y_train)



print(f"best parameters from grid search: {grid_search.best_params_} accuracy : {grid_search.score(X_valid, y_valid):.3f}") 
model = KNeighborsClassifier()



param_grid = {

    'n_neighbors': range(1,15,2)

}



grid_search = GridSearchCV(model, param_grid, cv=10, iid=False)

grid_search.fit(X_train, y_train)



print(f"best parameters from grid search: {grid_search.best_params_} accuracy : {grid_search.score(X_valid, y_valid):.3f}") 
model = XGBRegressor(n_estimators=1000, learning_rate=0.05, random_state=0)



param_grid = {

    'n_estimators': [100,200,500,750,1000],

    'learning_rate': [0.05, 0.1, 0.5, 1]

}



grid_search = GridSearchCV(model, param_grid , cv=10, iid=False)

grid_search.fit(X_train, y_train)



print(f"best parameters from grid search: {grid_search.best_params_} accuracy : {grid_search.score(X_valid, y_valid):.3f}")


# Define and fit model

model = RandomForestClassifier(n_estimators=150, random_state=0)

model.fit(X_train, y_train)



# Predict target vector

X_test = full[891:]
X_test
# Predict target vector

preds_test = model.predict(X_test)
# Save test predictions to file

output = pd.DataFrame({'PassengerId': df_test.PassengerId,

                       'Survived': preds_test})

output.to_csv('submission.csv', index=False)
output