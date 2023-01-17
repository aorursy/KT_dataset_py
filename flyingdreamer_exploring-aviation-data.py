# Python libraries

import math

import re

import datetime





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

from sklearn.preprocessing import StandardScaler, Imputer , Normalizer , scale

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

pylab.rcParams[ 'figure.figsize' ] = 12 , 10
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

    facet.set( xlim=( df[ var ].min() , df[ var ].max() ) )

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

    

def category_values(dataframe, categories):

    for c in categories:

        print('\n', dataframe.groupby(by=c)[c].count().sort_values(ascending=False))

        print('Nulls: ', dataframe[c].isnull().sum())
df = pd.read_csv('../input/AviationDataEnd2016UP.csv', sep=',', header=0, encoding = 'iso-8859-1')



df.sample(10)
df.info()
df.describe()
describe_more(df)
# splitting date field in the components



df['Year'] = df['Event.Date'].apply(lambda d: datetime.datetime.strptime(d, "%Y-%m-%d").year)

df['Month'] = df['Event.Date'].apply(lambda d: datetime.datetime.strptime(d, "%Y-%m-%d").month)

df['Day'] = df['Event.Date'].apply(lambda d: datetime.datetime.strptime(d, "%Y-%m-%d").day)



df = df[df['Year'] >= 1982]
categories = ['Investigation.Type',

             'Aircraft.Damage',

             'Aircraft.Category',

             'Amateur.Built',

             'Number.of.Engines',

             'Engine.Type',

             'FAR.Description',

             'Schedule',

             'Purpose.of.Flight',

             'Weather.Condition',

             'Broad.Phase.of.Flight',

             'Report.Status',

             'Air.Carrier']



for c in categories:

    print(c , df[c].unique())
category_values(df, categories)
# null damages can't be defined

df[df['Aircraft.Damage'].isnull()]

df['Aircraft.Damage'].fillna('Unknown', inplace=True)



# Fixing phase of flight nulls

df['Broad.Phase.of.Flight'].fillna('UNKNOWN', inplace=True)



# Fixing weather conditions

df['Weather.Condition'].fillna('UNK', inplace=True)



# null categories can't be defined

df['Aircraft.Category'].fillna('Unknown', inplace=True)



# can't define purpose of flight

df['Purpose.of.Flight'].fillna('Unknown', inplace=True)



# don't know ho to set missing schedules 

df['Schedule'].fillna('UNK', inplace=True)



# don't know ho to set missing FAR.Description

df['FAR.Description'].fillna('Unknown', inplace=True)



# don't know ho to set missing Aircraft.Damage

df['Aircraft.Damage'].fillna('Unknown', inplace=True)



# don't know ho to set missing Air Carriers

df['Air.Carrier'].fillna('Unknown', inplace=True)



# don't know ho to set missing Makers

df['Make'].fillna('UNKNOWN', inplace=True)



# don't know ho to set missing Models

df['Model'].fillna('Unknown', inplace=True)



# don't know ho to set missing airport names

df['Airport.Name'].fillna('Unknown', inplace=True)



# don't know ho to set missing Models

df['Airport.Code'].fillna('Unknown', inplace=True)



# don't know ho to set missing Locations

df['Location'].fillna('Unknown', inplace=True)
# Extracting producers and amateurs

producers = [x for x in df['Make'][df['Amateur.Built']== 'No'].unique() ]

amateurs  = [x for x in df['Make'][df['Amateur.Built']== 'Yes'].unique() ]



# -----------------------------------------------

# Function that fixes the null in amateur.built

def fix_amateur_built(ab, m):

    if type(ab) == str:

        return ab

    else:

        if m in producers:

            return 'No'

        else:

            return 'Yes'

# Fix for Amateur.Built field      

am_built = df.apply(lambda x: fix_amateur_built(x['Amateur.Built'], x['Make']), axis=1)

df = df.assign(AmateurBuilt = am_built, index=df.index)
# Function that fixes the null in number.of.engines

def fix_number_of_engines(noe, m):

    if noe >= 0:

        return noe

    else:

        # Setting number of engines at the mean number of engines for the producer

        r = np.round(df['Number.of.Engines'][df['Make']==m].mean())

        return r



# Setting 0 engines for balloons

df['Number.of.Engines'][df['Number.of.Engines'].isnull() & (df['Make'].str.contains('balloon', case=False))] = 0.0

# Correcting number of engines

num_engines = df.apply(lambda x: fix_number_of_engines(x['Number.of.Engines'], x['Make']), axis=1)

df = df.assign(NumberofEngines = num_engines, index=df.index)

# Still some null after number of engines correction

df['NumberofEngines'].fillna(1, inplace=True)
# Function that fixes the engine types

def fix_engine_type(et, model):

    if type(et) == str:

        return et

    else:

        # Setting engine type at the mode of engines for the model

        e = (df['Engine.Type'][df['Model']==model].mode())

        return  e[0] if e.count() > 0 else 'Unknown'

# Fix for Engine.Type field      

en_type = df.apply(lambda x: fix_engine_type(x['Engine.Type'], x['Model']), axis=1)

df = df.assign(EngineType = en_type, index=df.index)
# Function that fixes the Aircraft.Category

def fix_aircraft_category(cat, model):

    if type(cat) == str:

        return cat

    else:

        # Setting aircraft category at the mode of caterogories for the model

        e = (df['Aircraft.Category'][df['Model']==model].mode())

        return  e[0] if e.count() > 0 else 'Unknown'

# Fix for Aircraft.Category field      

aircraft_cat = df.apply(lambda x: fix_aircraft_category(x['Aircraft.Category'], x['Model']), axis=1)

df = df.assign(AircraftCategory = aircraft_cat, index=df.index)
# null countries are outside US

df[df['Country'].isnull()]

df['Country'].fillna('Foreign', inplace=True)
df['Injuries'] = df['Total.Fatal.Injuries'] + df['Total.Serious.Injuries'] + df['Total.Minor.Injuries']
#category_values(df, ['AircraftCategory', 'Country', 'EngineType', 'NumberofEngines', 'AmateurBuilt'])

#df['EngineType'].sample(100)



#df.groupby(by=['Location']).count()

df.isnull().sum()
df = df.drop(['Number.of.Engines', 'Aircraft.Category', 'Engine.Type', 'Amateur.Built', 'index'], axis='columns')

df = df.drop(['Publication.Date'], axis='columns')
plot_correlation_map(df)
# For the time series charts I start sorting data

df = df.sort_values(by=['Year', 'Month', 'Day'], ascending=True)



years = np.arange(1982, 2017)



sns.set(style="darkgrid")



plt.subplot(211)



g = sns.countplot(x="Year", data=df, palette="GnBu_d", order=years)

g.set_xticklabels(labels=years)

a = plt.setp(g.get_xticklabels(), rotation=90)
from sklearn.linear_model import LinearRegression, Ridge, Lasso

from sklearn.ensemble import RandomForestRegressor

from sklearn.preprocessing import PolynomialFeatures

from sklearn.pipeline import Pipeline

from sklearn.model_selection import cross_val_score





events_per_year = df.groupby(by='Year').count()['Event.Id']

events_per_year.drop(2017, axis=0, inplace=True)



X = [ [y] for y in events_per_year.index.values]

y = [ [e] for e in events_per_year.as_matrix()]





degrees = [1,2,3]

lr_pred_X = [[y] for y in range(1982, 2020)]

for i in range(len(degrees)):

    polynomial_features = PolynomialFeatures(degree=degrees[i],

                                             include_bias=False)

    linear_regression = LinearRegression()

    pipeline = Pipeline([("polynomial_features", polynomial_features),

                         ("linear_regression", linear_regression)])

    pipeline.fit(X, y)



    # Evaluate the models using crossvalidation

    scores = cross_val_score(pipeline, X, y,

                             scoring="neg_mean_squared_error", cv=10)

    lr_pred=pipeline.predict(lr_pred_X)

    plt.plot(lr_pred_X, lr_pred, alpha=.3)

    

    print("Score for degree %d: %.3f - prediction for 2017 is %d" % (i, pipeline.score(X, y), lr_pred[35]))



plt.plot(X, y)

plt.title("Linear regression with polynomial features")

plt.legend(labels=degrees)



plt.show()