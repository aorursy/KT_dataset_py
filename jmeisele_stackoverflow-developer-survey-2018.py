#Since this ia a quick and dirty Kaggle notebook we will just use the basics
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
print(os.listdir("../input"))
import seaborn as sns
%matplotlib inline
from sklearn.preprocessing import LabelEncoder
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
    corr = results.corr()
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
#Load the data
results = pd.read_csv("../input/survey_results_public.csv")
schema = pd.read_csv("../input/survey_results_schema.csv")
results.head()
#The metadata or schema provides a brief description of all our variables with respect to questions asked on the survey
schema
# I always like to move my tagret variable to the front column, just a personal preference
results.set_index('CareerSatisfaction').reset_index()
results.dtypes
#Check for NaN values
results.isnull().sum()
#Transform hobby catgeorical variable to dummy variables
hobby = results[['Hobby']]
hobby = pd.get_dummies(hobby, prefix = 'Hobby')
hobby.head()
#Drop Hobby column
results = results.drop(columns = ['Hobby'])
#Concatinate new Hobby dummy variables columns
results = pd.concat([results, hobby], axis = 1)
print("The shape of our data is: \n {}".format(results.shape))
results.head()
#Do the same for OpenSource since it also does not have any missing values
open_source = results[['OpenSource']]
open_source = pd.get_dummies(open_source, prefix = 'OpenSource')
results = results.drop(columns = ['OpenSource'])
results = pd.concat([results, open_source], axis = 1)
print("The shape of our data is: \n {}".format(results.shape))
results.head()
job_satisfaction = pd.Series(results.JobSatisfaction.value_counts())
plt.xlabel('Responses')
plt.title("Job Satisfaction")
job_satisfaction.plot(kind = 'barh')
gender = pd.Series(results.Gender.value_counts())
gender.dropna()
gender.plot.pie()
undergrad = pd.DataFrame(results.UndergradMajor.value_counts())
undergrad.dropna()
undergrad.plot(kind = 'barh')
yearscoding = pd.Series(results['YearsCoding'])
f, ax = plt.subplots(figsize = (15,4))
sns.countplot(y = yearscoding, palette = 'Greens_d')
#Lets convert a few other object datatypes in our DataFrame
results['Gender'] = results.Gender.astype(str)
results['Country'] = results.Country.astype(str)
results['Student'] = results.Student.astype(bool)
results['Employment'] = results.Employment.astype(str)
results['FormalEducation'] = results.FormalEducation.astype(str)
results['UndergradMajor'] = results.UndergradMajor.astype(str)
results['CompanySize'] = results.CompanySize.astype(str)
results['DevType'] = results.DevType.astype(str)
results['YearsCoding'] = results.YearsCoding.astype(str)
results['YearsCodingProf'] = results.YearsCodingProf.astype(str)
results['CareerSatisfaction'] = results.CareerSatisfaction.astype(str)
#Instantiate a Label Encoder Class
enc = LabelEncoder()
#Create a class to encode multiple columns in a DataFrame
class MultiColumnLabelEncoder:
    def __init__(self,columns = None):
        self.columns = columns # array of column names to encode

    def fit(self,X,y=None):
        return self # not relevant here

    def transform(self,X):
        '''
        Transforms columns of X specified in self.columns using
        LabelEncoder(). If no columns specified, transforms all
        columns in X.
        '''
        output = X.copy()
        if self.columns is not None:
            for col in self.columns:
                output[col] = LabelEncoder().fit_transform(output[col])
        else:
            for colname,col in output.iteritems():
                output[colname] = LabelEncoder().fit_transform(col)
        return output

    def fit_transform(self,X,y=None):
        return self.fit(X,y).transform(X)
MultiColumnLabelEncoder(columns = ['Gender', 'Country','Employment', 'FormalEducation', 'UndergradMajor', 'CompanySize', 'DevType']).fit_transform(results)

