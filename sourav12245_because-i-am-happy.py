import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from scipy import stats

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report, confusion_matrix

from sklearn.datasets import fetch_20newsgroups_vectorized

from sklearn.feature_selection import chi2

from sklearn.feature_selection import RFE

from sklearn.ensemble import ExtraTreesClassifier

from sklearn import datasets

from sklearn import metrics

import types

from sklearn.manifold import TSNE

import plotly.graph_objs as go

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=True)

%matplotlib inline



data_15 = pd.read_csv('../input/2015.csv')

data_16 = pd.read_csv('../input/2016.csv')

data_17 = pd.read_csv('../input/2017.csv')

#-------------------#------------------------#-------------------------#-----------------#-----------------

data_15 = data_15.drop(['Standard Error'], axis =1)

data_16 = data_16.drop(['Upper Confidence Interval','Lower Confidence Interval'], axis =1)

data_17 = data_17.drop(['Whisker.high','Whisker.low'], axis =1)

combine =[data_15,data_16,data_17]

#-------------------#----------------------------#----------------------#-----------------#---------------------

# clensing the data for 2017, so that it can be allined with 2015 and 2016 and renaming columns 

data_17 = data_17.reindex_axis(['Country', 'Happiness.Rank', 'Happiness.Score',

       'Economy..GDP.per.Capita.', 'Family', 'Health..Life.Expectancy.',

       'Freedom', 'Trust..Government.Corruption.', 'Generosity',

       'Dystopia.Residual'], axis =1)

data_17 = data_17.rename(columns={'Happiness.Rank': 'Happiness Rank', 'Happiness.Score': 'Happiness Score',

       'Economy..GDP.per.Capita.':'Economy (GDP per Capita)' , 'Health..Life.Expectancy.':'Health (Life Expectancy)' ,

       'Trust..Government.Corruption.':'Trust (Government Corruption)',

       'Dystopia.Residual':'Dystopia Residual'})

data_15 = data_15.rename(columns={'Economy (GDP per Capita)': 'Economy', 'Health (Life Expectancy)': 'Health',

       'Happiness Score':'Score' , 'Happiness Rank':'Rank' ,

       'Trust (Government Corruption)':'Trust',

       'Dystopia Residual':'Residual'})

data_16 = data_16.rename(columns={'Economy (GDP per Capita)': 'Economy', 'Health (Life Expectancy)': 'Health',

       'Happiness Score':'Score' , 'Happiness Rank':'Rank' ,

       'Trust (Government Corruption)':'Trust',

       'Dystopia Residual':'Residual'})

data_17 = data_17.rename(columns={'Economy (GDP per Capita)': 'Economy', 'Health (Life Expectancy)': 'Health',

       'Happiness Score':'Score' , 'Happiness Rank':'Rank' ,

       'Trust (Government Corruption)':'Trust',

       'Dystopia Residual':'Residual'})

#-------------------------------#---------------------#-------------------------#------------------

#------------#-------------------------#------------------#------------------#-------------#-----------------

#changing the name for better clarification of the region name.

# 2017 doent have region column in the dataset

combine_new = [data_15,data_16]

for dataset in combine_new:

    dataset['Region'] = dataset['Region'].replace('Western Europe','WEur')

    dataset['Region'] = dataset['Region'].replace('North America','NA')

    dataset['Region'] = dataset['Region'].replace('Australia and New Zealand','ANZ')

    dataset['Region'] = dataset['Region'].replace('Middle East and Northern Africa','MENA')

    dataset['Region'] = dataset['Region'].replace('Latin America and Caribbean','LAC')

    dataset['Region'] = dataset['Region'].replace('Southeastern Asia','SAsia')

    dataset['Region'] = dataset['Region'].replace('Southern Asia','SAsia')

    dataset['Region'] = dataset['Region'].replace('Central and Eastern Europe','EEur')

    dataset['Region'] = dataset['Region'].replace('Eastern Asia','EAsia')

    dataset['Region'] = dataset['Region'].replace('Sub-Saharan Africa','SAfr')

#---------------#-----------------------------#--------------------#---------------------#------

data_15['Health'].iloc[122] = 0.342008

data_16['Health'].iloc[110] = 0.3944

#---------------#----------------------------#-------------------#-----------------#-----------------#-----------

# need to convert the float values to integer, so that the probablity density could be found out.

data_15['Health'].iloc[23] = 9.0

data_15['Health'].iloc[71] = 9.0

for dataset in combine_new:

    dataset.loc[(dataset['Health'] < 0.1), 'Health'] = 0

    dataset.loc[(dataset['Health'] >= 0.1) & (dataset['Health'] < 0.2), 'Health'] = 1

    dataset.loc[(dataset['Health'] >= 0.2) & (dataset['Health'] < 0.3), 'Health'] = 2

    dataset.loc[(dataset['Health'] >= 0.3) & (dataset['Health'] < 0.4), 'Health'] = 3

    dataset.loc[(dataset['Health'] >= 0.4) & (dataset['Health'] < 0.5), 'Health'] = 4

    dataset.loc[(dataset['Health'] >= 0.5) & (dataset['Health'] < 0.6), 'Health'] = 5

    dataset.loc[(dataset['Health'] >= 0.6) & (dataset['Health'] < 0.7), 'Health'] = 6

    dataset.loc[(dataset['Health'] >= 0.7) & (dataset['Health'] < 0.8), 'Health'] = 7

    dataset.loc[(dataset['Health'] >= 0.8) & (dataset['Health'] < 0.9), 'Health'] = 8

    dataset.loc[(dataset['Health'] >= 0.9) & (dataset['Health'] < 1.0), 'Health'] = 9

    dataset['Health'] = dataset['Health'].astype(int)

#------------------#----------------------#-----------------------------#-------------------#-------

data_15['Economy'].iloc[119] = 0.0566

data_16['Economy'].iloc[75] = 0.025

#-----------------------------------------------------------------------------------------------

# giving a valid score

for dataset in combine_new:

    dataset.loc[(dataset['Score'] >= 7.5) & (dataset['Score'] < 8.5), 'Score'] = 15

    dataset.loc[(dataset['Score'] >= 7.0) & (dataset['Score'] < 7.5), 'Score'] = 14    

    dataset.loc[(dataset['Score'] >= 6.5) & (dataset['Score'] < 7.0), 'Score'] = 13

    dataset.loc[(dataset['Score'] >= 6.0) & (dataset['Score'] < 6.5), 'Score'] = 12

    dataset.loc[(dataset['Score'] >= 5.5) & (dataset['Score'] < 6.0), 'Score'] = 11

    dataset.loc[(dataset['Score'] >= 5.0) & (dataset['Score'] < 5.5), 'Score'] = 10

    dataset.loc[(dataset['Score'] >= 4.5) & (dataset['Score'] < 5.0), 'Score'] = 9

    dataset.loc[(dataset['Score'] >= 4.0) & (dataset['Score'] < 4.5), 'Score'] = 8

    dataset.loc[(dataset['Score'] >= 3.5) & (dataset['Score'] < 4.0), 'Score'] = 7

    dataset.loc[(dataset['Score'] >= 3.0) & (dataset['Score'] < 3.5), 'Score'] = 6

    dataset.loc[(dataset['Score'] >= 2.5) & (dataset['Score'] < 3.0), 'Score'] = 5

    dataset.loc[(dataset['Score'] >= 2.0) & (dataset['Score'] < 2.5), 'Score'] = 4

    dataset['Score'] = dataset['Score'].astype(int)



#----------------------------------------------------------------------------------------

# converting Economy into catagorical data

for dataset in combine_new:

    dataset.loc[(dataset['Economy'] >= 0.95), 'Economy'] = 2

    dataset.loc[(dataset['Economy'] >= 0.45) & (dataset['Economy'] < 0.95), 'Economy'] = 1

    dataset.loc[(dataset['Economy'] >= 0.00) & (dataset['Economy'] < 0.45), 'Economy'] = 0

    dataset['Economy'] = dataset['Economy'].astype(int)

#-------------------------------------------------------------------------------------------

data_16['Family'].iloc[154] = 0.431883

data_15['Family'].iloc[147] = 0.597896



data_15.head()
data_15.Family.unique()
plt.scatter(x='Health',y='Economy',data = data_15)

plt.show()
sns.boxplot(x= 'Health',y='Family',data = data_15)

plt.show()