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

# Any results you write to the current directory are saved as output.

#--------------------------------------------------------------------------------------

data_15 = data_15.drop(['Standard Error'], axis =1)

data_16 = data_16.drop(['Upper Confidence Interval','Lower Confidence Interval'], axis =1)

data_17 = data_17.drop(['Whisker.high','Whisker.low'], axis =1)

combine =[data_15,data_16,data_17]

#----------------------------------------------------------------------------------

data_17 = data_17.reindex_axis(['Country', 'Happiness.Rank', 'Happiness.Score',

       'Economy..GDP.per.Capita.', 'Family', 'Health..Life.Expectancy.',

       'Freedom', 'Trust..Government.Corruption.', 'Generosity',

       'Dystopia.Residual'], axis =1)

data_17 = data_17.rename(columns={'Happiness.Rank': 'Happiness Rank', 'Happiness.Score': 'Happiness Score',

       'Economy..GDP.per.Capita.':'Economy (GDP per Capita)' , 'Health..Life.Expectancy.':'Health (Life Expectancy)' ,

       'Trust..Government.Corruption.':'Trust (Government Corruption)',

       'Dystopia.Residual':'Dystopia Residual'})

#--------------------------------------------------------------------------------------

com_new =[data_15,data_16]

for dataset in com_new:

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

#-----------------------------------------------------------------------------------------------   

data_15.head()
g = data_15[(data_15['Happiness Score'] >= 4) & (data_15['Happiness Score'] <= 5)]

g.head()
data = dict(type = 'choropleth', 

           locations = data_15['Country'],

           locationmode = 'country names',

           z = data_15['Happiness Rank'], 

           text = data_15['Country'],

           colorbar = {'title':'Happiness'})



layout = dict(title = 'Global Happiness', 

             geo = dict(showframe = False, 

                       projection = {'type': 'Mercator'}))

choromap3 = go.Figure(data = [data], layout=layout)

iplot(choromap3)
data_15.Region.unique()
sns.countplot(x = 'Region', data = g)

plt.show()