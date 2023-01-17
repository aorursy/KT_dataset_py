# Data processing

import pandas as pd

import numpy as np

import json

from collections import Counter

from itertools import chain

from sklearn.feature_extraction.text import TfidfVectorizer

import re



# Data vizualizations

import random

import plotly

from plotly import tools

from plotly.offline import download_plotlyjs,init_notebook_mode,plot,iplot

init_notebook_mode(connected=True)

import plotly.offline as offline

import plotly.graph_objs as go



# Data modeling



from sklearn.feature_extraction.text import CountVectorizer

from sklearn.preprocessing import LabelEncoder

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.ensemble import VotingClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import KFold

from sklearn import model_selection

import warnings

warnings.filterwarnings('ignore')
# Import json files with train and test samples
train_data = pd.read_json('../input/whats-cooking-kernels-only/train.json')

test_data = pd.read_json('../input/whats-cooking-kernels-only/test.json')
# look training data



train_data.info()
train_data.shape # 39774 , 3 columns
print('The training data consist of {} recipes'.format((len(train_data))))
print('First five elements in out training sample :')

train_data.head()
# Take a quick look on test sample also:



test_data.info()
test_data.shape # 9944 observations , 2 columns
print('The test data conssit of {} recipes'.format(len(test_data)))
print('First five elements in out test samples:')

test_data.head()
print('Number of cuisine categories: {}'.format(len(train_data.cuisine.unique())))

train_data.cuisine.unique()
def random_colours(number_of_colors):

    

    """

    Simple function for random colours generation.

    Input:

         number_of_colors - integer value indicating

         the number of colours which are going to be generated

    

    Output:

          Color in the following format :['#E86DA4']

    """

    

    colors = []

    for i in range(number_of_colors):

        colors.append('#'+''.join([random.choice('0123456789ABCDEF') for j in range(6)]))

        return colors
trace = go.Table(

                header=dict(values=['Cuisine','Number of recipes'],

                fill = dict(color=['#EABEB0']), 

                align = ['left'] * 5),

                cells=dict(values=[train_data.cuisine.value_counts().index,train_data.cuisine.value_counts()],

               align = ['left'] * 5))



layout = go.Layout(title='Number of recipes in each cuisine category',

                   titlefont = dict(size = 20),

                   width=500, height=650, 

                   paper_bgcolor =  'rgba(0,0,0,0)',

                   plot_bgcolor = 'rgba(0,0,0,0)',

                   autosize = False,

                   margin=dict(l=30,r=30,b=1,t=50,pad=1),

                   )

data = [trace]

fig = dict(data=data, layout=layout)

iplot(fig)

# Create also plot with label distribution



# Label distribution in percents

labelpercents = []

for i in train_data.cuisine.value_counts():

    percent = (i / sum(train_data.cuisine.value_counts())) * 100

    percent = '%.2f' % percent

    percent = str(percent + '%')

    labelpercents.append(percent)
trace = go.Bar(

            x=train_data.cuisine.value_counts().values[::-1],

            y= [i for i in train_data.cuisine.value_counts().index][::-1],

            text =labelpercents[::-1],  textposition = 'outside', 

            orientation = 'h',marker = dict(color = random_colours(20)))

layout = go.Layout(title='Number of recipes in each cuisine category',

                   titlefont = dict(size = 25),

                   width=1000, height=450, 

                   plot_bgcolor = 'rgba(0,0,0,0)',

                   paper_bgcolor = 'rgba(255, 219, 227, 0.88)',

                   margin=dict(l=75,r=110,b=50,t=60),

                   )

data = [trace]

fig = dict(data=data, layout=layout)

iplot(fig, filename='horizontal-bar')
# Lets take a closer look at the ingredients in our training smaple

print('Maximum number of ingredients in a dish : ',train_data['ingredients'].str.len().max())

print('minimum number of ingredients in a dish : ',train_data['ingredients'].str.len().min())
trace = go.Histogram(

    x= train_data['ingredients'].str.len(),

    xbins=dict(start=0,end=90,size=1),

   marker=dict(color='#7CFDF0'),

    opacity=0.75)

data = [trace]

layout = go.Layout(

    title='Distribution of Recipe Length',

    xaxis=dict(title='Number of ingredients'),

    yaxis=dict(title='Count of recipes'),

    bargap=0.1,

    bargroupgap=0.2)



fig = go.Figure(data=data, layout=layout)

iplot(fig)
longrecipes = train_data[train_data['ingredients'].str.len() > 30]

print('It seems That {} recipes consist of more than 30 ingredients!'.format(len(longrecipes)))

print('Explore the ingredients in the longest recipe in out training set :' + '\n')

print(str(list(longrecipes[longrecipes['ingredients'].str.len() == 65].ingredients.values)) + '\n')

print('Cuisine : ' + str(list(longrecipes[longrecipes['ingredients'].str.len() == 65].cuisine)))

shortrecipes = train_data[train_data['ingredients'].str.len() <= 2]

print('it seems that {} recipes consist of less than or equal to 2 ingradients'.format(len(shortrecipes)))
print('Explore the ingredients in the shortest recipes in out trainig set:' + '\n')

print(list(train_data[train_data['ingredients'].str.len() == 1].ingredients.values))



print('and there corresponding labels' + '\n')

print(list(train_data[train_data['ingredients'].str.len() == 1].cuisine.values))