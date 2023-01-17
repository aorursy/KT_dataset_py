# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls

import warnings

from collections import Counter

from sklearn.feature_selection import mutual_info_classif

warnings.filterwarnings('ignore')



train = pd.read_csv('../input/train.csv')

train.head(5)
# Data Quality Check

import missingno as msno

train_copy = train

msno.matrix(df=train_copy, figsize=(20, 14), color=(0.42, 0.1, 0.05))
train.info()
cols = train_copy.columns



feature_numbers = []



for col in cols:

    feature_numbers.append(len(train_copy[col].unique()))

    



x, y = (list(x) for x in zip(*sorted(zip(feature_numbers , cols),reverse = True)))



trace2 = go.Bar(

    x=x ,

    y=y,

    marker=dict(

        color=x,

        colorscale = 'Viridis',

        reversescale = True

    ),

    name='Feature unique values',

    orientation='h',

)



layout = dict(

    title='Barplot of Feature Unique Values',

    yaxis=dict(

        showgrid=False,

        showline=False,

        showticklabels=True,

#         domain=[0, 0.85],

    ))



fig1 = go.Figure(data=[trace2])

fig1['layout'].update(layout)

py.iplot(fig1, filename='plots')
print('Sex')

print(train_copy['Sex'].unique())

print('Pclass')

print(train_copy['Pclass'].unique())

print('Embarked')

print(train_copy['Embarked'].unique())

print('Parch')

print(train_copy['Parch'].unique())

print('SibSp')

print(train_copy['SibSp'].unique())
# Find Correlation

colormap = plt.cm.afmhot

plt.figure(figsize=(16,12))

plt.title('Correspond of Features')

sns.heatmap(train_copy.corr(),linewidths=0.1,vmax=1.0, square=True, 

            cmap=colormap, linecolor='white', annot=True)
from sklearn import preprocessing



train_copy = train.copy()



train_copy['Cabin'] = train_copy['Cabin'].fillna('-1')



train_copy['Embarked'] = train_copy['Embarked'].fillna('-1')

cols = train_copy.columns



le = preprocessing.LabelEncoder()



for col in cols:

    train_copy[col] = le.fit_transform(train_copy[col])

    

train_copy.head(5)
from sklearn.ensemble import RandomForestClassifier



clf = RandomForestClassifier()



X = train_copy.drop(['PassengerId','Survived'], axis=1)

y = train_copy['Survived']



features = X.columns



clf.fit(X, y)



clf.feature_importances_
# Scatter plot 

trace = go.Scatter(

    y = clf.feature_importances_,

    x = features,

    mode='markers',

    marker=dict(

        sizemode = 'diameter',

        sizeref = 1,

        size = 13,

        color = clf.feature_importances_,

        colorscale='Portland',

        showscale=True

    ),

    text = features

)

data = [trace]



layout= go.Layout(

    autosize= True,

    title= 'Feature Importance',

    hovermode= 'closest',

     xaxis= dict(

         ticklen= 5,

         showgrid=False,

        zeroline=False,

        showline=False

     ),

    yaxis=dict(

        title= 'Feature Importance',

        showgrid=False,

        zeroline=False,

        ticklen= 5,

        gridwidth= 2

    ),

    showlegend= False

)

fig = go.Figure(data=data, layout=layout)

py.iplot(fig,filename='scatter2010')
data = [go.Bar(

            x= train['Survived'].value_counts().index.values,

            y= train['Survived'].value_counts().values,

    )]



layout = go.Layout(

        title="Number of Survived")



fig = go.Figure(data=data, layout=layout)



py.iplot(fig, filename='Survived')
# Survived Rate Check

trace1 = go.Bar(

    x= train[train.Survived == 1]['Sex'].value_counts().index.values,

    y= train[train.Survived == 1]['Sex'].value_counts().values,

    name='Survived'

)



trace2 = go.Bar(

    x= train[train.Survived == 0]['Sex'].value_counts().index.values,

    y= train[train.Survived == 0]['Sex'].value_counts().values,

    name='Unsurvived'

)



data = [trace1, trace2]



layout = go.Layout(

    barmode='stack'

)



fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename='stacked-bar')

# Survived Rate Check

trace1 = go.Bar(

    x= train[train.Survived == 1]['Pclass'].value_counts().index.values,

    y= train[train.Survived == 1]['Pclass'].value_counts().values,

    name='Survived'

)



trace2 = go.Bar(

    x= train[train.Survived == 0]['Pclass'].value_counts().index.values,

    y= train[train.Survived == 0]['Pclass'].value_counts().values,

    name='Unsurvived'

)



data = [trace1, trace2]



layout = go.Layout(

    barmode='stack'

)



fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename='stacked-bar')

# Survived Rate Check

trace1 = go.Bar(

    x= train[train.Survived == 1]['Embarked'].value_counts().index.values,

    y= train[train.Survived == 1]['Embarked'].value_counts().values,

    name='Survived'

)



trace2 = go.Bar(

    x= train[train.Survived == 0]['Embarked'].value_counts().index.values,

    y= train[train.Survived == 0]['Embarked'].value_counts().values,

    name='Unsurvived'

)



data = [trace1, trace2]



layout = go.Layout(

    barmode='stack'

)



fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename='stacked-bar')
