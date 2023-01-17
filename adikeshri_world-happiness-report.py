# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data=pd.read_csv('../input/world-happiness-report-2019.csv')

data.sample(10)
import matplotlib.pyplot as plt

import seaborn as sns

sns.set()

fig=plt.gcf()

fig.set_size_inches(18.5,10)

plt.scatter(data['Freedom'],

            data['Positive affect'],

            alpha=0.5,color='green',s=40)

plt.scatter(data['Freedom'],

            data['Negative affect'],

            alpha=0.5,color='maroon',s=40)

plt.scatter(data['Freedom'],

            data['Corruption'],

            alpha=0.5,color='blue',s=40)

plt.legend(loc='best')

m,c=np.polyfit(data['Freedom'].fillna(data['Freedom'].median()),

               data['Positive affect'].fillna(data['Positive affect'].median()),deg=1)

plt.plot(data['Freedom'],

         m*data['Freedom']+c,

         color='green')

m,c=np.polyfit(data['Freedom'].fillna(data['Freedom'].median()),

               data['Negative affect'].fillna(data['Negative affect'].median()),deg=1)

plt.plot(data['Freedom'],

         m*data['Freedom']+c,

         color='maroon')

m,c=np.polyfit(data['Freedom'].fillna(data['Freedom'].median()),

               data['Corruption'].fillna(data['Corruption'].median()),deg=1)

plt.plot(data['Freedom'],

         m*data['Freedom']+c,

         color='blue')

plt.xlabel('Freedom')

plt.title('Postive affect, Negative affect & Corruption vs Freedom')

plt.show()

data.columns

import plotly.plotly as py

import plotly.graph_objs as go

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=True)

d=[go.Choropleth(colorscale='YlOrRd',

                 locations=data['Country (region)'],

                 locationmode='country names',

                 z=data['Ladder'],

                 colorbar={'title':'World Happiness Rank'})]

l=dict(title='World Happiness Ranking',

      geo=dict(showframe=False,

              projection={'type':'natural earth'}))

map=go.Figure(d,l)

iplot(map)

d=[go.Choropleth(colorscale='YlOrRd',

                 locations=data['Country (region)'],

                 locationmode='country names',

                 z=data['Healthy life\nexpectancy'],

                 colorbar={'title':'Healthy life expectancy ranking'})]

l=dict(title='Healthy life expectancy ranking',

      geo=dict(showframe=False,

              projection={'type':'natural earth'}))

map=go.Figure(d,l)

iplot(map)
data=data.dropna()
from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(data.drop(['Country (region)','Ladder'],axis=1),data['Ladder'],test_size=0.2)

model=LinearRegression()

model.fit(x_train,y_train)

print(model.score(x_test,y_test))
