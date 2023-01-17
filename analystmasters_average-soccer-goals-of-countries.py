import pandas as pd

%matplotlib inline

import plotly.offline as py

py.init_notebook_mode(connected=True)

import os 

import sys



cwd=os.getcwd()

cwd

os.chdir('/kaggle/input/')

all_list=os.listdir()

if len(all_list)<5:

    os.chdir('/kaggle/input/earn-your-6-figure-prize-2/')
fresults6 = pd.read_csv('fresults6.csv')
country6 = pd.read_csv('country6.csv', names=['country'])
# Compute the goal of each match

df = pd.concat([fresults6, country6], axis=1)

df['goal'] = df[['4','0']].sum(axis=1)
# Compute the average goal of each country

mean_goal = df.groupby('country').agg({'goal': 'mean'})
countries = list(mean_goal.index)

data = [ dict(

type = 'choropleth',

locations = countries,

z = list(mean_goal['goal']),

locationmode = 'country names',

text = countries,

colorscale = 'Viridis',

marker = dict(

line = dict(color = 'rgb(0,0,0)', width = 1)),

colorbar = dict(autotick = True, tickprefix = '', 

title = 'Average\nGoals')

)

]



layout = dict(

    title = 'Average goals in countries',

    geo = dict(

        showframe = False,

        showocean = True,

        oceancolor = 'rgb(0,255,255)',

        projection = dict(

        type = 'orthographic',

            rotation = dict(

                    lon = 60,

                    lat = 10),

        ),

        lonaxis =  dict(

                showgrid = True,

                gridcolor = 'rgb(102, 102, 102)'

            ),

        lataxis = dict(

                showgrid = True,

                gridcolor = 'rgb(102, 102, 102)'

                )

            ),

        )



fig = dict(data=data, layout=layout)

py.iplot(fig, validate=False, filename='worldmap')