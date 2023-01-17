print('Today, we wiil explore in which country the most goals was scored, and explore the productivity of the football season over the past 45 years')

print('\n')

print('Our plan for today'+ '\n'+'1) find out how the number of goals scored in each football season has changed' +'\n'+'2) in which country scored the largest number of goals' + '\n' +'3) to explore the tendency of friendly matches')
## import all the necessary libraries



import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import csv

import pandas as pd

import  numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

filepath = '../input/results.csv'

data = pd.read_csv(filepath)



years = data['date'].map(lambda y: int(str(y)[:4]))  ## only year without month and days (type integer)

friendly_match = data['tournament'].map(lambda y: 1 if y=='Friendly' else 0 )

YEARS = list(np.unique(years))                       ## unique years

COUNTRY = list(np.unique(data['country']))           ## unique country



goalAll = pd.DataFrame(data['home_ft']+data['away_ft'])  ## how many goals were scored in one match by both teams

goalAll.insert(1, 'Date', years)                         ## creates a new data table (new DataFrame) (add column with year when this match was played)

goalAll.insert(2, 'Country', data['country'])            ## add column with name of country in which was a match

goalAll.insert(3, 'Tournament', data['tournament'])      ## type of match

goalAll.insert(4, 'Friendly', friendly_match)

goalAll.columns = ['Goals','Year','Country', 'Tournament', 'Friendly']             ## sets the name of the column







## now our new data table looks like this:

goalAll.head()
## now we will find how many goals were scored every year, regardless of team, country and match type



sum_goals = []

for year in YEARS:

    sum_goals.append(goalAll[goalAll['Year']==year]['Goals'].sum())


goalPerYear = go.Scatter(x=YEARS, y=sum_goals, name='goalsPerYears', type='bar')

data = go.Data([goalPerYear])

layout = go.Layout(title='Goals Per years', xaxis={'title':'Years'},yaxis={'title':'Number of goals'})

figure = go.Figure(data=data, layout=layout)



py.iplot(figure, filename='goalPerYears', image='jpeg')


sum_goals = []

for country in COUNTRY:

    sum_goals.append(goalAll[goalAll['Country']==country]['Goals'].sum())



goalPerCountry = [dict(

        colorscale=[

        [

          0,

          "rgb(255,255,255)"

        ],

        [

          0.1,

          "rgb(255,255,220)"

        ],

        [

          0.2,

          "rgb(255,255,200)"

        ],

        [

          0.3,

          "rgb(255,255,170)"

        ],

        [

          0.4,

          "rgb(255,255,120)"

        ],

        [

          0.5,

          "rgb(255,255,0)"

        ],

        [

          0.6,

          "rgb(200,255,0)"

        ],

        [

          0.7,

          "rgb(150,255,0)"

        ],

        [

          0.8,

          "rgb(0,255,0)"

        ],

        [

          0.9,

          "rgb(0,200,0)"

        ],

        [

          1,

          "rgb(0,129,0)"

        ]

      ],

        type = 'choropleth',

        locations = COUNTRY,

        z = sum_goals,

        locationmode = 'country names',

        text = COUNTRY,

        marker = dict(

            line = dict(color = 'rgb(0,0,0)', width = 1)),

            colorbar = dict(autotick = True, tickprefix = '',

            title = '# Number of goals for 45 years \n')

            )

       ]



layout = dict(

    title = 'Number goals scored in each country \n from 1972 to 2017',

    geo = dict(

        showframe = False,

        showocean = False,

        

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



figure = dict(data=goalPerCountry, layout=layout)

py.iplot(figure, validate=False, filename='worldmap', image='jpeg')



### sorting country by nubmer scored goals



sum_goals_bar, COUNTRY_bar = (list(x) for x in zip(*sorted(zip(sum_goals, COUNTRY), reverse=True)))

sns.set(font_scale=0.7)

f, ax = plt.subplots(figsize=(5,60))

color_sw = sns.color_palette('coolwarm', len(COUNTRY))

sns.barplot(sum_goals_bar, COUNTRY_bar, palette=color_sw[::-1])

Text = ax.set(xlabel='Number of goals', title='Country where goals was scored ( 1972 to 2017)')
## statistics of friendly matches



sum_friendly = []

for year in YEARS:

    sum_friendly.append(goalAll[goalAll['Year']==year]['Friendly'].sum())



data = go.Scatter(x = YEARS, y = sum_friendly,mode = 'lines',name = 'Number of friendly matches')  

layout = go.Layout(title='Friendly matches per year', xaxis={'title':'Years'},yaxis={'title':'Number of friendly matches'})

data = go.Data([data])      

figure = go.Figure(data=data, layout=layout)

py.iplot(figure)
