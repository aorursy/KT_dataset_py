import numpy as np

import pandas as pd



import plotly



import warnings

warnings.filterwarnings("ignore")



from plotly import tools

import plotly.figure_factory as ff

import plotly.plotly as py

from plotly.offline import init_notebook_mode, iplot

import plotly.figure_factory as ff

init_notebook_mode(connected=True)

import plotly.graph_objs as go
train = pd.read_csv('../input/train.csv')

train_data = train.copy()
# Delete unused columns

train.drop(columns=['homepage', 'imdb_id', 'original_title', 'poster_path'],

           axis = 1 ,inplace=True)



train_data.budget = train.budget.astype(float)

train_data.revenue = train.revenue.astype(float)
null_data = train_data.isna().sum().reset_index()



x = null_data.iloc[:,0]

y = null_data.iloc[:,1]



x = x.tolist()

y = y.tolist()
trace1 = go.Bar(x=x, y=y, name = 'Age', opacity = 0.7,

                marker=dict(color='rgb(55, 83, 109)'))



data_a = [trace1]



layout = go.Layout(barmode = "group")

fig = go.Figure(data = data_a, layout = layout)

iplot(fig)
import time

import datetime



movietime = train_data.loc[:,["original_title","release_date","budget","runtime","revenue"]]

movietime.dropna()



movietime.release_date = pd.to_datetime(movietime.release_date)

movietime.loc[:,"Year"] = movietime["release_date"].dt.year

movietime.loc[:,"Month"] = movietime["release_date"].dt.month



movietime = movietime[movietime.Year<2018]

movietime.head()
titles = movietime.Year.value_counts(sort=False).reset_index()

titles.sort_values('index', inplace = True)



x = titles.iloc[:,0]

y = titles.iloc[:,1]



x = x.tolist()

y = y.tolist()
trace = go.Scatter(x = x,y =y,mode = 'lines+markers',)

data = [trace]



layout = dict(title = 'Movie release vs year',

              xaxis = dict(title = 'Year'),

              yaxis = dict(title = 'Number of Releases'),

              )



fig = dict(data=data, layout=layout)

iplot(fig)
revenues = movietime.groupby("Year")["revenue"].aggregate(["min","mean","max"]).reset_index()

budget = movietime.groupby("Year")["budget"].aggregate(["min","mean","max"]).reset_index()



year = revenues.iloc[:,0]

min = revenues.iloc[:,1]

mean = revenues.iloc[:,2]

max = revenues.iloc[:,3]





year = year.tolist()

min = min.tolist()

mean = mean.tolist()

max = max.tolist()
trace0 = go.Scatter(x = year,y = min,name = 'min',

                    line = dict(color = ('rgb(180, 180, 180)'),width = 4))



trace1 = go.Scatter(x = year,y = max,name = 'max',

                    line = dict(color = ('rgb(205, 12, 24)'),width = 4,dash = 'dash'))



trace2 = go.Scatter(x = year,y = mean,name = 'mean',

                    line = dict(color = ('rgb(22, 96, 167)'),width = 4,dash = 'dot'))



data = [trace0, trace1, trace2]



# Edit the layout

layout = dict(title = 'Average High and Low Revenues',

              xaxis = dict(title = 'Year of Release'),

              yaxis = dict(title = 'Revenue',range =[-200000000,1600000000]),

              )



fig = dict(data=data, layout=layout)

iplot(fig)
trace1 = go.Scatter(x = train["runtime"],y = train["budget"],mode = 'markers',

        marker=dict(

        line = dict(width = 0.8)))

trace2 = go.Scatter(x = train["popularity"],y = train["budget"],mode = 'markers',

        marker=dict(color = '#FFBAD2',

        line = dict(width = 0.8)))



fig = tools.make_subplots(rows=1, cols=2,subplot_titles=('Runtime vs Budget', 'Polularity vs Budget'))



fig.append_trace(trace1, 1, 1)

fig.append_trace(trace2, 1, 2)



fig['layout']['xaxis1'].update(title='Runtime')

fig['layout']['xaxis2'].update(title='Popularity')

fig['layout']['yaxis1'].update(title='Budget')

fig['layout']['yaxis2'].update(title='Budget')

fig['layout'].update(height=600, width=1200)

iplot(fig)
# from https://www.kaggle.com/artgor/eda-feature-engineering-and-model-interpretation

dict_columns = ['belongs_to_collection', 'genres', 'production_companies',

                'production_countries', 'spoken_languages', 'Keywords', 'cast', 'crew']



import ast

from collections import Counter
def text_to_dict(df):

    for column in dict_columns:

        df[column] = df[column].apply(lambda x: {} if pd.isna(x) else ast.literal_eval(x) )

    return df

        

train = text_to_dict(train)
list_of_countries = (train['production_countries'].apply(lambda x: [i['name'] for i in x] if x != {} else []).values)

list = Counter([i for j in list_of_countries for i in j]).most_common(40)
con = pd.DataFrame(list, columns = ['COUNTRY' , 'No.'])

con.rename(columns={'0':'COUNTRY'}, inplace=True)

df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/2014_world_gdp_with_codes.csv')
s1 = pd.merge(df,con, how='inner', on=['COUNTRY'])

s1.drop(['GDP (BILLIONS)'], axis = 1,inplace = True)

df2 = pd.DataFrame([['United States', 'USA',2282 ], ['South Korea','KOR',22]], columns=('COUNTRY','CODE','No.'))

s1 = s1.append(df2).reset_index()
scl = [

    [0.0, 'rgb(242,240,247)'],

    [0.2, 'rgb(218,218,235)'],

    [0.4, 'rgb(188,189,220)'],

    [0.6, 'rgb(158,154,200)'],

    [0.8, 'rgb(117,107,177)'],

    [1.0, 'rgb(84,39,143)']]



data = [go.Choropleth(locations = s1['CODE'],z = s1['No.'],text = s1['COUNTRY'],colorscale = scl,

        autocolorscale = False,

        marker = go.choropleth.Marker(line = go.choropleth.marker.Line( color = 'rgb(180,180,180)',width = 0.5)),

        colorbar = go.choropleth.ColorBar(title = 'Number of players'))]



layout= go.Layout(title = go.layout.Title(text = 'Movies from Different Countries'),

            geo = go.layout.Geo(showframe = False,showcoastlines = False,

            projection = go.layout.geo.Projection(type = 'equirectangular')))



fig = go.Figure(data = data, layout = layout)

iplot(fig, filename = 'd3-world-map')

max0 = budget.iloc[:,3]
trace1 = go.Bar(x=year,y=max0,name='Budget')

trace2 = go.Bar(x=year,y=max,name='Revenue')



data = [trace1, trace2]

layout = go.Layout(barmode='stack',title = 'Max Revenue vs Max Budget',

                  xaxis = dict(title = 'Year'),

                  yaxis = dict(title = 'Budget',range =[-200000000,1600000000]),)



fig = go.Figure(data=data, layout=layout)

iplot(fig)
list_of_languages = (train['spoken_languages'].apply(lambda x: [i['name'] for i in x] if x != {} else []).values)

new_list = Counter([i for j in list_of_languages for i in j]).most_common(15)
languages = pd.DataFrame(new_list, columns = ['Languages' , 'No.'])

x_languages = languages["No."].sum()

languages["No."] = languages["No."].div(x_languages)#.mul(100).round(2)
list_of_genres = (train['genres'].apply(lambda x: [i['name'] for i in x] if x != {} else []).values)

list = Counter([i for j in list_of_genres for i in j]).most_common(15)
genres = pd.DataFrame(list, columns = ['Genres' , 'No.'])

x_genres = genres["No."].sum()

genres["No."] = genres["No."].div(x_genres)#.mul(100).round(2)
fig = {"data":[{"values": genres["No."],

                "labels": genres["Genres"],

                "textposition":"inside",

                "domain": {"x": [.01,.49]},

                "name": "Genres",

                "hoverinfo":"label+percent+name",

                "hole": .4,

                "type": "pie" },

                

                {"values": languages["No."],

                "labels": languages["Languages"],

                "textposition":"inside",

                "domain": {"x": [.51, 1]},

                "name": "Languages",

                "hoverinfo":"label+percent+name",

                "hole": .4,

                "type": "pie"

                },

                

                ],

       

        "layout": {"title":"Most Common Genres and Languages of Movies", 

                   "annotations":[{"font": {"size": 15},"showarrow": False,

                                    "text": "Genres","x": 0.21,"y": 0.5},

       

                                   {"font": {"size": 15},"showarrow": False,

                                    "text": "Languages","x": 0.81,"y": 0.5}]}

      }



iplot(fig)