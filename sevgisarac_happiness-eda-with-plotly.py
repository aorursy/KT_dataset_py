# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# plotly

# import plotly.plotly as py

from plotly.offline import init_notebook_mode, iplot, plot

import plotly as py

init_notebook_mode(connected=True)

import plotly.graph_objs as go



import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls



import missingno as msno

# word cloud library

from wordcloud import WordCloud



# matplotlib

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# Load data that we will use.

df2015 = pd.read_csv("../input/world-happiness/2015.csv")

df2016 = pd.read_csv("../input/world-happiness/2016.csv")

df2017 = pd.read_csv("../input/world-happiness/2017.csv")

df2018 = pd.read_csv("../input/world-happiness/2018.csv")

df2019 = pd.read_csv("../input/world-happiness/2019.csv")
df2015.columns
df2015.rename(columns = {'Country':'country','Happiness Rank':'rank','Happiness Score':'score', 'Economy (GDP per Capita)':'economy','Health (Life Expectancy)':'health','Freedom':'freedom','Trust (Government Corruption)':'trust','Generosity':'generosity'},inplace =True)
df2016.rename(columns = {'Country':'country','Happiness Rank':'rank','Happiness Score':'score', 'Economy (GDP per Capita)':'economy','Health (Life Expectancy)':'health','Freedom':'freedom','Trust (Government Corruption)':'trust','Generosity':'generosity'},inplace =True)

df2017.rename(columns = {'Country':'country','Happiness.Rank':'rank','Happiness.Score':'score', 'Economy..GDP.per.Capita.':'economy','Health..Life.Expectancy.':'health','Freedom':'freedom','Trust..Government.Corruption.':'trust','Generosity':'generosity'},inplace =True)

df2018.rename(columns = {'Country or region':'country','Overall rank':'rank','Score':'score', 'GDP per capita':'economy','Healthy life expectancy':'health','Freedom to make life choices':'freedom','Perceptions of corruption':'trust','Generosity':'generosity'},inplace =True)

df2019.rename(columns = {'Country or region':'country','Overall rank':'rank','Score':'score', 'GDP per capita':'economy','Healthy life expectancy':'health','Freedom to make life choices':'freedom','Perceptions of corruption':'trust','Generosity':'generosity'},inplace =True)

#First add column each data set as a year

df2015["year"] = '2015' #this is Broadcasting, we add a column name as year and we filled the column with 2015

df2016["year"] = '2016'

df2017["year"] = '2017'

df2018["year"] = '2018'

df2019["year"] = '2019'
df2019.columns
# let concanate all data set as df

df = pd.concat([df2015,df2016,df2017,df2018,df2019],axis=0,ignore_index=True,sort=True)
df.head()
df.drop(['Dystopia Residual', 'Dystopia.Residual', 'Family'], axis=1,inplace=True)
df.drop(['Lower Confidence Interval', 'Region', 'Social support','Standard Error','Upper Confidence Interval','Whisker.high','Whisker.low'], axis=1,inplace=True)
#our main data set we will use

df.head()
msno.bar(df)

plt.show()
# prepare data frame

df2 = df.iloc[:158,:]



# import graph objects as "go"

import plotly.graph_objs as go



# Creating trace1

trace1 = go.Scatter(

                    x = df2['rank'],

                    y = df2['economy'],

                    mode = "lines",

                    name = "economy",

                    marker = dict(color = 'rgba(255, 148, 120, 1)'),

                    text= df2.country)

# Creating trace2

trace2 = go.Scatter(

                    x = df2['rank'],

                    y = df2['freedom'],

                    mode = "lines",

                    name = "freedom",

                    marker = dict(color = 'rgba(219, 10, 91, 1)'),

                    text= df2.country)

trace3 = go.Scatter(

                    x = df2['rank'],

                    y = df2['generosity'],

                    mode = "lines",

                    name = "generosity",

                    marker = dict(color = 'rgba(242, 38, 19, 1)'),

                    text= df2.country)

trace4 = go.Scatter(

                    x = df2['rank'],

                    y = df2['health'],

                    mode = "lines",

                    name = "health",

                    marker = dict(color = 'rgba(238, 238, 0, 1)'),

                    text= df2.country)

trace5 = go.Scatter(

                    x = df2['rank'],

                    y = df2['trust'],

                    mode = "lines",

                    name = "trust",

                    marker = dict(color = 'rgba(30, 130, 76, 1)'),

                    text= df2.country)

data = [trace1, trace2, trace3, trace4, trace5]

layout = dict(title = '2015 Countries World Rank vs Features',

              xaxis= dict(title= 'World Rank',ticklen= 5, zeroline= False)

             )

fig = dict(data = data, layout = layout)

iplot(fig)
import plotly.express as px





fig = px.scatter(df, x="rank", y= 'economy', color= 'year',

                 size= 'economy', hover_data=['country'])

fig.show()
import plotly.express as px





fig = px.scatter(df, x="rank", y= 'health', color= 'year',

                 size= 'health', hover_data=['country'])

fig.show()
df.head()


df_2019 = df[df.year == '2019'].iloc[:5,:]

# import graph objects as "go"

import plotly.graph_objs as go

# create trace1 

trace1 = go.Bar(

                x = df_2019['country'],

                y = df_2019['economy'],

                name = "Economy",

                marker = dict(color = 'rgba(255, 0, 0, 1)',

                             line=dict(color='rgb(0,0,0)',width=1.5)),

                text = df_2019['country'])

# create trace2 

trace3 = go.Bar(

                x = df_2019['country'],

                y = df_2019['freedom'],

                name = "Freedom",

                marker = dict(color = 'rgba(255, 0, 0, 0.6)',

                              line=dict(color='rgb(0,0,0)',width=1.5)),

                text = df_2019['country'])

trace5 = go.Bar(

                x = df_2019['country'],

                y = df_2019['generosity'],

                name = "Generosity",

                marker = dict(color = 'rgba(255, 0, 0, 0.2)',

                              line=dict(color='rgb(0,0,0)',width=1.5)),

                text = df_2019['country'])

trace2 = go.Bar(

                x = df_2019['country'],

                y = df_2019['health'],

                name = "Health",

                marker = dict(color = 'rgba(255, 0, 0, 0.8)',

                              line=dict(color='rgb(0,0,0)',width=1.5)),

                text = df_2019['country'])

trace4 = go.Bar(

                x = df_2019['country'],

                y = df_2019['trust'],

                name = "Trust",

                marker = dict(color = 'rgba(255, 0, 0, 0.4)',

                              line=dict(color='rgb(0,0,0)',width=1.5)),

                text = df_2019['country'])

data = [trace1, trace2, trace3, trace4, trace5]

layout = go.Layout(barmode = "group")

layout = dict(title = 'Five Factors in 2019 for first five countries',

              xaxis= dict(title= 'Countries',ticklen= 5,zeroline= False),

             )

fig = go.Figure(data = data, layout = layout)

iplot(fig)
# prepare data frames

df_2019 = df[df.year == '2019'].iloc[:10,:]

# import graph objects as "go"

import plotly.graph_objs as go



x = df_2019.country



trace1 = {

  'x': x,

  'y': df_2019['trust'],

  'name': 'trust',

  'type': 'bar'

};

trace2 = {

  'x': x,

  'y': df_2019['freedom'],

  'name': 'freedom',

  'type': 'bar'

};

data = [trace1, trace2];

layout = {

  'xaxis': {'title': '2019 Top 10 countries'},

  'barmode': 'relative',

  'title': 'Freedom and Trust of top 10 countries in 2019'

};

fig = go.Figure(data = data, layout = layout)

iplot(fig)
# import graph objects as "go" and import tools

import plotly.graph_objs as go

from plotly import tools

import matplotlib.pyplot as plt

# prepare data frames

df_2016 = df[df.year == '2016'].iloc[147:160,:]



y_saving = [each for each in df_2016.trust]

y_net_worth  = [float(each) for each in df_2016.economy]

x_saving = [each for each in df_2016.country]

x_net_worth  = [each for each in df_2016.country]

trace0 = go.Bar(

                x=y_saving,

                y=x_saving,

                marker=dict(color='rgba(171, 50, 96, 0.6)',line=dict(color='rgba(171, 50, 96, 1.0)',width=1)),

                name='trust',

                orientation='h',

)

trace1 = go.Scatter(

                x=y_net_worth,

                y=x_net_worth,

                mode='lines+markers',

                line=dict(color='rgb(63, 72, 204)'),

                name='economy',

)

layout = dict(

                title='Trust and Economy 2016 for last ten countries',

                yaxis=dict(showticklabels=True,domain=[0, 0.85]),

                yaxis2=dict(showline=True,showticklabels=False,linecolor='rgba(102, 102, 102, 0.8)',linewidth=2,domain=[0, 0.85]),

                xaxis=dict(zeroline=False,showline=False,showticklabels=True,showgrid=True,domain=[0, 0.42]),

                xaxis2=dict(zeroline=False,showline=False,showticklabels=True,showgrid=True,domain=[0.47, 1],side='top',dtick=25),

                legend=dict(x=0.029,y=1.038,font=dict(size=10) ),

                margin=dict(l=200, r=20,t=70,b=70),

                paper_bgcolor='rgb(248, 248, 255)',

                plot_bgcolor='rgb(248, 248, 255)',

)

annotations = []

y_s = np.round(y_saving, decimals=2)

y_nw = np.rint(y_net_worth)

# Adding labels

for ydn, yd, xd in zip(y_nw, y_s, x_saving):

    # labeling the scatter savings

    annotations.append(dict(xref='x2', yref='y2', y=xd, x=ydn - 4,text='{:,}'.format(ydn),font=dict(family='Arial', size=12,color='rgb(63, 72, 204)'),showarrow=False))

    # labeling the bar net worth

    annotations.append(dict(xref='x1', yref='y1', y=xd, x=yd + 3,text=str(yd),font=dict(family='Arial', size=12,color='rgb(171, 50, 96)'),showarrow=False))



layout['annotations'] = annotations



# Creating two subplots

fig = tools.make_subplots(rows=1, cols=2, specs=[[{}, {}]], shared_xaxes=True,

                          shared_yaxes=False, vertical_spacing=0.001)



fig.append_trace(trace0, 1, 1)

fig.append_trace(trace1, 1, 2)



fig['layout'].update(layout)

iplot(fig)
# data preparation

df_2017 = df[df.year == '2017'].iloc[:7,:]

pie1 = df_2017.score



labels = df_2017.country

# figure

fig = {

  "data": [

    {

      "values": pie1,

      "labels": labels,

      "domain": {"x": [0, .5]},

      "name": "Score",

      "hoverinfo":"label+percent+name",

      "hole": .5,

      "type": "pie"

    },],

  "layout": {

        "title":"2017 first 7 Countries Happiness Scores",

        "annotations": [

            { "font": { "size": 20},

              "showarrow": False,

              "text":'',

                "x": 0.20,

                "y": 1

            },

        ]

    }

}

iplot(fig)
df.head()
import plotly.express as px



df_2019 = df[df.year == '2019']

fig = px.scatter(df_2019, x="economy", y="health",

	         size="trust", color="rank",

                 hover_name="country", log_x=True, size_max=60)

fig.show()
# prepare data

x2015 = df.economy[df.year == '2015']

x2016 = df.economy[df.year == '2016']

x2017 = df.economy[df.year == '2017']

x2018 = df.economy[df.year == '2018']

x2019 = df.economy[df.year == '2019']



trace1 = go.Histogram(

    x=x2015,

    opacity=0.75,

    name = "2015",

    marker=dict(color='rgba(255, 0, 0, 0.2)'))

trace2 = go.Histogram(

    x=x2016,

    opacity=0.75,

    name = "2016",

    marker=dict(color='rgba(255, 0, 0, 0.4)'))

trace3 = go.Histogram(

    x=x2017,

    opacity=0.75,

    name = "2017",

    marker=dict(color='rgba(255, 0, 0, 0.6)'))

trace4 = go.Histogram(

    x=x2018,

    opacity=0.75,

    name = "2018",

    marker=dict(color='rgba(255, 0, 0, 0.8)'))

trace5 = go.Histogram(

    x=x2019,

    opacity=0.75,

    name = "2019",

    marker=dict(color='rgba(255, 0, 0, 1.0)'))



data = [trace1, trace2, trace3, trace4, trace5]

layout = go.Layout(barmode='overlay',

                   title='Economy in 2015 to 2019',

                   xaxis=dict(title='economy'),

                   yaxis=dict( title='Count'),

)

fig = go.Figure(data=data, layout=layout)

iplot(fig)
# data prepararion

x2015 = df.country[df.year == '2015']

plt.subplots(figsize=(10,10))

wordcloud = WordCloud(

                          background_color='white',

                          width=512,

                          height=384

                         ).generate(" ".join(x2015))

plt.imshow(wordcloud)

plt.axis('off')

plt.savefig('graph.png')



plt.show()
df.head()
# data preparation

x2019 = df[df.year == '2019']





trace1 = go.Box(

    y=x2019.economy,

    name = 'economy scores of countries in 2019',

    marker = dict(

        color = 'rgba(255, 0, 0, 0.2)',

    )

)

trace2 = go.Box(

    y=x2019.health,

    name = 'health scores of countries in 2019',

    marker = dict(

        color = 'rgba(255, 0, 0, 0.4)',

    )

)

trace3 = go.Box(

    y=x2019.trust,

    name = 'trust scores of countries in 2019',

    marker = dict(

        color = 'rgba(255, 0, 0, 0.6)',

    )

)

trace4 = go.Box(

    y=x2019.freedom,

    name = 'freedom scores of countries in 2019',

    marker = dict(

        color = 'rgba(255, 0, 0, 0.8)',

    )

)

trace5 = go.Box(

    y=x2019.generosity,

    name = 'generosity scores of countries in 2015',

    marker = dict(

        color = 'rgba(255, 0, 0, 1.0)',

    )

)

data = [trace1, trace2, trace3, trace4, trace5]

iplot(data)
df.head()
# import figure factory

import plotly.figure_factory as ff

# prepare data

dataframe = df[df.year == '2015']

data2015 = dataframe.loc[:,["economy",'freedom', "generosity", 'health', "score", 'trust']]

data2015["index"] = np.arange(1,len(data2015)+1)

# scatter matrix

fig = ff.create_scatterplotmatrix(data2015, diag='box', index='index',colormap='Portland',

                                  colormap_type='cat',

                                  height=700, width=700)

iplot(fig)


# first line plot

trace1 = go.Scatter(

    x=df2019['rank'],

    y=df2019['health'],

    name = "health",

    marker = dict(color = 'rgba(16, 112, 2, 0.8)'),

)

# second line plot

trace2 = go.Scatter(

    x=df2019['rank'],

    y=df2019['economy'],

    xaxis='x2',

    yaxis='y2',

    name = "economy",

    marker = dict(color = 'rgba(160, 112, 20, 0.8)'),

)

data = [trace1, trace2]

layout = go.Layout(

    xaxis2=dict(

        domain=[0.6, 0.95],

        anchor='y2',        

    ),

    yaxis2=dict(

        domain=[0.6, 0.95],

        anchor='x2',

    ),

    title = '2019 Economy and Health vs World Rank of Countries'



)



fig = go.Figure(data=data, layout=layout)

iplot(fig)
# create trace 1 that is 3d scatter

trace1 = go.Scatter3d(

    x=df2019['rank'],

    y=df2019['economy'],

    z=df2019['health'],

    mode='markers',

    marker=dict(

        size=df2019['trust']*100,

        color=df2019['score'],                # set color to an array/list of desired values      

    )

)



data = [trace1]

layout = go.Layout(

    margin=dict(

        l=0,

        r=0,

        b=0,

        t=0  

    )

    

)

fig = go.Figure(data=data, layout=layout)

iplot(fig)
trace1 = go.Scatter(

    x=df2018['rank'],

    y=df2018['freedom'],

    name = "freedom"

)

trace2 = go.Scatter(

    x=df2018['rank'],

    y=df2018['generosity'],

    xaxis='x2',

    yaxis='y2',

    name = "generosity"

)

trace3 = go.Scatter(

   x=df2018['rank'],

    y=df2018['health'],

    xaxis='x3',

    yaxis='y3',

    name = "health"

)

trace4 = go.Scatter(

    x=df2018['rank'],

    y=df2018['score'],

    xaxis='x4',

    yaxis='y4',

    name = "total_score"

)

data = [trace1, trace2, trace3, trace4]

layout = go.Layout(

    xaxis=dict(

        domain=[0, 0.45]

    ),

    yaxis=dict(

        domain=[0, 0.45]

    ),

    xaxis2=dict(

        domain=[0.55, 1]

    ),

    xaxis3=dict(

        domain=[0, 0.45],

        anchor='y3'

    ),

    xaxis4=dict(

        domain=[0.55, 1],

        anchor='y4'

    ),

    yaxis2=dict(

        domain=[0, 0.45],

        anchor='x2'

    ),

    yaxis3=dict(

        domain=[0.55, 1]

    ),

    yaxis4=dict(

        domain=[0.55, 1],

        anchor='x4'

    ),

    title = '2018 Freedom, generosity, health and total score VS World Rank of Countries'

)

fig = go.Figure(data=data, layout=layout)

iplot(fig)


import plotly.express as px





#Find out more at https://plot.ly/python/choropleth-maps/

data = [ dict(

        type = 'choropleth',

        locations = df2015['country'],

        locationmode = 'country names',

        z = df2015['rank'],

        text = df2015['country'],

        

        color_continuous_scale=px.colors.sequential.Viridis, 

        autocolorscale = False,

        reversescale = True,

        marker = dict(

            line = dict (

                color = 'rgb(180,180,180)',

                width = 0.5

            ) ),

        colorbar = dict(

            autotick = False,

            tickprefix = '',

            title = 'Happiness Rank 2015'),

      ) ]



layout = dict(

    title = 'Happiness Rank 2015 by Country',

    geo = dict(

        showframe = False,

        showcoastlines = True,

        projection = dict(

            type = 'Mercator'

        )

    )

)



fig = dict( data=data, layout=layout )

py.iplot( fig, validate=False, filename='happinessrank-world-map')
fig = px.choropleth(df, locations="country", locationmode='country names', color="rank", 

                    hover_name="country", animation_frame=df["year"],

                    title='Cases over time',  color_continuous_scale="Viridis")

fig.update_geos(fitbounds="locations", visible=False)

fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

fig.show()
import plotly.express as px



px.scatter(df, x="rank", y="score", animation_frame="country", animation_group="year",

           size="score", color="year", hover_name="country", range_x=[1,160], range_y=[0,10])