# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# plotly

import plotly.plotly as py

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)

import plotly.graph_objs as go

# matplotlib

import matplotlib.pyplot as plt



# word cloud library

from wordcloud import WordCloud



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
humfree= pd.read_csv('../input/hfi_cc_2018.csv')
humfree.head()
humfree.info()
#preparing Data

df2016 = humfree[humfree.year == 2016].iloc[:,:]

df2015 = humfree[humfree.year == 2015].iloc[:,:]

df2014 = humfree[humfree.year == 2014].iloc[:,:]

new_index = (df2016['hf_rank'].sort_values(ascending=True)).index.values

df2016 = df2016.reindex(new_index) # with this code we sort our data according to human freedom rank

# import graph objects as "go"

import plotly.graph_objs as go

# creating trace1

trace1 =go.Scatter(

                    x = df2014.hf_rank,

                    y = df2014.hf_score,

                    mode = "markers",

                    name = "2014",

                    marker = dict(color = 'rgba(255, 128, 255, 0.8)'),

                    text= df2014.countries)

# creating trace2

trace2 =go.Scatter(

                    x = df2015.hf_rank,

                    y = df2015.hf_score,

                    mode = "markers",

                    name = "2015",

                    marker = dict(color = 'rgba(255, 128, 2, 0.8)'),

                    text= df2015.countries)

# creating trace3

trace3 =go.Scatter(

                    x = df2016.hf_rank,

                    y = df2016.hf_score,

                    mode = "lines",

                    name = "2016",

                    marker = dict(color = 'rgba(0, 255, 200, 0.8)'),

                    text= df2016.countries)

data = [trace1, trace2, trace3]

layout = dict(title = 'Human Freedom score vs Human Freedom rank of Countries ',

              xaxis= dict(title= 'Freedom Rank',ticklen= 5,zeroline= False),

              yaxis= dict(title= 'Freedom Score',ticklen= 5,zeroline= False)

             )

fig = dict(data = data, layout = layout)

iplot(fig)
pd.set_option("display.max_columns",123) # with this code we can see all the columns
#preparing data

df2016 = humfree[humfree.year == 2016].iloc[:,:]

df2015 = humfree[humfree.year == 2015].iloc[:,:]

df2014 = humfree[humfree.year == 2014].iloc[:,:]

new_index = (df2016['hf_rank'].sort_values(ascending=True)).index.values

df2016 = df2016.reindex(new_index)

# import graph objects as "go"

import plotly.graph_objs as go

# creating trace1

trace1 =go.Scatter(

                    x = df2014.hf_rank,

                    y = df2014.pf_rol,

                    mode = "markers",

                    name = "2014",

                    marker = dict(color = 'rgba(55, 157, 94, 0.8)'),

                    text= df2014.countries)

# creating trace2

trace2 =go.Scatter(

                    x = df2015.hf_rank,

                    y = df2015.pf_rol,

                    mode = "markers",

                    name = "2015",

                    marker = dict(color = 'rgba(255, 18, 03, 0.8)'),

                    text= df2015.countries)

# creating trace3

trace3 =go.Scatter(

                    x = df2016.hf_rank,

                    y = df2016.pf_rol,

                    mode = "markers", # as you realised Scatter and Line charts are almost the same plot

                    name = "2016",     # we change only 'MODE' it is only 3 possibility 'marker, line or line + markers'

                    marker = dict(color = 'rgba(230, 25, 200, 0.8)'),

                    text= df2016.countries)

data = [trace1, trace2, trace3]

layout = dict(title = 'Rule of law vs Human Freedom rank of Countries ',

              xaxis= dict(title= 'Freedom Rank',ticklen= 5,zeroline= False),

              yaxis= dict(title= 'Freedom Score',ticklen= 5,zeroline= False)

             )

fig = dict(data = data, layout = layout)

iplot(fig)
# prepare data frames

df2016 = humfree[humfree.year == 2016].iloc[:10,:]





# import graph objects as "go"

import plotly.graph_objs as go

# create trace1 

trace1 = go.Bar(

                x = df2016.countries,

                y = df2016.ef_government_consumption,

                name = "Government consumption",

                marker = dict(color = 'rgba(55, 114, 55, 0.8)',# It takes RGB "0-255" for all values for opacity "0-1"

                             line=dict(color='rgb(0,0,0)',width=1.5)),

                text = df2016.countries)

# create trace2 

trace2 = go.Bar(

                x = df2016.countries,

                y = df2016.ef_government_enterprises,

                name = "Government enterprises and investments",

                marker = dict(color = 'rgba(235, 155, 12, 0.9)',

                              line=dict(color='rgb(0,0,0)',width=1.5)),

                text = df2016.countries)

data = [trace1, trace2]

layout = go.Layout(barmode = "group")

fig = go.Figure(data = data, layout = layout)

iplot(fig)
# prepare data frames

df2008 = humfree[humfree.year == 2008].iloc[:15,:]

# import graph objects as "go"

import plotly.graph_objs as go



x = df2008.countries



trace1 = {

  'x': x,

  'y': df2008.pf_religion_harassment,

  'name': 'Harassment and physical hostilities',

  'type': 'bar'

};

trace2 = {

  'x': x,

  'y': df2008.pf_religion_restrictions,

  'name': 'Legal and regulatory restrictions',

  'type': 'bar'

};

data = [trace1, trace2];

layout = {

  'xaxis': {'title': ' Countries at 2008'},

  'barmode': 'relative',

  'title': 'Religion Restriction and Harassment'

};

fig = go.Figure(data = data, layout = layout)

iplot(fig)
# prepare data

x2011 = humfree.pf_rol_criminal[humfree.year == 2011]

x2012 = humfree.pf_rol_criminal[humfree.year == 2012]



trace1 = go.Histogram(

    x=x2011,

    opacity=0.75,

    name = "2011",

    marker=dict(color='rgba(191, 200, 06, 0.6)'))

trace2 = go.Histogram(

    x=x2012,

    opacity=0.75,

    name = "2012",

    marker=dict(color='rgba(62, 50, 146, 0.6)'))



data = [trace1, trace2]

layout = go.Layout(barmode='overlay',

                   title=' Countries-Criminal justice in 2011 and 2012',

                   xaxis=dict(title='Criminal justice'),

                   yaxis=dict( title='Count'),

)

fig = go.Figure(data=data, layout=layout)

iplot(fig)
#preparing Data

df2015 = humfree[humfree.year == 2015]



# import figure factory

import plotly.figure_factory as ff

df = df2015.loc[:,["pf_rol_criminal","pf_ss_homicide", "pf_ss"]]

df["index"] = np.arange(1,len(df)+1)

fig = ff.create_scatterplotmatrix(df, diag='box', index='index',colormap='Portland',

                                  colormap_type='cat',

                                  height=700, width=700)

iplot(fig)
#preparing Data

x2015 = humfree.region[humfree.year == 2015]

x2015.value_counts()
# data prepararion

x2011 = humfree.region[humfree.year == 2011]

plt.subplots(figsize=(10,10))

wordcloud = WordCloud(

                          background_color='white',

                          width=512,

                          height=384

                         ).generate(" ".join(x2011))

plt.imshow(wordcloud)

plt.axis('off')

plt.savefig('graph.png')



plt.show()
# data preparation

x2016 = humfree[humfree.year == 2016]



trace0 = go.Box(

    y=x2016.hf_score,

    name = 'Human Freedom score of countries in 2016',

    marker = dict(

        color = 'rgb(34, 245, 140)',

    )

)

trace1 = go.Box(

    y=x2016.ef_score,

    name = 'Economic Freedom score of countries in 2016',

    marker = dict(

        color = 'rgb(123, 128, 128)',

    )

)

data = [trace0, trace1]

iplot(data)
x2013 = humfree[humfree.year == 2013]

new_index = (x2013['hf_rank'].sort_values(ascending=True)).index.values

x2013 = x2013.reindex(new_index)

# first line plot

trace1 = go.Scatter(

    x=x2013.hf_rank,  #Human Freedom rank

    y=x2013.hf_score, # Human Freedom Score

    name = "Human Freedom score",

    marker = dict(color = 'rgba(164, 12, 200, 0.8)'),

)

# second line plot

trace2 = go.Scatter(

    x=x2013.hf_rank,    #Human Freedom rank

    y=x2013.ef_score,#Economic Freedom Score

    xaxis='x2',

    yaxis='y2',

    name = "Economic Freedom score",

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

    title = 'Human Freedom vs Economic Freedom scores'



)



fig = go.Figure(data=data, layout=layout)

iplot(fig)

#preparing Data

x2014 = humfree[humfree.year == 2014]

new_index = (x2014['hf_rank'].sort_values(ascending=True)).index.values

x2014 = x2014.reindex(new_index)



# create trace 1 that is 3d scatter

trace1 = go.Scatter3d(

    x=x2014.hf_rank,

    y=x2014.ef_regulation_business_bribes,

    z=x2014.ef_regulation_business_licensing,

    mode='markers',

    marker=dict(

        size=10,

        color='rgb(145,35,200)',                # set color to an array/list of desired values      

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
#preparing Data

x2011 = humfree[humfree.year == 2011]

new_index = (x2011['hf_rank'].sort_values(ascending=True)).index.values

x2011 = x2011.reindex(new_index)



trace1 = go.Scatter(

    x=x2011.hf_rank,

    y=x2011.ef_government_consumption,

    name = "Government consumption"

)

trace2 = go.Scatter(

    x=x2011.hf_rank,

    y=x2011.ef_money_inflation,

    xaxis='x2',

    yaxis='y2',

    name = "Inflation"

)

trace3 = go.Scatter(

    x=x2011.hf_rank,

    y=x2011.pf_expression_influence,

    xaxis='x3',

    yaxis='y3',

    name = "Laws and regulations that influence media content"

)

trace4 = go.Scatter(

    x=x2011.hf_rank,

    y=x2011.pf_expression,

    xaxis='x4',

    yaxis='y4',

    name = "Freedom of expression"

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

    title = 'Government consumption,Inflation,Laws and regulations,Freedom of expression VS World Rank of countries'

)

fig = go.Figure(data=data, layout=layout)

iplot(fig)