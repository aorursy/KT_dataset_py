import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import collections



pd.options.display.max_columns = 999



import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls



import warnings

warnings.filterwarnings('ignore')
#lets get the dataset now

data = pd.read_csv('../input/us_companies.csv')

# and lets check out how it looks 

data.head()
#lets see which features have how many NaN values

# my bad, the only one matplotlib plot :P

missing_df = data.isnull().sum(axis=0).reset_index()

missing_df.columns = ['column_name', 'missing_count']

missing_df = missing_df.ix[missing_df['missing_count']>0]

missing_df = missing_df.sort_values(by='missing_count')



ind = np.arange(missing_df.shape[0])

width = 1.7

fig, ax = plt.subplots(figsize=(8,10))

rects = ax.barh(ind, missing_df.missing_count.values, color='#4A148C')

ax.set_yticks(ind)

ax.set_yticklabels(missing_df.column_name.values, rotation='horizontal')

ax.set_xlabel("Count of missing values")

ax.set_title("Number of missing values in each column")

plt.show()
# there is a feature 'country', as all are US based, I dont think there is any use of it

len(data.country.unique())
# Now lets check out the state feature

a = data.state.values

counter=collections.Counter(a)



key = list(counter.keys())

population = list(counter.values())



scale=[[0, '#84FFFF'], [0.25, '#00E5FF'], [0.65, '#40C4FF'],[1, '#01579B']]





dataa = [ dict(

        type='choropleth',

        colorscale = scale,

        locations = key,

        z = population,

        locationmode = "USA-states",

        marker = dict(

            line = dict (

                color = 'rgb(255,255,255)',

                width = 2

            ) ),

        colorbar = dict(

            title = "Number of Companies")

        ) ]



layout = dict(

        title = 'Frequency of companies by state<br>(Hover for number of cos)',

        geo = dict(

            scope='usa',

            projection=dict( type='albers usa' ),

            showlakes = True,

            lakecolor = 'rgb(255, 255, 255)'),

             )

    

fig = dict( data=dataa, layout=layout )

py.iplot( fig, filename='statefreq' )
# now lets check out the company_type feature

a = data.company_type.values

counttype=collections.Counter(a)



keytype = list(counttype.keys())

populationtype = list(counttype.values())



dataa = [go.Bar(

            y= populationtype,

            x = keytype,

            width = 0.5,

            marker=dict(

               color = populationtype,

            colorscale='Portland',

            showscale=True,

            reversescale = False

            ),

            opacity=0.6

        )]



layout= go.Layout(

    autosize= True,

    title= 'Distribution of Company type<br>(please pull the ticks on the X-axis to see the remaining types)',

    hovermode= 'closest',

    yaxis=dict(

        title= 'Number of companies',

        ticklen= 5,

        gridwidth= 2

    ),

    showlegend= False

)

fig = go.Figure(data=dataa, layout=layout)

py.iplot(fig, filename='barplottype')
# now for the company sector feature

a = data.company_category.values

countercoscat=collections.Counter(a)



keytype = list(countercoscat.keys())

populationtype = list(countercoscat.values())



dataa = [go.Bar(

            y= populationtype,

            x = keytype,

            width = 0.5,

            marker=dict(

               color = populationtype,

            colorscale='Portland',

            showscale=True,

            reversescale = False

            ),

            opacity=0.6

        )]



layout= go.Layout(

    #autosize= True,

    title= 'Distribution of Company Sector<br>(please scroll along the x-axis to see the remaining sectors)',

    hovermode= 'closest',

    yaxis=dict(

        title= 'Number of companies',

        ticklen= 1,

        gridwidth= 0.5

    ),

    showlegend= False

)

fig = go.Figure(data=dataa, layout=layout)

py.iplot(fig, filename='barplothouse')
# lets also analyse the year_founded, seems useful

a = data.year_founded.values

counteryear=collections.Counter(a)

counteryear