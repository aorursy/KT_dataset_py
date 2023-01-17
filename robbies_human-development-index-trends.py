# Import the relevant libraries

%matplotlib inline

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')
# Input the Data

country = pd.read_csv('../input/Human Development Index trends.csv',encoding='ISO-8859-1',dtype=object)

country.head()
print(country.shape)
# Create new list of countries

country_list = ['Afghanistan','Angola','Albania','Argentina','Armenia','Australia'

,'Austria','Azerbaijan','Burundi','Belgium','Benin','Burkina Faso','Bangladesh','Bulgaria'

,'Bahrain','Bosnia and Herzegovina','Belarus','Belize','Bolivia','Brazil','Barbados','Brunei Darussalam'

,'Bhutan','Botswana','Central African Republic','Canada','Switzerland','Chile','China','Cameroon'

,'Congo','Colombia','Comoros','Cabo Verde','Costa Rica','Cuba','Cyprus','Czech Republic','Germany'

,'Denmark','Dominican Republic','Algeria','Ecuador','Egypt','Spain','Estonia','Ethiopia','Finland','Fiji'

,'France','Gabon','United Kingdom','Georgia','Ghana','Guinea','Greece','Guatemala','Guyana','Hong Kong'

,'Honduras','Croatia','Haiti','Hungary','Indonesia','India','Ireland','Iran','Iraq','Iceland','Israel'

,'Italy','Jamaica','Jordan','Japan','Kazakhstan','Kenya','Cambodia','Korea','Kuwait','Lebanon','Liberia'

,'Libya','Sri Lanka','Lesotho','Lithuania','Luxembourg','Latvia','Macao','Morocco','Moldova','Madagascar'

,'Maldives','Mexico','Macedonia','Mali','Malta','Myanmar','Montenegro','Mongolia','Mozambique','Mauritania'

,'Mauritius','Malawi','Malaysia','North America','Namibia','Niger','Nigeria','Nicaragua','Netherlands'

,'Norway','Nepal','New Zealand   ','Oman','Pakistan','Panama','Peru','Philippines','Papua New Guinea'

,'Poland','Puerto Rico','Portugal','Paraguay','Qatar','Romania','Russian Federation','Rwanda','Saudi Arabia'

,'Sudan','Senegal','Singapore','Solomon Islands','Sierra Leone','El Salvador','Somalia','Serbia','Slovenia'

,'Sweden','Swaziland','Syrian Arab Republic','Chad','Togo','Thailand','Tajikistan','Turkmenistan','Timor-Leste'

,'Trinidad and Tobago','Tunisia','Turkey','Tanzania','Uganda','Ukraine','Uruguay','United States','Uzbekistan'

,'Venezuela, RB','Vietnam','Yemen, Rep.','South Africa','Congo, Dem. Rep.','Zambia','Zimbabwe' 

]
country_clean = country[country['Country'].isin(country_list)]
# Plotting 2000 and 2014 visuals

metricscale1=[[0, 'rgb(102,194,165)'], [0.05, 'rgb(102,194,165)'], [0.15, 'rgb(171,221,164)'], [0.2, 'rgb(230,245,152)'], [0.25, 'rgb(255,255,191)'], [0.35, 'rgb(254,224,139)'], [0.45, 'rgb(253,174,97)'], [0.55, 'rgb(213,62,79)'], [1.0, 'rgb(158,1,66)']]

data = [ dict(

        type = 'choropleth',

        autocolorscale = False,

        colorscale = metricscale1,

        showscale = True,

        locations = country_clean['Country'].values,

        z = country_clean['2000'].values,

        locationmode = 'country names',

        text = country_clean['Country'].values,

        marker = dict(

            line = dict(color = 'rgb(250,250,225)', width = 0.5)),

            colorbar = dict(autotick = True, tickprefix = '', 

            title = 'HD Index')

            )

       ]



layout = dict(

    title = 'World Map of Human Development in the Year 2000',

    geo = dict(

        showframe = True,

        showocean = True,

        oceancolor = 'rgb(28,107,160)',

        #oceancolor = 'rgb(222,243,246)',

        projection = dict(

        type = 'orthographic',

            rotation = dict(

                    lon = 60,

                    lat = 10),

        ),

        lonaxis =  dict(

                showgrid = False,

                gridcolor = 'rgb(102, 102, 102)'

            ),

        lataxis = dict(

                showgrid = False,

                gridcolor = 'rgb(102, 102, 102)'

                )

            ),

        )

fig = dict(data=data, layout=layout)

py.iplot(fig, validate=False, filename='worldmap2000')



metricscale2=[[0, 'rgb(102,194,165)'], [0.05, 'rgb(102,194,165)'], [0.15, 'rgb(171,221,164)'], [0.2, 'rgb(230,245,152)'], [0.25, 'rgb(255,255,191)'], [0.35, 'rgb(254,224,139)'], [0.45, 'rgb(253,174,97)'], [0.55, 'rgb(213,62,79)'], [1.0, 'rgb(158,1,66)']]

data = [ dict(

        type = 'choropleth',

        autocolorscale = False,

        colorscale = metricscale2,

        showscale = True,

        locations = country_clean['Country'].values,

        z = country_clean['2014'].values,

        locationmode = 'country names',

        text = country_clean['Country'].values,

        marker = dict(

            line = dict(color = 'rgb(250,250,200)', width = 0.5)),

            colorbar = dict(autotick = True, tickprefix = '', 

            title = 'HD Index')

            )

       ]



layout = dict(

    title = 'World Map of Human Development in the Year 2014',

    geo = dict(

        showframe = True,

        showocean = True,

        oceancolor = 'rgb(28,107,160)',

        projection = dict(

        type = 'orthographic',

            rotation = dict(

                    lon = 60,

                    lat = 10),

        ),

        lonaxis =  dict(

                showgrid = False,

                gridcolor = 'rgb(202, 202, 202)',

                width = '0.05'

            ),

        lataxis = dict(

                showgrid = False,

                gridcolor = 'rgb(102, 102, 102)'

                )

            ),

        )

fig = dict(data=data, layout=layout)

py.iplot(fig, validate=False, filename='worldmap2014')
# Scatter plot of 2000 hdi

trace = go.Scatter(

    y = country_clean['2000'].values,

    mode='markers',

    marker=dict(

        size= country_clean['2000'].values,

        #color = np.random.randn(500), #set color equal to a variable

        color = country_clean['2000'],

        colorscale='Viridis',

        showscale=True

    ),

    text = country_clean['Country'].values

)

data = [trace]



layout= go.Layout(

    autosize= True,

    title= 'Scatter plot of Human Development in 2000',

    hovermode= 'closest',

#     xaxis= dict(

#         title= 'Pop',

#         ticklen= 5,

#         zeroline= False,

#         gridwidth= 2,

#     ),

    yaxis=dict(

        title= 'Human Development Index',

        ticklen= 5,

        gridwidth= 2,

    ),

    showlegend= False

)

fig = go.Figure(data=data, layout=layout)

py.iplot(fig,filename='scatter2000')







# Scatter plot of 2014 Human Development Index

trace1 = go.Scatter(

    y = country_clean['2014'].values,

    mode='markers',

    marker=dict(

        size=country_clean['2014'].values,

        #color = np.random.randn(500), #set color equal to a variable

        color = country_clean['2014'].values,

        colorscale='Viridis',

        showscale=True

    ),

    text = country_clean['Country']

)

data = [trace1]



layout= go.Layout(

    title= 'Scatter plot of Human Development in 2014',

    hovermode= 'closest',

    xaxis= dict(

        ticklen= 5,

        zeroline= False,

        gridwidth= 2,

    ),

    yaxis=dict(

        title= 'Human Development Index',

        ticklen= 5,

        gridwidth= 2,

    ),

    showlegend= False

)



fig = go.Figure(data=data, layout=layout)

py.iplot(fig,filename='scatter2014')