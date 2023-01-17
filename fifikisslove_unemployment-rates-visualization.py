# Import the relevant libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import plotly.offline as py

py.init_notebook_mode(connected=False)

import plotly.graph_objs as go

import plotly.tools as tls

import seaborn as sns
data_all = pd.read_csv("../input/API_ILO_country_YU.csv")
europeans = ["DEU","BEL","BGR","DNK","EST","FIN","FRA","GRC","IRL","ITA","LVA","LTU","LUX","MLT","NLD","AUT","POL","PRT","ROU","SWE","SVK","SVN","ESP","CZE","HUN","GBR","CYP","HRV",]

great_power = ["USA","IND","CHN","JPN", "RUS", "BRA", "DEU", "CAN", "FRA", "GBD"]
data_all["variation"] = data_all["2014"] - data_all["2010"]
def extractSpecificCountries(data_all, CountryList):

    for index, row in data_all.iterrows():

        if row['Country Code'] not in CountryList:

            data_all = data_all.drop(index)

    return data_all



data_europe = extractSpecificCountries(data_all, europeans)

data_great_power = extractSpecificCountries(data_all, great_power)
pos_variation = data_all.nlargest(5, "variation")

neg_variation = data_all.nsmallest(5, "variation")



data2 = []



for index, row in pos_variation.iterrows():  

    data2.append(go.Scatter(

        x = ["2010","2011","2012","2013","2014"],

        y = row[["2010","2011","2012","2013","2014"]].values.tolist(),

        mode = 'lines',

        legendgroup = 'Good',

        name = row["Country Name"],

        line = dict(

        color = ('rgb(39, 174, 96)'),

        shape='spline'

        )



    ))



for index, row in neg_variation.iterrows():  

    data2.append(

        go.Scatter(

        x = ["2010","2011","2012","2013","2014"],

        y = row[["2010","2011","2012","2013","2014"]].values.tolist(),

        mode = 'lines',

        legendgroup = 'Bad',

        name = row["Country Name"],

        line = dict(

        color = ('rgb(52, 73, 94)'),

                shape='spline'

        )



    ))

    

layout = dict(title = 'Countries showing the biggest delta between 2010 and 2014',

              xaxis = dict(title = 'Year'),

              yaxis = dict(title = 'Unemployement Rate (%)', showgrid=False, zeroline=False)

              )   



py.iplot(dict(data=data2, layout=layout), filename='line-mode')
data2 = []



for index, row in data_great_power.iterrows():  

    data2.append(go.Scatter(

        x = ["2010","2011","2012","2013","2014"],

        y = row[["2010","2011","2012","2013","2014"]].values.tolist(),

        mode = 'lines',

        legendgroup = 'Good',

        name = row["Country Name"],

        line = dict(

        shape='spline'

        )



    ))

    

layout = dict(title = 'Great Power Unemployement Rate',

              xaxis = dict(title = 'Year'),

              yaxis = dict(title = 'Unemployement Rate (%)'),

              ) 

py.iplot(dict(data=data2, layout=layout), filename='line-mode')



scl = [[0.0, 'rgb(46, 204, 113)'],[1.0, 'rgb(231, 76, 60)']]



data = [ dict(

        type='choropleth',

        colorscale = scl,

        autocolorscale = False,

        locations = data_all['Country Code'],

        z = data_all["variation"].astype(float),

        #locationmode = 'EUROPE',

        #text = df['text'],

        marker = dict(

            line = dict (

                color = 'rgb(255,255,255)',

                width = 1

            ) ),

        colorbar = dict(

            title = "%")

        ) ]

layout = dict(

        title = 'Younth Unemployement Rate in Europe',



        geo = dict(

            scope='world'

             ))

    

fig = dict( data=data, layout=layout)

py.iplot( fig, filename='d3-cloropleth-map')