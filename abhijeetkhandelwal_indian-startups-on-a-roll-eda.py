# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import collections

pd.options.display.max_columns = 999

import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls

import warnings
warnings.filterwarnings('ignore')


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
df = pd.read_csv("../input/startup_funding.csv")
print(df.head(5))
# Any results you write to the current directory are saved as output.
#Finding out the shape and Null values in the given dataset
print(df.shape)
df.isnull().sum()

#dropping remarks and subvertical due to large amount of null value
del df['SubVertical']
del df['Remarks']

#mapping null values in columns with general or unknown values for better understanding
df['CityLocation'] = df['CityLocation'].fillna('India')
df['InvestorsName'] = df['InvestorsName'].fillna('Unknown')
df['IndustryVertical'] = df['IndustryVertical'].fillna('Other/Unknown')

# Finding Data type of the columns 
#formating columns = AmountinUSD and Date
df["AmountInUSD"] = df["AmountInUSD"].apply(lambda x: float(str(x).replace(",",""))).astype(float)
df['AmountInUSD'] = df['AmountInUSD'].fillna(df['AmountInUSD'].mean())
df = df.dropna(axis=0, how='any')
#date formating
def format_Investdate(d):
    if "." in d:
        d=d.replace("." , "/")
    elif "//" in d:
        d=d.replace("//","/")
    return d
df['Date'] = df['Date'].apply(format_Investdate)
df['Date']=pd.to_datetime(df['Date'], format='%d/%m/%Y')
#Plotting investment type against count
# now for the company sector feature

a = df.InvestmentType.values
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
    title= 'Investment Type',
    hovermode= 'closest',
    yaxis=dict(
        title= 'Total',
        ticklen= 1,
        gridwidth= 0.5
    ),
    showlegend= False
)
fig = go.Figure(data=dataa, layout=layout)
py.iplot(fig, filename='barplothouse')


dt_amo=df['IndustryVertical'].groupby([df.IndustryVertical]).agg('count').nlargest(10)

keytype = list(dt_amo.keys())
populationtype = list(dt_amo.values)

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
    title= 'Industry Vertical',
    hovermode= 'closest',
    yaxis=dict(
        title= 'Number of investment',
        ticklen= 1,
        gridwidth= 0.5
    ),
    showlegend= False
)
fig = go.Figure(data=dataa, layout=layout)
py.iplot(fig, filename='barplothouse')

#which city is getting maximum number of funding

dt_loc=df['CityLocation'].groupby([df.CityLocation]).agg('count').nlargest(10)

keytype = list(dt_loc.keys())
populationtype = list(dt_loc.values)

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
    title= 'Cities getting Maximum Number of Funding',
    hovermode= 'closest',
    yaxis=dict(
        title= 'Number of investment',
        ticklen= 1,
        gridwidth= 0.5
    ),
    showlegend= False
)
fig = go.Figure(data=dataa, layout=layout)
py.iplot(fig, filename='barplothouse')

#Top 10 indian cities by funding

dt_cit=df['AmountInUSD'].groupby([df.CityLocation]).agg('sum').nlargest(10)

keytype = list(dt_cit.keys())
populationtype = list(dt_cit.values)

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
    title= 'Cities getting maximum funding',
    hovermode= 'closest',
    yaxis=dict(
        title= 'Number of investment',
        ticklen= 1,
        gridwidth= 0.5
    ),
    showlegend= False
)
fig = go.Figure(data=dataa, layout=layout)
py.iplot(fig, filename='barplothouse')


#Top 10 indian cities by funding

dt_start=df['AmountInUSD'].groupby([df.StartupName]).agg('sum').nlargest(10)

keytype = list(dt_start.keys())
populationtype = list(dt_start.values)

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
    title= 'Top 10 Startup by Funding',
    hovermode= 'closest',
    yaxis=dict(
        title= 'Investment Amount',
        ticklen= 1,
        gridwidth= 0.5
    ),
    showlegend= False
)
fig = go.Figure(data=dataa, layout=layout)
py.iplot(fig, filename='barplothouse')

#df.groupby(pd.TimeGrouper(freq='M'))
#Top 10 indian cities by funding
dt_year=df['AmountInUSD'].groupby(pd.DatetimeIndex(df['Date']).year).agg('sum')

keytype = list(dt_year.keys())
populationtype = list(dt_year.values)

dataa = [go.Bar(
            y= populationtype,
            x = keytype,
            width = .5,
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
    title= 'Total Funding by year',
    hovermode= 'closest',
    yaxis=dict(
        title= 'Total Investment',
        ticklen= 3,
        gridwidth= 1
    ),
    xaxis= dict(title= 'Year',ticklen= 5,zeroline= False),
    showlegend= False
)
fig = go.Figure(data=dataa, layout=layout)
py.iplot(fig, filename='barplothouse')

#Month - > Jan =1 , Feb = 2,...and so on
dt_month=df['AmountInUSD'].groupby(pd.DatetimeIndex(df['Date']).month).agg('sum')

keytype = list(dt_month.keys())
populationtype = list(dt_month.values)

dataa = [go.Bar(
            y= populationtype,
            x = keytype,
            width = .5,
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
    title= 'Total Funding by Month',
    hovermode= 'closest',
    yaxis=dict(
        title= 'Amount',
        ticklen= 1,
        gridwidth= 1
    ),
    showlegend= False
)
fig = go.Figure(data=dataa, layout=layout)
py.iplot(fig, filename='barplothouse')

dataa = [go.Scatter(
            y= df.AmountInUSD,
            x = df.Date,
            mode = "markers",
            marker=dict(
               color = populationtype,
            colorscale='Portland',
            showscale=True,
            reversescale = True
            ),
            text= df.StartupName,
            opacity=0.6
        )]

layout= go.Layout(
    #autosize= True,
    title= 'Total Funding by year',
    hovermode= 'closest',
    yaxis=dict(
        title= 'Investment',
        ticklen= 1,
        gridwidth= 1
    ),
    showlegend= False
)
fig = go.Figure(data=dataa, layout=layout)
py.iplot(fig, filename='barplothouse')


