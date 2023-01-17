# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib as plt

# plotly
import plotly.plotly as py
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go

# word cloud library
from wordcloud import WordCloud

# matplotlib
import matplotlib.pyplot as plt


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
dataframe = pd.read_csv('../input/countries of the world.csv')

dataframe.info()
dataframe.head()
trace1 = go.Scatter(
    x = dataframe.index,
    y = dataframe.Deathrate,
    mode = 'lines+markers',
    name = 'Deathrate',
    marker = dict(color = 'rgba(255, 51, 51, 0.5)'),
    text = dataframe.Country)

trace2 = go.Scatter(
    x = dataframe.index,
    y = dataframe.Birthrate,
    mode = 'lines+markers',
    name = 'Birthrate',
    marker = dict(color = 'rgba(51, 102, 255, 0.5)'),
    text = dataframe.Country)

layout = dict(title = 'Birthrate and Deathrate of Countries',
             xaxis= dict(zeroline= False)
             )

data = [trace1, trace2]

fig = dict(data = data, layout = layout)

iplot(fig)
sorted_data = (dataframe.sort_values(ascending=False,by=['GDP ($ per capita)']))
sorted_data = (sorted_data.reset_index(drop=True)).loc[:4]


trace0 = go.Bar(
    x = sorted_data.Country,
    y = sorted_data['Agriculture'],
    name = "Agriculture",
    marker = dict(color = 'rgba(255, 26, 26, 0.5)',
                    line=dict(color='rgb(100,100,100)',width=2)))

trace1 = go.Bar(
    x = sorted_data.Country,
    y = sorted_data['Industry'],
    name = "Industry",
    marker = dict(color = 'rgba(255, 255, 51, 0.5)',
                line=dict(color='rgb(100,100,100)',width=2)))

trace2 = go.Bar(
    x = sorted_data.Country,
    y = sorted_data['Service'],
    name = "Service",
    marker = dict(color = 'rgba(77, 77, 255, 0.5)',
                    line=dict(color='rgb(100,100,100)',width=2)))

data = [trace0, trace1, trace2]

layout = go.Layout(barmode = "group")

fig = go.Figure(data = data,layout = layout)

iplot(fig)
regions = dataframe.Region.unique()
num_regions = []

for i in regions:
    num_region = len(dataframe.Region[dataframe.Region==i])
    num_regions.append(num_region)

fig = {
    "data" : [{
        "values" : num_regions,
        "labels" : regions,
        "domain" : {"x": [0, .5]},
        "name" : "Number of Region",
        "hoverinfo":"label+percent",
        "hole" : .3,
        "type" : "pie"
    },],
    "layout" : {
        "title" : "Regions of Countries",
        "annotations": [
            { "font": { "size": 20},
              "showarrow": True,
              "text": "World Regions Ratio",
                "x": 0.20,
                "y": 1
             }]
    }
}       

iplot(fig)
sorted_data2 = (dataframe.sort_values(ascending=False,by=['Population']))#sorting data
sorted_data2 = (sorted_data2.reset_index(drop=True)).loc[:20]#locating first 20 data
Literacy  = [float(each.replace(',', '.')) for each in sorted_data2['Literacy (%)']]#transforming datatype to float
Population = sorted_data2['Population']/10000000 #normalization

data = [
    {
        'x' : sorted_data2.Country,
        'y' : sorted_data2['GDP ($ per capita)'],
        'mode' : 'markers',
        'marker' : {
            'color' : Literacy,
            'size' : Population,
            'showscale' : True,
            'colorbar' : { 'title' : "Population(x10,000,000)"}
        },
        
        "text" :  "GDP ($ per capita)"
    }
]
iplot(data)
trace0 = go.Histogram(
        x = dataframe['Net migration'],
        opacity = 0.8,
        name = "Net Migration",
        marker = dict(color='rgba(0, 255, 255, 0.5)'))

layout = go.Layout(
        barmode='overlay',
        title='Migration Distribution',
        xaxis=dict(title='Net Migration'),
        yaxis=dict( title='Countries')
 )

data = [trace0]

fig = go.Figure(data=data, layout=layout)

iplot(fig)
plt.subplots(figsize=(24,15))

#separating words of country names
separate = dataframe.Country.str.split(' ')#separating words of country names  
a,b = zip(*separate)
country_name = a+b

wordcloud = WordCloud(
                          background_color='white',
                          width = 1920,
                          height = 1080
                         ).generate(" ".join(country_name))

plt.imshow(wordcloud)
plt.axis('off')
plt.savefig('graph.png')

plt.show()
trace0 = go.Box(
    y = dataframe.Birthrate,
    name = "Birthrates of Countries",
    marker = dict(color = 'rgba(0, 255, 0, 0.8)')
    )

trace1 = go.Box(
    y = dataframe.Deathrate,
    name = "Deathrates of Countries",
    marker = dict(color = 'rgba(255, 0, 0, 0.8)')
    )
data = [trace0,trace1]
iplot(data)
import plotly.figure_factory as ff

a = dataframe.loc[:,["Agriculture","Industry","Service"]]#we need only these columns

a.dropna(inplace = True)#droping nan values
a.Agriculture = [float(each.replace(',','.')) for each in a.Agriculture]#transforming data from string to float
a.Industry = [float(each.replace(',','.')) for each in a.Industry]
a.Service = [float(each.replace(',','.')) for each in a.Service]

a["index"] = np.arange(1,len(a)+1)#rearrange index
fig = ff.create_scatterplotmatrix(a, diag='box', index='index',colormap='Portland',
                                  colormap_type='cat',
                                  height=800, width=800)
iplot(fig)

trace1 = go.Scatter3d(
    x = a.Agriculture,
    y = a.Industry,
    z = a.Service,
    mode='markers',
    marker=dict(
        size=10,
        color='rgb(153, 153, 255)'               
    )
)
data = [trace1]

iplot(data)


trace1 = go.Scatter(
    x = dataframe.index,
    y = dataframe.Population,
    mode = 'lines+markers',
    name = 'Population',
    marker = dict(color = 'rgba(255, 102, 0, 0.5)'),
    text = dataframe.Country)

trace2 = go.Scatter(
    x = dataframe.index,
    y = dataframe['GDP ($ per capita)'],
    mode = 'lines+markers',
    xaxis='x2',
    yaxis='y2',
    name = 'GDP',
    marker = dict(color = 'rgba(77, 255, 77, 0.5)'),
    text = dataframe.Country)

layout = go.Layout(
    xaxis2=dict(
        domain=[0.5, 1],
        anchor='y2',        
    ),
    yaxis2=dict(
        domain=[0.5, 1],
        anchor='x2',
    ),
    title = 'Population and GDP'
)



data = [trace1, trace2]

fig = dict(data = data, layout = layout)

iplot(fig)
trace0 = go.Scatter(
    x = dataframe.Country,
    y = dataframe['Phones (per 1000)'],
    name = "Phones(per 1000 people)")
trace1 = go.Scatter(
    x = dataframe.Country,
    y = dataframe['Coastline (coast/area ratio)'],
    xaxis='x2',
    yaxis='y2',
    name = "Coastline (coast/area ratio)")
trace2 = go.Scatter(
    x = dataframe.Country,
    y = dataframe['Pop. Density (per sq. mi.)'],
    xaxis='x3',
    yaxis='y3',
    name = "Population Density (per sq.mi.)")
trace3 = go.Scatter(
    x = dataframe.Country,
    y = dataframe['Infant mortality (per 1000 births)'],
    xaxis='x4',
    yaxis='y4',
    name = "Infant mortality (per 1000 births)")
data = [trace0, trace1, trace2, trace3]

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
    title = 'World',
    width = 1024,
    height = 768)

fig = go.Figure(data=data, layout=layout)
iplot(fig)