# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
gd = pd.read_csv("../input/gender_development.csv")
hd = pd.read_csv("../input/human_development.csv")
ia = pd.read_csv("../input/inequality_adjusted.csv")
hi = pd.read_csv("../input/historical_index.csv")
#data preparition
hd["Gross National Income (GNI) per Capita"] = [',' + each if ',' not in each else each for each in hd["Gross National Income (GNI) per Capita"]]
hd["Gross National Income (GNI) per Capita"] = [float(each.replace(',','.')) for each in hd["Gross National Income (GNI) per Capita"]]
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected = True)
import plotly.graph_objs as go

trace1 = go.Scatter(
                    x = hd["Gross National Income (GNI) per Capita"],
                    y = hd["Human Development Index (HDI)"],
                    mode = "markers",
                    marker = dict(color = 'rgba(255,189,1, 0.7)'),
                    text = hd["Country"]
)
layout_result = dict(title = "Income and Human Development Correlation", #hovermode='closest',
             xaxis = dict(title = "Gross National Income (GNI) per Capita", ticklen = 5, zeroline = False),
             yaxis = dict(title = "Human Development", ticklen = 5, zeroline = False))
fig = dict(data = [trace1], layout = layout_result)
iplot(fig)

fig = dict(
    data = [go.Scatter(
                    x = hd["Life Expectancy at Birth"],
                    y = hd["Human Development Index (HDI)"],
                    mode = "markers",
                    marker = dict(color = 'rgba(1,189,255, 0.7)'),
                    text = hd["Country"]
            )], 
    
    layout = dict(title = "Life Expectancy and Human Development Correlation",
             xaxis = dict(title = "Life Expectancy at Birth", ticklen = 5, zeroline = False),
             yaxis = dict(title = "Human Development", ticklen = 5, zeroline = False)))
iplot(fig)
fig = dict(
    data = [go.Scatter(
                    x = hd["Mean Years of Education"],
                    y = hd["Human Development Index (HDI)"],
                    mode = "markers",
                    marker = dict(color = 'rgba(255,1,189, 0.7)'),
                    text = hd["Country"]
            )], 
    
    layout = dict(title = "Education and Human Development Correlation",
             xaxis = dict(title = "Mean Years of Education", ticklen = 5, zeroline = False),
             yaxis = dict(title = "Human Development", ticklen = 5, zeroline = False)))
iplot(fig)
trace1 = go.Scatter(
    y = ia["Inequality-adjusted HDI (IHDI)"],
    x = ia["Human Development Index (HDI)"],
    name = "Human Development and Inequality-adjusted",
    mode = "markers",
    marker = dict(color = 'rgba(205,125,79, 0.7)'),
    text = ia["Country"]
)

trace2 = go.Bar(
    x=ia["Country"].head(8),
    y=ia["Human Development Index (HDI)"].head(8),
    xaxis = 'x2',
    yaxis = 'y2',
    name = "Human Development",
    marker = dict(color = "rgba(0,230,255, 0.8)"),
)
trace3 = go.Bar(
    x=ia["Country"].head(8),
    y=ia["Inequality-adjusted HDI (IHDI)"].head(8),
    xaxis = 'x2',
    yaxis = 'y2',
    name = "Inequality-adjusted",
    marker = dict(color = "rgba(0,180,255, 0.8)"),
)

layout = go.Layout(
    xaxis = dict(
        domain=[0, 0.45],
        title = "Human Development",
    ),
    yaxis = dict(
         domain=[0, 1],
         title = "Inequality-adjusted"
    ),
    xaxis2=dict(
        domain=[0.55, 1],
        anchor='y2',
        title = "Top Countries According To Human Development"
    ),
    yaxis2=dict(
        domain=[0, 1],
        anchor='x2',
    ),
    barmode = "group",
    title = "Inequality-adjusted and Human Development Correlation",
    legend=dict(x=-.1, y=1.12),
)


fig = go.Figure(data = [trace1, trace2, trace3], layout =  layout )
iplot(fig)
#gni, educated
countries = hd.loc[hd["Mean Years of Education"].sort_values(ascending=False).head(15).keys(), "Country"]


trace1 = go.Bar(
    x=countries.values,
    y=hd.loc[countries.keys(),"Gross National Income (GNI) per Capita"].head(15),
    name = "Top 15 Educated Countries",
    marker = dict(color = "rgba(0,170,255, 0.8)"),
    text = "Mean Years of Education : " + hd.loc[countries.keys(), "Mean Years of Education"].astype(str)
)

trace2 = go.Scatter(
    y = hd["Gross National Income (GNI) per Capita"],
    x = hd["Mean Years of Education"],
    name = "Education and Gross National Income (GNI) per Capita",
    xaxis='x2',
    yaxis='y2',
    mode = "markers",
    marker = dict(color = 'rgba(205,125,79, 0.5)'),
    text = ia["Country"]
)


layout = go.Layout(
    xaxis2=dict(
        domain=[0.6, 1],
        anchor='y2',
    ),
    yaxis2=dict(
        domain=[0.55, 1],
        anchor='x2',
    ),
    title = "Top 15 Educated Countries And GNI per Capita",
    yaxis = dict(title = "Gross National Income (GNI) per Capita", ticklen = 5, zeroline = False),
    #hovermode='closest'
    legend=dict(x=-.1, y=1.12),
)

fig = go.Figure(data = [trace1,trace2], layout =  layout )
iplot(fig)


countries = hd.loc[hd["Gross National Income (GNI) per Capita"].sort_values(ascending=False).head(15).keys(), "Country"]

trace1 = go.Scatter(
    x = hd["Gross National Income (GNI) per Capita"],
    y = gd["Gender Development Index (GDI)"],
    name = "GNI Per Capita and Gender Development",
    mode = "markers",
    marker = dict(color = 'rgba(205,125,79, 0.5)'),
    text = ia["Country"]
)
trace2 = go.Bar(
    x=countries.values,
    y=hd.loc[countries.keys(),"Human Development Index (HDI)"].head(15),
    xaxis='x2',
    yaxis='y2',
    name = "Human Development",
    text = "Per capita : " + hd.loc[countries.keys(), "Gross National Income (GNI) per Capita"].astype(str),
    marker = dict(color = "rgba(0,170,255, 0.8)"),
)
trace3 = go.Bar(
    x=countries.values,
    y=gd.loc[countries.keys(),"Gender Development Index (GDI)"].head(15),
    xaxis='x2',
    yaxis='y2',
    name = "Gender Development",
    text = "Per capita : " + hd.loc[countries.keys(), "Gross National Income (GNI) per Capita"].astype(str),
    marker = dict(color = "rgba(0,230,255, 0.8)"),
)

layout = go.Layout(
    xaxis=dict(
        domain=[0, 1],                
    ),
    yaxis=dict(
        domain=[0.55, 1],
        title = "Gender Development",
    ),
    
    xaxis2=dict(
        domain=[0, 1],
        anchor='y2',    
    ),
    yaxis2=dict(
        domain=[0, .45],
        anchor='x2',
    ),
    barmode = "group",
    title = "GNI per capita and Gender Development Correlation",
    #yaxis = dict(title = "Gross National Income (GNI) per Capita", ticklen = 5, zeroline = False),
    hovermode = "closest",
    legend=dict(x = 0.6,  y=0.8),
)

fig = go.Figure(data = [trace1,trace2, trace3], layout =  layout )
iplot(fig)
trace1 = go.Box(
    y=gd["Life Expectancy at Birth (Female)"],
    name = "Life Expectancy at Birth (Female)"
)
trace2 = go.Box(
    y=gd["Life Expectancy at Birth (Male)"],
    name = "Life Expectancy at Birth (Male)"
)
fig = go.Figure( data = [trace1, trace2], layout = dict(title = "Life Expectancy According To Years"))
iplot(fig)
trace1 = go.Box(
    y=gd["Estimated Gross National Income per Capita (Female)"],
    name = "Female"
)
trace2 = go.Box(
    y=gd["Estimated Gross National Income per Capita (Male)"],
    name = "Male"
)
fig = go.Figure( data = [trace1, trace2], layout = dict(title = "Estimated Gross National Income per Capita"))
iplot(fig)
columnNames = hi.loc[:, 'Human Development Index (1990)':].columns
traces = [0] * len(columnNames)
i=0
for column in columnNames:
    traces[i] = go.Box(
    y=hi[column],
    name = column.split(" ")[3]
    )
    i += 1
fig = go.Figure( data = traces , layout = dict(title = "Human Development According To Years"))
iplot(fig) 
