import numpy as np 
import pandas as pd 

import plotly.plotly as py
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go

from wordcloud import WordCloud

import matplotlib.pyplot as plt

import os
print(os.listdir("../input"))


data1=pd.read_csv("../input/winemag-data_first150k.csv")
data2=pd.read_csv("../input/winemag-data-130k-v2.csv")
data2.info()
country_list=list(data1["country"].unique())
wine_point_average=[]
wine_price_average=[]
for i in country_list:
    x=data1[data1["country"]==i]
    if len(x)!=0:
        country_price_average=round((sum(x.price)/len(x)),2)
        country_points_average=round((sum(x.points)/len(x)),2)
        wine_price_average.append(country_price_average)
        wine_point_average.append(country_points_average)
    else:
        wine_price_average.append("0")
        wine_point_average.append("0")
print(country_list)
df=pd.DataFrame({"country_list":country_list,"wine_price_average":wine_price_average,"wine_point_average":wine_point_average})

trace1 = go.Scatter(
                    x = df.country_list,
                    y = df.wine_point_average,
                    mode = "lines+markers",
                    name = "points",
                    marker = dict(color = 'blue'),
                    text= df.country_list)

trace2 = go.Scatter(
                    x = df.country_list,
                    y = df.wine_price_average,
                    mode = "lines+markers",
                    name = "price ($)",
                    marker = dict(color = 'green'),
                    text= df.country_list)
data1 = [trace1, trace2]
layout = dict(title = 'Price and Points Average of Wines by Country',
              xaxis= dict(title= 'Countries',ticklen= 5,zeroline= False)
             )
fig = dict(data = data1, layout = layout)
iplot(fig)

country_list=list(data2["country"].unique())
wine_point_average=[]
wine_price_average=[]
for i in country_list:
    x=data2[data2["country"]==i]
    if len(x)!=0:
        country_price_average=round((sum(x.price)/len(x)),2)
        country_points_average=round((sum(x.points)/len(x)),2)
        wine_price_average.append(country_price_average)
        wine_point_average.append(country_points_average)
    else:
        wine_price_average.append("0")
        wine_point_average.append("0")
print(country_list)
df2=pd.DataFrame({"country_list":country_list,"wine_price_average":wine_price_average,"wine_point_average":wine_point_average})

trace1 = go.Scatter(
                    x = df2.country_list,
                    y = df2.wine_point_average,
                    mode = "lines+markers",
                    name = "points",
                    marker = dict(color = 'blue'),
                    text= df2.country_list)

trace2 = go.Scatter(
                    x = df2.country_list,
                    y = df2.wine_price_average,
                    mode = "lines+markers",
                    name = "price ($)",
                    marker = dict(color = 'green'),
                    text= df2.country_list)
data2 = [trace1, trace2]
layout = dict(title = 'Price and Points Average of Wines by Country',
              xaxis= dict(title= 'Countries',ticklen= 5,zeroline= False)
             )
fig = dict(data = data2, layout = layout)
iplot(fig)