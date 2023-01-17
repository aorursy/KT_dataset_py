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
df = pd.read_csv("../input/datam.csv",usecols=range(0, 15))

df


import plotly.graph_objs as go



trace1 = go.Scatter(

                    x = df.sıra,

                    y = df.twitter,

                    mode = "lines",

                    name = "Twitter",

                    marker = dict(color = 'rgba(16, 112, 2, 0.8)'),

                    text= df.adi)

trace2 = go.Scatter(

                    x = df.sıra,

                    y = df.facebook,

                    mode = "lines+markers",

                    name = "Facebook",

                    marker = dict(color = 'rgba(255, 182, 193, .9)'),

                    text= df.adi)





data = [trace1,trace2]

layout = dict(title = 'Twitter and World Rank of Top 100 Universities',

              xaxis= dict(title= 'Turkey Rank',ticklen= 5,zeroline= False)

             )

fig = dict(data = data, layout = layout)

iplot(fig)