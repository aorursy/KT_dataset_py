from IPython.display import YouTubeVideo

YouTubeVideo('mQ94xbXnYu4', width=800, height=450)
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
%matplotlib inline

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


import itertools
import warnings
warnings.filterwarnings("ignore")
from wordcloud import WordCloud,STOPWORDS
import io
import base64
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.basemap import Basemap
import folium
import folium.plugins
import os
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
import plotly.figure_factory as ff
import random
number_of_colors = 1000
color = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
             for i in range(number_of_colors)]
color_theme = dict(color = color)

#---------------------Bar Plot----------------------------------------#
def plot_insight(x,y,header="",xaxis="",yaxis=""):
    trace = go.Bar(y=y,x=x,marker=color_theme)
    layout = go.Layout(title = header,xaxis=dict(title=xaxis,tickfont=dict(size=13,)),
                       yaxis=dict(title=yaxis,titlefont=dict(size=16),tickfont=dict(size=14)))
    data = [trace]
    fig = go.Figure(data=data, layout=layout)
    return iplot(fig,filename='basic-bar1')

#---------------------Word Cloud Plot----------------------------------------#
def word_cloud_graph(df):
    # data prepararion
    plt.subplots(figsize=(20,12))
    wordcloud = WordCloud(background_color='white',width=512,height=384,).generate(" ".join(df))
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.savefig('graph.png')
    return plt.show()
events = pd.read_csv("../input/athlete_events.csv")
events.head()
print("Dataset Shape\nRows:{0}\nColumns:{1}".format(events.shape[0], events.shape[1]))
# Check Missing values
x = events.isna().sum().to_frame()
x.columns = ["Columns"]
x.plot.bar(rot=90, figsize=(20,9),title = "Missing value Count By Column",fontsize=16)
plt.xlabel("Columns",fontsize=18)
plt.ylabel("Missing Value Count",fontsize=18)
events.dtypes.value_counts().plot.bar(rot=90,figsize=(20,9),title = "Dataset Types",fontsize=16)
plt.xlabel("Data Types",fontsize=18)
plt.ylabel("Count",fontsize=18)
events["Age"] = events["Age"].fillna(events["Age"].mean)
events["Height"] = events["Height"].fillna(events["Height"].mean)
events["Weight"] = events["Weight"].fillna(events["Weight"].mean)
events["Medal"] = events["Medal"].fillna("Runners_ups")
# Check Missing values
x = events.isna().sum().to_frame()
x.columns = ["Columns"]
x.plot.bar(rot=90, figsize=(20,9),title = "Missing value Count By Column",fontsize=16)
plt.xlabel("Columns",fontsize=18)
plt.ylabel("Missing Value Count",fontsize=18)
from plotly.offline import init_notebook_mode, iplot
import plotly.graph_objs as go
init_notebook_mode(connected=True)

df_teams = events.groupby("Team")["Medal"].count().to_frame()
plot_insight(df_teams.index,df_teams["Medal"],header="Team By Total Medal",xaxis="Team Name",yaxis="Medals")
df_medal = events.groupby("Team")["Medal"].value_counts().to_frame()
df_medal.plot(kind="bar", figsize = (20,10))
