# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns  # visualization tool
from wordcloud import WordCloud, STOPWORDS
import warnings
from collections import Counter
import datetime
import wordcloud
import json
import matplotlib as mpl
%matplotlib inline
from subprocess import check_output
import os
import warnings
warnings.filterwarnings('ignore')

# Data Munging
pd.options.mode.chained_assignment = None
pd.options.display.max_columns = 999
from IPython.display import HTML

# Data Visualizations
plt.style.use('fivethirtyeight')
import seaborn as sns
import squarify
# Plotly has such beautiful graphs
import plotly
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.tools as tls
import plotly.graph_objs as go
import plotly.tools as tls
import plotly.figure_factory as fig_fact
plotly.tools.set_config_file(world_readable=True, sharing='public')




# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory


from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/battles.csv')
data3=pd.read_csv('../input/character-predictions.csv')
data2 = pd.read_csv('../input/character-deaths.csv')
PLOT_COLORS = ["#268bd2", "#0052CC", "#FF5722", "#b58900", "#003f5c"]
pd.options.display.float_format = '{:.2f}'.format
sns.set(style="ticks")
plt.rc('figure', figsize=(8, 5), dpi=100)
plt.rc('axes', labelpad=20, facecolor="#ffffff", linewidth=0.4, grid=True, labelsize=14)
plt.rc('patch', linewidth=0)
plt.rc('xtick.major', width=0.2)
plt.rc('ytick.major', width=0.2)
plt.rc('grid', color='#9E9E9E', linewidth=0.4)
plt.rc('font', family='Arial', weight='400', size=10)
plt.rc('text', color='#282828')
plt.rc('savefig', pad_inches=0.3, dpi=300)
mpl.rcParams['font.size']=20              #10 
mpl.rcParams['savefig.dpi']=100             #72 
mpl.rcParams['figure.subplot.bottom']=.1
f,ax = plt.subplots(figsize=(10, 10))

stopwords = set(STOPWORDS)
data2 = pd.read_csv('../input/character-deaths.csv')

wordcloud = WordCloud(
                          background_color='white',
                          stopwords=stopwords,
                          max_words=200,
                          max_font_size=40, 
                          random_state=42
                         ).generate(str(data2['Name']))

print(wordcloud)
fig = plt.figure(1)
plt.imshow(wordcloud)
plt.axis('off')
plt.show()
data.info()
data.corr()
data.columns
data.head(10)
#correlation map
f,ax = plt.subplots(figsize=(8, 8))
sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
plt.show()
# Line Plot
# color = color, label = label, linewidth = width of line, alpha = opacity, grid = grid, linestyle = sytle of line
data.major_death.plot(kind = 'line', color = 'blue',label = 'major_death',linewidth=1,alpha = 1,grid = True,linestyle = ':')
data.major_capture.plot(color = 'r',label = 'major_capture',linewidth=1, alpha = 1,grid = True,linestyle = '-.')
plt.legend(loc='upper right')     # legend = puts label into plot
plt.xlabel('x axis')              # label = name of label
plt.ylabel('y axis')
plt.title('Line Plot')            # title = title of plot
plt.show()
data.head(10)
data.boxplot(column='battle_number', by = 'year')
plt.show()
data.describe()
data_1 = data.head(6)
melted = pd.melt(frame = data_1,id_vars = 'attacker_king',value_vars=['attacker_size', 'defender_size'])
melted
fig, ax = plt.subplots()
_ = sns.distplot(data[data["attacker_size"] < 25e6]["attacker_size"], kde=False, 
                 color=PLOT_COLORS[3], hist_kws={'alpha': 1}, ax=ax)
_ = ax.set(xlabel="attacker_size")
fig, ax = plt.subplots()
_ = sns.distplot(data[data["defender_size"] < 25e6]["defender_size"], kde=False, 
                 color=PLOT_COLORS[4], hist_kws={'alpha': 1}, ax=ax)
_ = ax.set(xlabel=" Defender Size")
data2.head(25)
trace = []
for name, group in data2.groupby("Allegiances"):
    trace.append ( 
        go.Box(
            x=group["Book Intro Chapter"].values,
            name=name
        )
    )
layout = go.Layout(
    title='Book Intro Chapter',
    width = 800,
    height = 800
)
#data = [trace0, trace1]
fig = go.Figure(data=trace, layout=layout)
py.iplot(fig, filename="Book Intro Chapter")

data3.head(5)
print(data3['house'].value_counts(dropna=False))
sns.catplot(x='alive', y='house', data=data3.head(30), kind ='bar')

data.head(10)
