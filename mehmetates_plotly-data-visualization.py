# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# import plotly.plotly as py
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go

# wordcloud
from wordcloud import WordCloud

# matplotlib
import matplotlib.pyplot as plt
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
timesData = pd.read_csv("../input/world-university-rankings/timesData.csv")
timesData.info()
timesData.head()
# Data tanımlama ilk 100 datayı alıyoruz
df = timesData.iloc[:100,:]

# import plotly.graph_objs as go (Yukarıda import ettik)

# ilk trace oluşturalım
trace1 = go.Scatter(x = df.world_rank,
                    y = df.citations,
                    mode="lines",
                    name="citations",
                    marker=dict(color= 'rgba(16,112,2,0.8)'),
                    text=df.university_name)

# ikinci trace oluşturalım
trace2 = go.Scatter(x = df.world_rank,
                    y = df.teaching,
                    mode = "lines+markers",
                    name = "teaching",
                    marker = dict(color = 'rgba(80,26,80,0.8)'),
                    text = df.university_name)
# Grafikte kullanacağımız trace leri bir data da listelidik.
data = [trace1, trace2]
# Layout ile dict yapısı ile grafiğin başlığı, x eksenin ismi gibi tanımlamalar yaptık.
layout = dict(title = "Citation and Teaching vs Word Rank of Top 100 University",
              xaxis = dict(title = "Word Rank", ticklen = 5, zeroline = False))
fig = dict(data = data, layout = layout)
iplot(fig)
# data tanımları 2014,2015,2015 top 100 üniversite
df2014 = timesData[timesData.year == 2014].iloc[:100,:]
df2015 = timesData[timesData.year == 2015].iloc[:100,:]
df2016 = timesData[timesData.year == 2016].iloc[:100,:]

# import graph object as go

import plotly.graph_objs as go

#creating trace1

trace1 = go.Scatter(x = df2014.world_rank,
                    y = df2014.citations,
                    mode = "markers",
                    name = "2014",
                    marker = dict(color = "rgba(255, 128, 255, 0.8)"),
                    text = df2014.university_name)

trace2 = go.Scatter(x = df2015.world_rank,
                    y = df2015.citations,
                    mode = "markers",
                    name = "2015",
                    marker = dict(color = "rgba(255, 128, 2, 0.8)"),
                    text = df2015.university_name)

trace3 = go.Scatter(x = df2016.world_rank,
                    y = df2016.citations,
                    mode = "markers",
                    name = "2016",
                    marker = dict(color = "rgba(0, 225, 200, 0.8)"),
                    text = df2016.university_name)

data = [trace1, trace2, trace3]
layout = dict(title = "Citation vs Word Rank of Top 100 universities with 2014,2015,2016 years",
              xaxis = dict(title = "Word Rank", ticklen = 5, zeroline = False),
              yaxis = dict(title = "Citation", ticklen = 5, zeroline = False))

fig = dict(data = data, layout = layout)
iplot(fig)
df2014 = timesData[timesData.year == 2014].iloc[:3,:]

trace1 = go.Bar(x = df2014.university_name,
                y = df2014.citations,
                name = "citations",
                marker = dict(color = "rgba(255,175,255,0.5)",
                              line = dict(color = "rgb(0,0,0)", width = 1.5)),
                text = df2014.country)

trace2 = go.Bar(x = df2014.university_name,
                y = df2014.teaching,
                name = "teaching",
                marker = dict(color = "rgba(255,255,128,0.5)",
                              line = dict(color = "rgb(0,0,0)", width = 1.5)),
                text = df2014.country)

data = [trace1, trace2]
layout = go.Layout(barmode="group")
fig = dict(data=data, layout=layout)
iplot(fig)
x = df2014.university_name

trace1 = {
    "x": x,
    "y": df2014.citations,
    "name": "citation",
    "type": "bar"
};

trace2 = {
    "x": x,
    "y": df2014.teaching,
    "name": "teaching",
    "type": "bar"
};

data = [trace1, trace2]
layout = {
    "xaxis": {"title": "top 3 universities"},
    "barmode": "relative",
    "title": "citations and teaching of top 3 university in 2014"
};

fig = go.Figure(data=data, layout=layout)
iplot(fig)