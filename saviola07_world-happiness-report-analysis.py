# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import plotly.plotly as py
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go

import seaborn as sns
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import plotly.offline as py
py.init_notebook_mode(connected=True)
import warnings
warnings.filterwarnings('ignore')
%matplotlib inline

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
data2015 = pd.read_csv('../input/2015.csv')
data2016 = pd.read_csv('../input/2016.csv')
data2017 = pd.read_csv('../input/2017.csv')
data2015.head()
data2015.info()
data2016.head()
data2016.info()
data2017.head()
data2017.info()
data2015.columns
data2015.rename(columns = {'Happiness Rank' : 'Happiness_Rank',
                           'Happiness Score' : 'Happiness_Score',
                           'Standard Error' : 'Standard_Error',
                           'Economy (GDP per Capita)' : 'Economy',
                           'Health (Life Expectancy)' : 'Health',
                          'Trust (Government Corruption)' : 'Trust',
                          'Dystopia Residual' : 'Dystopia_Residual'}, inplace = True)
data2016.columns
data2016.rename(columns = {'Happiness Rank' : 'Happiness_Rank',
                           'Happiness Score' : 'Happiness_Score',
                           'Lower Confidence Interval' : 'Lower_Confidence_Interval',
                           'Upper Confidence Interval' : 'Upper_Confidence_Interval',
                           'Economy (GDP per Capita)' : 'Economy',
                           'Health (Life Expectancy)' : 'Health',
                          'Trust (Government Corruption)' : 'Trust',
                          'Dystopia Residual' : 'Dystopia_Residual'}, inplace = True)
data2017.columns
data2017.rename(columns = {'Happiness.Rank' : 'Happiness_Rank',
                           'Happiness.Score' : 'Happiness_Score',
                           'Whisker.low' : 'Whisker_Low',
                           'Whisker.high' : 'Whisker_High',
                           'Economy..GDP.per.Capita.' : 'Economy',
                           'Health..Life.Expectancy.' : 'Health',
                          'Trust..Government.Corruption.' : 'Trust',
                          'Dystopia.Residual' : 'Dystopia_Residual'}, inplace = True)
data2016['Standard_Error'] = (data2016.Upper_Confidence_Interval - data2016.Lower_Confidence_Interval) / 2
data2016.head()
data2017['Standard_Error'] = (data2017.Whisker_High - data2017.Whisker_Low) / 2
data2017.head()
f,ax = plt.subplots(figsize = (12, 12))
sns.heatmap(data2017.corr(), annot = True, linewidths = 0.1, fmt = '.1f', ax = ax, square = True)
x2015 = data2015['Country']
plt.subplots(figsize=(8,8))
wordcloud = WordCloud(
                          background_color = 'white',
                          width = 512,
                          height = 384
                         ).generate(" ".join(x2015))
plt.imshow(wordcloud)
plt.axis('off')
plt.savefig('graph.png')
plt.show()
data2016.Region.head(10)
data2016.Region.value_counts()
dt2016 = data2016.Region.value_counts()
plt.figure(figsize=(10,7))
sns.barplot(x = dt2016.index, y = dt2016.values)
plt.xlabel('Region')
plt.xticks(rotation = 90)
plt.ylabel('Number of Countries')
plt.title('Number of Countries According to the Region', color = 'blue', fontsize = 20)
plt.show()
labels = data2016.Region.value_counts().index
colors = ['grey', 'blue', 'red', 'yellow', 'green', 'brown', 'orange', 'purple', 'cyan', 'pink']
explode = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
sizes = data2016.Region.value_counts().values

plt.figure(figsize = (7,7))
plt.pie(sizes, explode = explode, labels = labels, colors = colors, autopct = '%1.1f%%')
plt.title("Ratio of Participating Countries to Regions", color = 'blue', fontsize = 20)
plt.show()
x = sns.stripplot(x = "Region", y = "Happiness_Score", data = data2016, jitter = True)
plt.xticks(rotation = 90)
plt.title("Countries' Happiness Score According to the Region", color = 'blue', fontsize = 15)
plt.show()
data2017.mean()
above_mean_HS =['Above World Average' if i >= 5.35 else 'Below World Average' for i in data2017.Happiness_Score]
df = pd.DataFrame({'Happiness_Score' : above_mean_HS})
sns.countplot(x = df.Happiness_Score)
plt.xlabel('Happiness Score')
plt.ylabel('Number of Countries')
plt.title('Number of Countries based on Happiness Score Average', color = 'blue', fontsize = 15)
plt.show()
trace1 = go.Box(
    y = data2017.Economy,
    name = 'Economic Situations of Countries in 2017',
    marker = dict(
        color = 'rgb(12, 12, 140)',
    )
)
trace2 = go.Box(
    y = data2017.Trust,
    name = 'Government Corruption Index in 2017',
    marker = dict(
        color = 'rgb(12, 128, 128)',
    )
)
data = [trace1, trace2]
iplot(data)
dataframe = data2017

trace1 = go.Scatter(
    x = dataframe.Happiness_Rank,
    y = dataframe.Economy,
    name = "Economy",
    marker = dict(color = 'rgba(16, 112, 2, 0.8)')
                    )
# second line plot
trace2 = go.Scatter(
    x = dataframe.Happiness_Rank,
    y = dataframe.Trust,
    xaxis = 'x2',
    yaxis = 'y2',
    name = "Gov. Corruption",
    marker = dict(color = 'rgba(160, 112, 20, 0.8)')
                    )
data = [trace1, trace2]
layout = go.Layout(
        xaxis2 = dict(
            domain = [0.65, 0.95],
            anchor = 'y2'        
                     ),
        yaxis2 = dict(
            domain = [0.65, 0.95],
            anchor = 'x2'
                     ),
    title = 'Economy and Government Corruption Correlation'
                  )
fig = go.Figure(data = data, layout = layout)
iplot(fig)
import plotly.figure_factory as ff

dataframe = data2017
dt2017 = dataframe.loc[:,["Economy", "Health", "Freedom"]]
dt2017["index"] = np.arange(1, len(dt2017) + 1)

fig = ff.create_scatterplotmatrix(dt2017, diag = 'box', index = 'index', colormap = 'Portland',
                                  colormap_type = 'cat',
                                  height = 700, width = 700)
iplot(fig)
dataframe = data2017
trace1 = go.Scatter3d(
    x = dataframe.Economy,
    y = dataframe.Freedom,
    z = dataframe.Health,
    mode = 'markers',
    marker = dict(
        size = 7,
        color = 'rgb(255,0,0)'     
                 )
                      )
data = [trace1]
layout = go.Layout(
    title = 'Happiness Status According to the Economy, Freedom and Health',
    margin = dict(
        l = 0,
        r = 0,
        b = 0,
        t = 30  
                )
                  )
fig = go.Figure(data = data, layout = layout)
iplot(fig)
dataframe = data2017
trace1 = go.Scatter(
    x = dataframe.Happiness_Rank,
    y = dataframe.Economy,
    name = "Economy"
                    )
trace2 = go.Scatter(
    x = dataframe.Happiness_Rank,
    y = dataframe.Health,
    xaxis = 'x2',
    yaxis = 'y2',
    name = "Health"
                    )
trace3 = go.Scatter(
    x = dataframe.Happiness_Rank,
    y = dataframe.Freedom,
    xaxis = 'x3',
    yaxis = 'y3',
    name = "Freedom"
)
trace4 = go.Scatter(
    x = dataframe.Happiness_Rank,
    y = dataframe.Trust,
    xaxis = 'x4',
    yaxis = 'y4',
    name = "Gov. Corruption"
)
data = [trace1, trace2, trace3, trace4]
layout = go.Layout(
    xaxis = dict(
        domain = [0, 0.45]
                ),
    yaxis = dict(
        domain = [0, 0.45]
                ),
    xaxis2 = dict(
        domain = [0.55, 1]
                 ),
    xaxis3 = dict(
        domain = [0, 0.45],
        anchor = 'y3'
                 ),
    xaxis4 = dict(
        domain = [0.55, 1],
        anchor = 'y4'
                 ),
    yaxis2 = dict(
        domain = [0, 0.45],
        anchor = 'x2'
                 ),
    yaxis3 = dict(
        domain = [0.55, 1]
                 ),
    yaxis4 = dict(
        domain = [0.55, 1],
        anchor = 'x4'
                 ),
    title = 'Economy, Health, Freedom and Gov. Corruption Effect in Happiness Rank of Countries'
)
fig = go.Figure(data = data, layout = layout)
iplot(fig)
df = data2017.iloc[:100, :]

import plotly.graph_objs as go

trace1 = go.Scatter(
                    x = df.Happiness_Rank,
                    y = df.Economy,
                    mode = "lines + markers",
                    name = "Economy",
                    marker = dict(color = 'rgba(55, 20, 50, 0.9)'),
                    text = df.Country
                    )
trace2 = go.Scatter(
                    x = df.Happiness_Rank,
                    y = df.Freedom,
                    mode = "lines + markers",
                    name = "Freedom",
                    marker = dict(color = 'rgba(10, 180, 80, 0.9)'),
                    text = df.Country
                    )

data = [trace1, trace2]
layout = dict(title = 'Economy and Freedom Relations in Happiness Rank Top 100 Countries',
             xaxis = dict(title = 'Happiness Rank', ticklen = 5, zeroline = True)
             )
fig = dict(data = data, layout = layout)
iplot(fig)
df2017 = data2017.iloc[:3, :]

import plotly.graph_objs as go

trace1 = go.Bar(
                x = df2017.Country,
                y = df2017.Economy,
                name = "Economy",
                marker = dict(color = 'rgba(125, 25, 200, 0.8)',
                             line = dict(color = 'rgb(0,0,0)', width = 1.5))
                )
trace2 = go.Bar(
                x = df2017.Country,
                y = df2017.Family,
                name = "Family",
                marker = dict(color = 'rgba(25, 25, 25, 0.8)',
                             line = dict(color = 'rgb(0,0,0)', width = 1.5))
                )
trace3 = go.Bar(
                x = df2017.Country,
                y = df2017.Health,
                name = "Health",
                marker = dict(color = 'rgba(190, 200, 100, 0.8)',
                             line = dict(color = 'rgb(0,0,0)', width = 1.5))
                )
trace4 = go.Bar(
                x = df2017.Country,
                y = df2017.Trust,
                name = "Trust",
                marker = dict(color = 'rgba(50, 150, 50, 0.8)',
                             line = dict(color = 'rgb(0,0,0)', width = 1.5))
                )
trace5 = go.Bar(
                x = df2017.Country,
                y = df2017.Generosity,
                name = "Generosity",
                marker = dict(color = 'rgba(255, 70, 12, 0.8)',
                             line = dict(color = 'rgb(0,0,0)', width = 1.5))
                )
trace6 = go.Bar(
                x = df2017.Country,
                y = df2017.Freedom,
                name = "Freedom",
                marker = dict(color = 'rgba(255, 25, 55, 0.8)',
                             line = dict(color = 'rgb(0,0,0)', width = 1.5))
                )
data = [trace1, trace2, trace3, trace4, trace5, trace6]
layout = go.Layout(barmode = 'group', title = 'Top 3 Countries in 2017 According to the Different Parameters')
fig = go.Figure(data = data, layout = layout)
iplot(fig)
df2017 = data2017.iloc[:3, :]

import plotly.graph_objs as go

trace1 = {
  'x': df2017.Country,
  'y': df2017.Economy,
  'name': 'Economy',
  'type': 'bar'
};
trace2 = {
  'x': df2017.Country,
  'y': df2017.Family,
  'name': 'Family',
  'type': 'bar'
};
trace3 = {
  'x': df2017.Country,
  'y': df2017.Health,
  'name': 'Health',
  'type': 'bar'
};
trace4 = {
  'x': df2017.Country,
  'y': df2017.Trust,
  'name': 'Trust',
  'type': 'bar'
};
trace5 = {
  'x': df2017.Country,
  'y': df2017.Generosity,
  'name': 'Generosity',
  'type': 'bar'
};
trace6 = {
  'x': df2017.Country,
  'y': df2017.Freedom,
  'name': 'Freedom',
  'type': 'bar'
};
data = [trace1, trace2, trace3, trace4, trace5, trace6];
layout = {
  'xaxis': {'title': 'Top 3 Countries'},
  'barmode': 'relative',
  'title': 'Top 3 Countries in 2017 According to the Different Parameters'
};
fig = go.Figure(data = data, layout = layout)
iplot(fig)
df2015 = data2015.iloc[:30, :]  
df2016 = data2016.iloc[:30, :]
df2017 = data2017.iloc[:30, :]

import plotly.graph_objs as go

trace1 = go.Scatter(
                    x = df2015.Country,
                    y = df2015.Freedom,
                    mode = "markers",
                    name = "2015",
                    marker = dict(color = 'rgba(55, 20, 50, 0.8)'),
                    text = df2015.Happiness_Rank
                    )

trace2 = go.Scatter(
                    x = df2016.Country,
                    y = df2016.Freedom,
                    mode = "markers",
                    name = "2016",
                    marker = dict(color = 'rgba(10, 180, 80, 0.8)'),
                    text = df2016.Happiness_Rank
                    )

trace3 = go.Scatter(
                    x = df2017.Country,
                    y = df2017.Freedom,
                    mode = "markers",
                    name = "2017",
                    marker = dict(color = 'rgba(255, 128, 2, 0.8)'),
                    text = df2017.Happiness_Rank
                    )
data = [trace1, trace2, trace3]
layout = dict(title = 'Freedom vs Happiness Rank of Top 30 Countries in 2015, 2016 and 2017 Years',
             xaxis = dict(tickangle = 315, ticklen = 3, zeroline = False),
             yaxis = dict(title = 'Freedom', ticklen = 3, zeroline = False)
             )
fig = dict(data = data, layout = layout)
iplot(fig)
data = dict(type = 'choropleth', 
           locations = data2017['Country'],
           locationmode = 'country names',
           z = data2017['Happiness_Rank'],
           colorbar = {'title':'Happiness Scale'})
layout = dict(title = 'Global Happiness Ranking', 
             geo = dict(showframe = False, 
                       projection = {'type': 'natural earth'}))
choromap = go.Figure(data = [data], layout = layout)
iplot(choromap)