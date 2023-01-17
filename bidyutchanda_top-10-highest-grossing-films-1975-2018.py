# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
dataset = pd.read_csv('../input/blockbusters.csv')
dataset.head()
for i, row in dataset.iterrows(): #Iterate through each row of dataframe
    gross = dataset.worldwide_gross[i]
    gross = gross.replace('$','') #Trims $ from the values
    gross = gross.replace(',','') #Trims , from the values.
    dataset.worldwide_gross[i] = gross
for i, row in dataset.iterrows():
    gross = dataset.worldwide_gross[i]
    gross = float(gross)
    gross = gross/1000000
    dataset.worldwide_gross[i] = int(gross)
dataset.worldwide_gross[0]
type(dataset.worldwide_gross[0])
# mandatory imports for offline plotting of plotly
from plotly.offline import init_notebook_mode, iplot
import plotly.graph_objs as go

trace1 = go.Scatter( # for plotting scatter graph, go.Scatter is used
        y = dataset.imdb_rating,
        x = dataset.Main_Genre,
        marker = dict(color = 'orange', size = 7, opacity = 0.5), # marker attributes inside graph
        mode = "markers",
        text = dataset.title # text for each marker during hover
)

data = [trace1]

layout = go.Layout(
    title = 'IMDb ratings v/s Genres of Highest Grossing Films (1975-2018)',
    xaxis = dict(showgrid = True, zeroline = True, gridwidth = 2), 
    yaxis = dict(title = 'IMDb Rating', gridwidth = 2),
    hovermode = 'closest',
    paper_bgcolor='rgb(243, 243, 243)',
    plot_bgcolor='rgb(243, 243, 243)',
)

fig = go.Figure(data = data, layout = layout)
# for offline glue-ing of plot into notebook
init_notebook_mode(connected=True)
iplot(fig)
from plotly.offline import init_notebook_mode, iplot
import plotly.graph_objs as go

trace = go.Scatter(
    x = dataset.imdb_rating,
    y = dataset.length,
    text = dataset.title,
    marker = dict(color = 'blue', size = 7, opacity = 0.5),
    mode = 'markers'
)

data = [trace]

layout = go.Layout(
    title = 'Lengths v/s IMDb Ratings of Highest Grossing Films (1975-2018)',
    xaxis = dict(title = 'IMDb Rating', gridwidth = 2),
    yaxis = dict(title = 'Length (in minutes)', gridwidth = 2),
    hovermode = 'closest',
    paper_bgcolor='rgb(243, 243, 243)',
    plot_bgcolor='rgb(243, 243, 243)'
)

figure = go.Figure(data = data, layout = layout)
init_notebook_mode(connected=True)
iplot(figure)
from plotly.offline import init_notebook_mode, iplot
import plotly.graph_objs as go

trace = go.Box(
    x = dataset.rating,
    y = dataset.imdb_rating,
    text = dataset.title,
    marker = dict(color = 'green')
)

data = [trace]

layout = go.Layout(
    title = 'Maturity Rating v/s IMDb Ratings of Highest Grossing Films (1975-2018)',
    xaxis = dict(title = 'Maturity Rating', gridwidth = 2),
    yaxis = dict(title = 'IMDb Rating', gridwidth = 2),
    hovermode = 'closest',
    paper_bgcolor='rgb(243, 243, 243)',
    plot_bgcolor='rgb(243, 243, 243)'
)

figure = go.Figure(data = data, layout = layout)
init_notebook_mode(connected=True)
iplot(figure)
from plotly.offline import init_notebook_mode, iplot
import plotly.graph_objs as go
from numpy import * 



trace = go.Box(
    x = dataset.studio,
    y = dataset.imdb_rating,
    text = dataset.title,
    marker = dict(color = 'rgb(107,176,156)')
)

data = [trace]

layout = go.Layout(
    title = 'Distribution of IMDb Ratings for different Studios of Highest Grossing Films (1975-2018)',
    xaxis = dict(gridwidth = 2),
    yaxis = dict(title = 'IMDb Rating', gridwidth = 2),
    hovermode = 'closest',
    paper_bgcolor='rgb(243, 243, 243)',
    plot_bgcolor='rgb(243, 243, 243)'
)

figure = go.Figure(data = data, layout = layout)
init_notebook_mode(connected=True)
iplot(figure)
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize = (25,9))
sns.set_style('darkgrid')

fig = sns.boxplot( x=dataset.year, y=dataset.imdb_rating, palette = "cubehelix")

plt.ylabel('IMDb Rating', fontsize = 17)
plt.xlabel('Years of Release', fontsize = 17)
plt.xticks(rotation = 70, fontsize = 13)
plt.yticks(fontsize = 13)
plt.title('Variation of IMDb Ratings over the Years for the Highest Grossing Films (1975-2018)', fontsize = 25)
plt.grid(True, alpha = 0.5, linestyle = '-.', color = '#000000')
plt.show()
from plotly.offline import init_notebook_mode, iplot
import plotly.graph_objs as go

trace1 = go.Box( # for plotting scatter graph, go.Scatter is used
        y = dataset.length,
        x = dataset.Main_Genre,
        marker = dict(color = 'darkturquoise'), # marker attributes inside graph
        #mode = "markers",
        text = dataset.title # text for each marker during hover
)

data = [trace1]

layout = go.Layout(
    title = 'Lengths v/s Genres of Highest Grossing Films (1975-2018)',
    xaxis = dict(showgrid = True, zeroline = True, gridwidth = 2), 
    yaxis = dict(title = 'Length (in minutes)', gridwidth = 2),
    hovermode = 'closest',
    paper_bgcolor='rgb(243, 243, 243)',
    plot_bgcolor='rgb(243, 243, 243)',
)

fig = go.Figure(data = data, layout = layout)
# for offline glue-ing of plot into notebook
init_notebook_mode(connected=True)
iplot(fig)
from plotly.offline import init_notebook_mode, iplot
import plotly.graph_objs as go


trace = go.Box(
    x = dataset.studio,
    y = dataset.length,
    text = dataset.title,
    marker = dict(color = 'firebrick')
)

data = [trace]

layout = go.Layout(
    title = 'Variation of Lengths for different Studios of Highest Grossing Films (1975-2018)',
    xaxis = dict(gridwidth = 2),
    yaxis = dict(title = 'Length (in minutes)', gridwidth = 2),
    hovermode = 'closest',
    paper_bgcolor='rgb(243, 243, 243)',
    plot_bgcolor='rgb(243, 243, 243)'
)

figure = go.Figure(data = data, layout = layout)
init_notebook_mode(connected=True)
iplot(figure)
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize = (15,10))
sns.set_style('white')

fig = sns.regplot(x="year", y="length", data=dataset)

plt.ylabel('Length of film (in minutes)', fontsize = 17)
plt.xlabel('Years of Release', fontsize = 17)
plt.xticks(fontsize = 13)
plt.yticks(fontsize = 13)
plt.title('Variation of lengths of films over the Years for the Highest Grossing Films (1975-2018)', fontsize = 23)
plt.show()
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize = (15,10))
sns.set_style('white')

fig = sns.regplot(x="length", y="worldwide_gross", data=dataset, color='g')

plt.ylabel('Worldwide Gross (in millions of dollars)', fontsize = 17)
plt.xlabel('Length of film (in minutes)', fontsize = 17)
plt.xticks(fontsize = 13)
plt.yticks(fontsize = 13)
plt.title('Length versus Worldwide Gross for the Highest Grossing Films (1975-2018)', fontsize = 23)
plt.show()
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize = (20,13))
sns.set_style('white')

fig = sns.countplot(y="Main_Genre", hue="rating", data=dataset, palette="husl")

plt.ylabel('',fontsize = 17)
plt.xlabel('Number of films', fontsize = 17)
plt.xticks(fontsize = 13)
plt.yticks(fontsize = 13)
plt.legend(bbox_to_anchor=(1,1), loc=2, prop={'size':15}, title="Maturity Rating")
plt.title('Number of films for each Main Genre based on their Maturity Rating', fontsize = 23)
plt.show()
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize = (17,13))
sns.set_style('white')

fig = sns.countplot(y="studio", hue="rating", data=dataset, palette="YlGnBu")

plt.ylabel('',fontsize = 17)
plt.xlabel('Number of films', fontsize = 17)
plt.xticks(fontsize = 13)
plt.yticks(fontsize = 13)
plt.legend(bbox_to_anchor=(1,1), loc=2, prop={'size':13}, title="Maturity Rating")
plt.title('Count of films for all popular Studios based on their Maturity Rating', fontsize = 23)
plt.show()
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize = (12,7))
sns.set_style('white')

dataset.worldwide_gross = dataset.worldwide_gross.astype(float) #because violinplot needs one axis to be of float values.
ax = sns.violinplot(x="rating", y="worldwide_gross", data=dataset, inner = None) #inner=None to delete the middle line inside the violins. 
ax = sns.swarmplot(x="rating", y="worldwide_gross", data=dataset, color = '#00CED1', edgecolor = 'gray')

plt.ylabel('Worldwide Gross (in millions of dollars)', fontsize = 17)
plt.xlabel('Maturity Ratings', fontsize = 17)
plt.xticks(fontsize = 13)
plt.yticks(fontsize = 13)
plt.grid(True)
plt.title('Maturity Rating versus Worldwide Gross for the Highest Grossing Films (1975-2018)', fontsize = 20)
plt.show()
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize = (10,20))
sns.set_style('white')

fig = sns.countplot(y="year", hue="rating", data=dataset, palette="Set2")

plt.ylabel('Year of Release', fontsize = 17)
plt.xlabel('Count of films', fontsize = 17)
plt.xticks(fontsize = 13)
plt.yticks(fontsize = 13)
plt.legend(bbox_to_anchor=(1,1), loc=2, prop={'size':13}, title="Maturity Rating")
plt.title('Number of films over the years for all maturity levels', fontsize = 20)
plt.show()
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize = (15,10))
sns.set_style('white')

fig = sns.regplot(x="year", y="worldwide_gross", data=dataset, color='#C71585')

plt.ylabel('Worldwide Gross (in millions of dollars)', fontsize = 17)
plt.xlabel('Year of Release', fontsize = 17)
plt.xticks(fontsize = 13)
plt.yticks(fontsize = 13)
plt.title('Variation of Worldwide Gross over the years for the Highest Grossing Films (1975-2018)', fontsize = 23)
plt.show()
from plotly.offline import init_notebook_mode, iplot
import plotly.graph_objs as go


trace = go.Box(
    x = dataset.studio,
    y = dataset.worldwide_gross,
    text = dataset.title,
    marker = dict(color = '#4B0082')
)

data = [trace]

layout = go.Layout(
    title = 'How does the famous studios perform at the box office?',
    xaxis = dict(gridwidth = 2),
    yaxis = dict(title = 'Worldwide Gross (in millions of dollars)', gridwidth = 2, zeroline = False),
    hovermode = 'closest',
    paper_bgcolor='rgb(243, 243, 243)',
    plot_bgcolor='rgb(243, 243, 243)'
)

figure = go.Figure(data = data, layout = layout)
init_notebook_mode(connected=True)
iplot(figure)