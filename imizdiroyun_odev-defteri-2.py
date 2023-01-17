# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
%matplotlib inline
from wordcloud import WordCloud
from plotly.offline import init_notebook_mode, iplot, plot
import plotly.graph_objs as go

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
# Reading data 

movieData = pd.read_csv('../input/tmdb-movie-metadata/tmdb_5000_movies.csv')

# Information About Data

movieData.info()
# Datas About first 10 movies
movieData.head()
# More information about data
movieData.describe()
movieData.head()
data1 = movieData.original_language.value_counts()
data1
# Bar Plot 
# Dillere gore vote avarage ortalamalari:

lang_list = list(movieData['original_language'].unique())
lang_avarage = []

# Datayi siralamak
for i in lang_list:
    x = movieData[movieData['original_language']==i]
    vote_av = sum(x.vote_average)/len(x)
    lang_avarage.append(vote_av)
    
data = pd.DataFrame({'Language': lang_list, 'Vote Avarage': lang_avarage})
new_index = (data['Vote Avarage'].sort_values(ascending=False)).index.values
sorted_data = data.reindex(new_index)

# Visualization

plt.figure(figsize=(15,10))
sns.barplot(x=sorted_data['Language'], y=sorted_data['Vote Avarage'], palette = sns.cubehelix_palette(len(lang_list)))
plt.xticks(rotation=45)
plt.xlabel('Language')
plt.ylabel('Vote Avarage')
plt.title('Vote Avarage of Languages')



movieData.head(5)
myData1 = movieData[movieData['original_language']=='en']
myData = myData1.head(100)
new_index = (myData['vote_average'].sort_values(ascending=False)).index.values
lineData = myData.reindex(new_index)



# Line Charts 
# Top 100 Ingilizce filmlerin  budget ve revenue 

trace1 = go.Scatter(
                    x = lineData.vote_average,
                    y = lineData.budget,
                    mode = "lines",
                    marker = dict(color = 'rgb(14,209,69)'),
                    text = lineData.original_title)

trace2 = go.Scatter(
                    x = lineData.vote_average,
                    y = lineData.revenue,
                    mode = "lines+markers",
                    marker = dict(color = 'rgb(255,0,0)'),
                    text = lineData.original_title)

data = [trace1,trace2]
layout = dict(title='IMDB Top 100 English Movies Budget vs Revenue')
fig = dict(data=data, layout = layout)
iplot(fig)


# Scatter Plot

# Cikis Yillarina Gore TOP 1 Budget Filmin Vote Avarage vs Vote Count


# Adding New Year Column 
movieData1 = movieData.release_date.astype(str)
movieData["release_year"] = [each[0:4] for each in list(movieData1) ]

year_list = list(movieData["release_year"].unique())
max_budget = []


for i in year_list:
    x = movieData[movieData['release_year']==i]
    max_deger = max(x.budget)
    max_budget.append(max_deger)


# Sorting
data = pd.DataFrame({'Release Years': year_list, 'Max Budget Value': max_budget})
new_index = (data['Max Budget Value'].sort_values(ascending=False)).index.values
sorted_data = data.reindex(new_index)


# Visualization

plt.figure(figsize=(20,15))
sns.barplot(x=sorted_data['Release Years'], y=sorted_data['Max Budget Value'], palette = sns.cubehelix_palette(len(year_list)))
plt.xticks(rotation=90)
plt.xlabel('Release Years')
plt.ylabel('Max Budget Value')
plt.title('Compare Of The Highest Budget Values For Years')

                   


movieData.head()
sns_data = movieData.drop(columns= "id")

f,ax = plt.subplots(figsize=(15, 15))
sns.heatmap(sns_data.corr(), annot=True, linewidths=0.3,linecolor="black", fmt= '.1f', ax=ax)
plt.show()
movieData.head(5)
sorted_data.head(5)
# Bar Chart
# Compare of popularity vs revenue of 2011's Top 10 Movies

# Preparing Data

data = movieData[movieData['release_year'] == '2011']
new_index = (data['vote_average'].sort_values(ascending=False)).index.values
sorted_data = data.reindex(new_index)
bar_data = sorted_data.head(10)

# Popularity plot
trace1 = go.Bar(
                x = bar_data.original_title, 
                y = bar_data['runtime'], 
                name = "Runtime",
                marker = dict(color = 'rgba(255, 0, 0, 0.5)',
                             line=dict(color='rgb(0,0,0)',width=1.5)), 
                text = bar_data.vote_average)

# Revenue Plot
trace2 = go.Bar(
                x = bar_data.original_title, 
                y = bar_data.vote_count, 
                name = "Vote Counts",
                marker = dict(color = 'rgba(10, 10, 255, 0.5)',
                             line=dict(color='rgb(0,0,0)',width=1.5)), 
                text = bar_data.vote_average)


# Layout
data = [trace1, trace2] 
layout = go.Layout(barmode = "group")
                
    
fig = go.Figure(data = data, layout = layout)
iplot(fig)    


movieData.head()
plt.hist(movieData["original_language"])
plt.show()

# Pie Chart
# IMDB TOP 500 Movie Language Pie Chart

keys = movieData["original_language"].value_counts().keys()
values = movieData["original_language"].value_counts().values
values1 = list(values)    

labels = movieData["original_language"]
fig = {
  "data": [
    {
      "values": values1,
      "labels": labels, 
      "domain": {"x": [0, .5]}, 
      "name": "Movie Language Rates",
      "hoverinfo":"label+percent",
      "hole": .2,
      "type": "pie" 
    },],
  "layout": {
        "title":"TOP 500 Movie Language Rates",
        "annotations": [
            { "font": { "size": 20},
              "showarrow": False,
              "text": "Movie Language Rates",
                "x": 0.60,
                "y": 1,
            },
        ]
    }
}
iplot(fig)
    
   
    
movieData.head()
# Bubble Chart
# Vote Average Rank Top 10 Movies in 2011 revenue vs budget

bubbleData1 = movieData[movieData["release_year"] == "2011"]
bubbleData = bubbleData1.reindex(new_index)
myData = bubbleData.head(10)
runtime = [each/3.5 for each in myData.runtime]


data = [
    {
        'y': myData.revenue,
        'x': myData.vote_average,
        'mode': 'markers',
        'marker': {
            'color': myData.popularity,
            'size': runtime,
            'showscale': True
        },
        "text" :  myData.title    
    }
]
iplot(data)
movieData.original_language.value_counts()
# Histogram
# Histogram of vote average French vs Deutch

frData = movieData[movieData["original_language"] == "fr"]
deData = movieData[movieData["original_language"] == "de"]

fr = frData.vote_average
de = deData.vote_average

trace1 = go.Histogram(
    x= fr,
    opacity= 0.75,
    name = "French Movies",
    marker=dict(color='rgba(171, 50, 96, 0.6)'))
trace2 = go.Histogram(
    x= de,
    opacity= 0.75,
    name = "Deutch",
    marker=dict(color='rgba(12, 50, 196, 0.6)'))

data = [trace1, trace2]
layout = go.Layout(barmode='overlay',
                   title=' Histogram of Vote Average French vs Deutch',
                   xaxis=dict(title='vote_average'),
                   yaxis=dict( title='Frequency'),
)
fig = go.Figure(data=data, layout=layout)
iplot(fig)
# World Cloud
# Word Cloud of the Top 50 Movies Language

wordCloud_data = movieData.reindex(new_index)
wordCloud_data = wordCloud_data.head(50)
myData = wordCloud_data.original_language

wordcloud = WordCloud( 
                          background_color='white', 
                          width=520, 
                          height=380 
                         ).generate(" ".join(myData))  
plt.imshow(wordcloud) 
plt.axis('off') 
plt.show()
# Box Plot
# Box Plot of popularity and revenue of Movies which relased in 2016,  

myData = movieData[movieData["release_year"] == "2016"]
runtime_data = [int(each) for each in myData.runtime]
voteCount_data = [each/100 for each in myData.vote_count]


trace0 = go.Box(
    y= runtime_data,
    name = 'Runtime of 2016 Movies',
    marker = dict(
        color = 'rgb(12, 12, 140)',
    )
)
trace1 = go.Box(
    y=voteCount_data,
    name = 'Vote Count of 2016 Movies',
    marker = dict(
        color = 'rgb(12, 128, 128)',
    )
)
data = [trace0, trace1]
iplot(data)
import plotly.figure_factory as ff

# Scatter Matrix Plot
# Vote Average vs Vote Count vs Revenue vs Budget of 2016 Movies

data = movieData[movieData.release_year == "2016"]
myData = data.loc[:,["vote_average","vote_count","budget","revenue"]]
myData["index"] = np.arange(1,len(myData)+1)
                            
fig = ff.create_scatterplotmatrix(myData, diag='box', index='index',colormap='Portland',
                                  colormap_type='cat',
                                  height=700, width=700)                            
iplot(fig)



movieData.head()
00# 3D Scatter Plot 
# 3D Scatter Plot of Runtime, Vote Count and Vote average 
runtimeData = [each*10 for each in movieData.runtime]  
votecountData= [each/10 for each in movieData.vote_count]



trace1 = go.Scatter3d( 
    x=runtimeData,
    y=votecountData, 
    z=movieData.vote_average, 
    mode='markers',  
    marker=dict(
        size=5
    )
)

data = [trace1]
layout = go.Layout(
    margin=dict( 
        l=1, 
        r=1, 
        b=1, 
        t=1  
    )
    
)
fig = go.Figure(data=data, layout=layout)
iplot(fig)

