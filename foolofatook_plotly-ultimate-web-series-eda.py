#importing the necessary libraries

import pandas as pd

import numpy as np

import re

from wordcloud import WordCloud

from wordcloud import STOPWORDS

import plotly as plt

import plotly.express as px

import plotly.graph_objects as go

from plotly.offline import init_notebook_mode

import matplotlib.pyplot as mplt

from PIL import Image

import requests

from io import BytesIO

init_notebook_mode()
#reading the dataset

webSeries = pd.read_csv('../input/web-series-ultimate-edition/All_Streaming_Shows.csv')

webSeries.head()
#finding the number of rows in the dataset

print(f'There are {webSeries.shape[0]} TV series in this dataset')
#filling the null values with -1

webSeries['Content Rating'] = webSeries['Content Rating'].fillna('-1')

webSeries['IMDB Rating'] = webSeries['IMDB Rating'].fillna(-1)

webSeries['Streaming Platform'] = webSeries['Streaming Platform'].fillna('-1')
#creating the plot of number of movies released each year and the cumsum of that calculation

moviesReleasedEachYear = webSeries['Year Released'].value_counts().sort_index()

cumsumMoviesReleased = webSeries['Year Released'].value_counts().sort_index().cumsum()



trace1 = go.Bar(x = list(moviesReleasedEachYear.index), y = list(moviesReleasedEachYear.values),name = 'Movies Released Each Year')

trace2 = go.Scatter(x = list(cumsumMoviesReleased.index), y = list(cumsumMoviesReleased.values), mode='lines',name = 'Trendline of Number of Movies')



fig = plt.subplots.make_subplots(specs=[[{"secondary_y": True}]])

fig.add_trace(trace1, secondary_y=False)

fig.add_trace(trace2, secondary_y=True)



fig.update_layout(template = 'plotly_white',margin=dict(l=80, r=80, t=25, b=10),

                  title = { 'text' : '<b>Web Series over the Years</b>', 'x' : 0.5},

                 font_family = 'Fira Code',title_font_color= '#ff0d00', showlegend = False)

fig.show()
#creating plots of rotten tomatoes rating along each content rating

fig = px.scatter(webSeries[(webSeries['R Rating'] != -1) & (webSeries['Content Rating'] != '-1')], x = 'Year Released',

                 y = 'R Rating', facet_col = 'Content Rating', trendline = 'ols', opacity=0.7)

fig.update_layout(template = 'plotly_dark',

                  title = { 'text' : '<b>Content Rating and Rotten Tomatoes Rating</b>', 'x' : 0.5},

                 font_family = 'Fira Code',title_font_color= '#72bcd4', showlegend = False)

fig.show()
#creating plots of imdb rating along each content rating

fig = px.scatter(webSeries[(webSeries['IMDB Rating'] != -1) & (webSeries['Content Rating'] != '-1')], x = 'Year Released',

                 y = 'IMDB Rating', facet_col = 'Content Rating', trendline = 'ols', opacity=0.7)

fig.update_layout(template = 'plotly_dark',

                  title = { 'text' : '<b>Content Rating and IMDB Rating</b>', 'x' : 0.5},

                 font_family = 'Fira Code',title_font_color= '#72bcd4', showlegend = False)

fig.show()
initial_cols = webSeries.columns #the columns we had at the start of the analysis



#using MultilabelBinarizer to create a column for each genre

from sklearn.preprocessing import MultiLabelBinarizer

mlb = MultiLabelBinarizer()

webSeries['GenreList'] = webSeries['Genre'].apply(lambda x: [y.strip() for y in x.split(',')])



mlb.fit(webSeries['GenreList'])

#creating columns = the classes of the multilabelbinarizer

webSeries[mlb.classes_] = mlb.transform(webSeries['GenreList'])
#printing the different genres that are there in the dataset

print(f'The listed genres are {mlb.classes_}')
#selecting only a those genres that make sense



#genres like the year in which the movie was released are not actually genres

selected_genres = ['Action & Adventure', 'Animation','Anime','Biography','Children','Comedy','Crime','Cult','DIY', 'Documentary',

                      'Drama','Family','Fantasy','Food','Game Show','History','Home & Garden','Horror',

                      'Lifetime','Musical','Mystery','Pet','Reality','Romance','Science','Science-Fiction','Showtime', 'Sport',

                      'Stand-up & Talk','Thriller','Travel']

selected_cols = list(initial_cols)

selected_cols.extend(selected_genres) #adding these new genres to the list of columns

webSeries = webSeries[selected_cols]
seriesPerCategory = webSeries[selected_genres].apply(sum, axis = 0) #calculating the number of series in each category



#selecting colours for the bars in the bar chart

colors = ['lightslategray',] * len(selected_genres)

"""

The indexes used correspond to different genres as identified from the the list selected_genres

0 -> Action and Adventure

1 -> Animation

5 -> Comedy

9 -> Documentary

10 -> Drama

22 -> Reality

"""

for idx in [0,1,5,9,10,22]:

    colors[idx] = 'slateblue'

    



trace = go.Bar(x = seriesPerCategory.index, y = seriesPerCategory.values,marker_color = colors)

fig = go.Figure([trace])

fig.update_layout(template = 'plotly_white',

                  title = { 'text' : '<b>Number of Web Series in Each Genre</b>', 'x' : 0.5},

                  font_family = 'Fira Code',title_font_color= '#72bcd4', 

                  showlegend = False)

fig.show()
#calculating the overall mean imdb rating and the overall rotten tomatoes rating

overall_mean_imdb = webSeries[webSeries['IMDB Rating'] != -1]['IMDB Rating'].mean()

overall_mean_r = webSeries[webSeries['R Rating'] != -1]['R Rating'].mean()
mean_imdb_rating = {}

mean_r_rating = {}



colors_imdb = ['lightslategray'] * len(selected_genres)

colors_r = ['lightslategray'] * len(selected_genres)



for i,genre in enumerate(selected_genres):

    #calculating the mean imdb and rotten tomatoes rating for each genre

    mean_imdb_rating[genre] = webSeries[(webSeries[genre] == 1 )& (webSeries['IMDB Rating'] != -1)]['IMDB Rating'].mean()

    mean_r_rating[genre] = webSeries[(webSeries[genre] == 1 )& (webSeries['R Rating'] != -1)]['R Rating'].mean()

    

    #if the mean rating for that genre is more than the overall rating, I change the color of the bar to rosybrown to make it stand out

    if(mean_imdb_rating[genre] > overall_mean_imdb):

        colors_imdb[i] = 'rosybrown'

    if(mean_r_rating[genre] > overall_mean_r):

        colors_r[i] = 'rosybrown'
#creating subplots to plot for both IMDB ratings as well as Rotten Tomatoes rating. 

fig = plt.subplots.make_subplots(rows = 1, cols = 2,

                                 horizontal_spacing=0.15, 

                                 subplot_titles=['IMDB Ratings','Rotten Tomatoes Rating'])



#adding graph for IMDB rating for each genre

trace_imdb = go.Bar(x = list(mean_imdb_rating.values()), y = list(mean_imdb_rating.keys()), 

                    name = 'Mean IMDB Rating', orientation = 'h',

                    marker_color = colors_imdb)

#adding the mean imdb rating to the graph so that we compare individual genres to this threshold

trace_mean_imdb = go.Scatter(y = list(mean_imdb_rating.keys()), x = [overall_mean_imdb]*len(mean_imdb_rating),

                             name = 'Overall Mean IMDB Rating',

                             mode = 'lines', line = {'color' : '#ffb6c1'})



#adding graph for Rotten Tomatoes rating for each genre

trace_r = go.Bar(x = list(mean_r_rating.values()), y = list(mean_r_rating.keys()),

                 name = "Mean Rotten Tomattoes Rating", orientation = 'h', 

                 marker_color = colors_r)

#adding the mean rotten tomatoes rating to the graph so that we compare individual genres to this threshold

trace_mean_r = go.Scatter(y = list(mean_r_rating.keys()), x = [overall_mean_r]*len(mean_r_rating), 

                          name = 'Overall Mean Rotten Tomatoes Rating',

                          mode = 'lines', line = {'color' : '#ffb6c1'})



#adding each trace to the their respective subplots

fig.add_trace(trace_imdb, row = 1, col = 1)

fig.add_trace(trace_mean_imdb, row = 1, col = 1)

fig.add_trace(trace_r, row = 1, col = 2)

fig.add_trace(trace_mean_r, row = 1, col = 2)



fig.update_layout(template = 'plotly_white',

                  height = 800, margin=dict(l=80, r=80, t=50, b=20),

                  title = { 'text' : '<b>Ratings in Each Genre</b>', 'x' : 0.5},

                  font_family = 'Fira Code',title_font_color= 'crimson', 

                  showlegend = False)

fig.show()
mlb = MultiLabelBinarizer()

webSeries['StreamingPlatformList'] = webSeries['Streaming Platform'].apply(lambda x: [y.strip() for y in x.split(',')])

mlb.fit(webSeries['StreamingPlatformList'])

webSeries[mlb.classes_] = mlb.transform(webSeries['StreamingPlatformList'])
selected_streaming = list(mlb.classes_)

selected_streaming.remove('-1')

selected_streaming.remove('DIY')

selected_streaming.remove('History')

print(f'The streaming platforms are {selected_streaming}')
seriesPerPlatform = webSeries[selected_streaming].apply(sum, axis = 0)

colors = ['gray'] * len(selected_streaming)

for idx in [27,33,34,41,44,61]:

    colors[idx] = '#8d81d9'

trace = go.Bar(x = seriesPerPlatform.index, y = seriesPerPlatform.values,marker_color = colors)

fig = go.Figure([trace])

fig.update_layout(template = 'ggplot2',margin=dict(l=80, r=80, t=25, b=10),

                  title = { 'text' : '<b>Number of Web Series on Each Streaming Platform</b>', 'x' : 0.5},

                 font_family = 'Fira Code',title_font_color= 'blue', showlegend = False)

fig.show()
mean_imdb_rating = {}

mean_r_rating = {}

for platform in selected_streaming:

    mean_imdb_rating[platform] = webSeries[(webSeries[platform] == 1 )& (webSeries['IMDB Rating'] != -1)]['IMDB Rating'].mean()

    mean_r_rating[platform] = webSeries[(webSeries[platform] == 1 )& (webSeries['R Rating'] != -1)]['R Rating'].mean()

color = ['gray']*len(selected_streaming)

for idx in [27,33,34,41,44,61]:

    color[idx] = '#8fa6bc'
fig = plt.subplots.make_subplots(rows = 1, cols = 2, horizontal_spacing=0.15, subplot_titles=['IMDB Ratings','Rotten Tomatoes Rating'])

trace_imdb = go.Bar(x = list(mean_imdb_rating.values()), y = list(mean_imdb_rating.keys()), name = 'Mean IMDB Rating', orientation = 'h', marker_color = color)

trace_mean_imdb = go.Scatter(y = list(mean_imdb_rating.keys()), x = [overall_mean_imdb]*len(mean_imdb_rating), name = 'Overall Mean IMDB Rating',mode = 'lines')

trace_r = go.Bar(x = list(mean_r_rating.values()), y = list(mean_r_rating.keys()), name = "Mean Rotten Tomattoes Rating", orientation = 'h', marker_color = color)

trace_mean_r = go.Scatter(y = list(mean_r_rating.keys()), x = [overall_mean_r]*len(mean_r_rating), name = 'Overall Mean Rotten Tomatoes Rating',mode = 'lines')

fig.add_trace(trace_imdb, row = 1, col = 1)

fig.add_trace(trace_mean_imdb, row = 1, col = 1)

fig.add_trace(trace_r, row = 1, col = 2)

fig.add_trace(trace_mean_r, row = 1, col = 2)

fig.update_layout(template = 'plotly_white',height = 1000,margin=dict(l=80, r=80, t=50, b=20),

                  title = { 'text' : '<b>Ratings in Each Streaming Platform</b>', 'x' : 0.5},

                 font_family = 'Fira Code',title_font_color= 'crimson', showlegend = False)

fig.show()
major_platforms = ['Prime Video', 'Netflix', 'Hulu', 'fuboTV','Hoopla','Funimation']

major_genres = ['Action & Adventure', 'Animation','Comedy', 'Documentary', 'Drama', 'Reality']
fig = plt.subplots.make_subplots(rows = 2, cols = 3, subplot_titles = major_platforms,

                                horizontal_spacing=0.15,

                                vertical_spacing=0.2)

for i, platform in enumerate(major_platforms):

    genreCounts = {}

    for genre in major_genres:

        genreCounts[genre] = webSeries[(webSeries[genre] == 1) & webSeries[platform] == 1].shape[0]

    trace = go.Bar(y = list(genreCounts.keys()), x = list(genreCounts.values()),name = platform, orientation = 'h')

    fig.add_trace(trace, row = (i//3) + 1, col = (i%3) + 1)

fig.update_layout(showlegend = False)

fig.update_layout(template = 'presentation',

                  title = { 'text' : '<b>Number of Web Series of Major Genres in Major Streaming Platform</b>', 'x' : 0.5},

                 font_family = 'Fira Code',title_font_color= 'black', showlegend = False)

fig.show()
fig = plt.subplots.make_subplots(rows = 2, cols = 3, subplot_titles = major_platforms,

                                horizontal_spacing=0.15,

                                vertical_spacing=0.2)

for i, platform in enumerate(major_platforms):

    genreCounts = {}

    for genre in major_genres:

        genreCounts[genre] = webSeries[(webSeries[genre] == 1) & (webSeries[platform] == 1) & (webSeries['IMDB Rating'] != -1)]['IMDB Rating'].mean()

    trace = go.Bar(y = list(genreCounts.keys()), x = list(genreCounts.values()),name = platform, orientation = 'h')

    fig.add_trace(trace, row = (i//3) + 1, col = (i%3) + 1)

fig.update_layout(showlegend = False)

fig.update_layout(template = 'presentation',

                  title = { 'text' : '<b>Average IMDB Rating of Major Genres in Major Streaming Platform</b>', 'x' : 0.5},

                 font_family = 'Fira Code',title_font_color= 'black', showlegend = False)

fig.show()
fig = plt.subplots.make_subplots(rows = 2, cols = 3, subplot_titles = major_platforms,

                                horizontal_spacing=0.15,

                                vertical_spacing=0.2)

for i, platform in enumerate(major_platforms):

    genreCounts = {}

    for genre in major_genres:

        genreCounts[genre] = webSeries[(webSeries[genre] == 1) & (webSeries[platform] == 1) & (webSeries['R Rating'] != -1)]['R Rating'].mean()

    trace = go.Bar(y = list(genreCounts.keys()), x = list(genreCounts.values()),name = platform, orientation = 'h')

    fig.add_trace(trace, row = (i//3) + 1, col = (i%3) + 1)

fig.update_layout(showlegend = False)

fig.update_layout(template = 'presentation',

                  title = { 'text' : '<b>Average Rotten Tomatoes Rating of Major Genres in Major Streaming Platform</b>', 'x' : 0.5},

                 font_family = 'Fira Code',title_font_color= 'black', showlegend = False)

fig.show()
def extract_num_of_seasons(seasons):

    numOfSeasons = re.findall(r'\d+', seasons)[0]

    return int(numOfSeasons)



webSeries['Seasons'] = webSeries['No of Seasons'].apply(lambda x:extract_num_of_seasons(x))
seasonsCounts = webSeries['Seasons'].value_counts().sort_index()

trace = go.Scatter(x = seasonsCounts.index, y = seasonsCounts.values, mode = 'lines+markers')

fig = go.Figure([trace])

fig.update_layout(template = 'presentation',

                  title = { 'text' : '<b>Number of Series with given Number of Seasons</b>', 'x' : 0.5},

                  font_family = 'Fira Code',title_font_color= 'black',

                  xaxis_title="Number of Seasons", yaxis_title="Number of Webseries(in log Scale)",

                  showlegend = False)

fig.update_yaxes(type="log")

fig.show()
webSeries.sort_values(by = 'Seasons', ascending=False)[['Series Title','Year Released','Genre','IMDB Rating', 'R Rating','Streaming Platform','Seasons']].head(10).reset_index(drop = True)
seasonMeanIMDB = {}

seasonMeanR = {}



for i in range(1,21):

    seasonDfIMDB = webSeries[(webSeries['Seasons'] == i) & (webSeries['IMDB Rating'] != -1)]

    seasonMeanIMDB[i] = seasonDfIMDB['IMDB Rating'].mean()

    

    seasonDfR = webSeries[(webSeries['Seasons'] == i) & (webSeries['R Rating'] != -1)]

    seasonMeanR[i] = seasonDfR['R Rating'].mean()
trace1 = go.Scatter(x = list(seasonMeanIMDB.keys()), y = list(seasonMeanIMDB.values()), mode = 'lines+markers',name = 'IMDB Ratings Across Season')

trace2 = go.Scatter(x = list(seasonMeanR.keys()), y = list(seasonMeanR.values()), mode = 'lines+markers',name = 'Rotten Tomatoes Ratings Across Season')

fig = plt.subplots.make_subplots(specs=[[{"secondary_y": True}]])

fig.add_trace(trace1, secondary_y=False)

fig.add_trace(trace2, secondary_y=True)

fig.update_layout(template = 'presentation',

                  title = { 'text' : '<b>Mean Rating Across Seasons</b>', 'x' : 0.5},

                  font_family = 'Fira Code',title_font_color= 'black',

                  xaxis_title="Number of Seasons", 

                  showlegend = False)

fig.show()
seasonsAcrossGenres = {}

for genre in selected_genres:

    df = webSeries[webSeries[genre] == 1]

    seasonsAcrossGenres[genre] = int(df['Seasons'].mean())
colors = ['#6aaa96']*len(seasonsAcrossGenres)

fig = px.bar(pd.DataFrame({'Genre' : list(seasonsAcrossGenres.keys()), 'Average Num of Seasons' : list(seasonsAcrossGenres.values())}), 

             x = 'Genre',y = 'Average Num of Seasons', color_discrete_sequence = colors)

fig.update_layout(template = 'plotly_white',

                  title = { 'text' : '<b>Average Number of Seasons in each Genre</b>', 'x' : 0.5},

                  font_family = 'Fira Code',title_font_color= 'black',

                  showlegend = False)

fig.show()
webSeries['Description'] = webSeries['Description'].apply(lambda x: str(x).lower())
stopwords = set(STOPWORDS).union(set(['episode','episodes','season','seasons','one', 'imdb']))

def createCorpus(genre):

    df = webSeries[webSeries[genre] == 1]

    corpus = ""

    for des in df['Description'].to_list():

        corpus += des[:100]

    return corpus



def generateWordCloud(genre, mask = None):

    mplt.subplots(figsize=(12,8))

    corpus = createCorpus(genre)

    wordcloud = WordCloud(background_color='White',

                          mask = mask,

                          contour_color='orange', contour_width=4, 

                          stopwords=stopwords,

                          width=1500, margin=10,

                          height=1080

                         ).generate(corpus)

    mplt.imshow(wordcloud)

    mplt.axis('off')

    mplt.show()
response = requests.get('https://d2gg9evh47fn9z.cloudfront.net/800px_COLOURBOX3687345.jpg')

img = Image.open(BytesIO(response.content))

generateWordCloud('Action & Adventure', mask = np.asarray(img))
response = requests.get('https://i.pinimg.com/originals/76/47/9d/76479dd91dc55c2768ddccfc30a4fbf5.png')

img = Image.open(BytesIO(response.content))

generateWordCloud('Animation',mask = np.asarray(img))
response = requests.get('https://cached.imagescaler.hbpl.co.uk/resize/scaleWidth/815/cached.offlinehbpl.hbpl.co.uk/news/OMC/mrbeanthumb-20170717092605928.jpg')

img = Image.open(BytesIO(response.content))

generateWordCloud('Comedy',mask = np.asarray(img))
response = requests.get('https://banner2.cleanpng.com/20180217/jde/kisspng-earth-clip-art-internet-animation-cliparts-5a88c5d9c593d5.2603841115189129858093.jpg')

img = Image.open(BytesIO(response.content))

generateWordCloud('Documentary',mask = np.asarray(img))
response = requests.get('https://i7.pngguru.com/preview/544/942/462/drama-theatre-comedy-tragedy-mask-actor.jpg')

img = Image.open(BytesIO(response.content))

generateWordCloud('Drama',mask = np.asarray(img))
response = requests.get('http://clipart-library.com/img/865458.jpg')

img = Image.open(BytesIO(response.content))

generateWordCloud('Reality',mask = np.asarray(img))


fig = plt.subplots.make_subplots(rows = 3, cols = 2, specs = [[{"type":"table"}] * 2]*3,

                                subplot_titles = major_platforms,

                                horizontal_spacing=0.03,vertical_spacing = 0.05)

for i, platform in enumerate(major_platforms):

    df = webSeries[webSeries[platform] == 1].sort_values(by = 'IMDB Rating', ascending = False)

    trace = go.Table(header = dict(values = ['<b>Series Title</b>','<b>IMDB Rating</b>']), cells = dict(values = [df['Series Title'][:3], df['IMDB Rating'][:3]]))

    fig.add_trace(trace, row = (i//2)+1 , col = (i%2)+1)

fig.update_layout(height = 500,margin=dict(l=80, r=80, t=100, b=20),

                  title = { 'text' : '<b>Top 3 IMDB Rated Web Series on Major Streaming Services</b>', 'x' : 0.5},)

fig.show()
fig = plt.subplots.make_subplots(rows = 3, cols = 2, specs = [[{"type":"table"}] * 2]*3,

                                subplot_titles = major_platforms,

                                horizontal_spacing=0.03,

                                vertical_spacing=0.05)

for i, platform in enumerate(major_platforms):

    df = webSeries[webSeries[platform] == 1].sort_values(by = 'R Rating', ascending = False)

    trace = go.Table(header = dict(values = ['<b>Series Title</b>','<b>R Rating</b>']), cells = dict(values = [df['Series Title'][:3], df['R Rating'][:3]]))

    fig.add_trace(trace, row = (i//2)+1 , col = (i%2)+1)

fig.update_layout(height = 550,margin=dict(l=80, r=80, t=100, b=20),

                  title = { 'text' : '<b>Top 3 Rotten Tomatoes Rated Web Series on Major Streaming Services</b>', 'x' : 0.5},)

fig.show()
fig = plt.subplots.make_subplots(rows = 3, cols = 2, specs = [[{"type":"table"}] * 2]*3,

                                subplot_titles = major_genres,

                                horizontal_spacing=0.03,

                                vertical_spacing=0.1)

for i, genre in enumerate(major_genres):

    df = webSeries[webSeries[genre] == 1].sort_values(by = 'IMDB Rating', ascending = False)

    trace = go.Table(header = dict(values = ['<b>Series Title</b>','<b>IMDB Rating</b>']), cells = dict(values = [df['Series Title'][:3], df['IMDB Rating'][:3]]))

    fig.add_trace(trace, row = (i//2)+1 , col = (i%2)+1)

fig.update_layout(height = 500,margin=dict(l=80, r=80, t=100, b=20),

                  title = { 'text' : '<b>Top 3 IMDB Rated Web Series for Major Genres</b>', 'x' : 0.5},)

fig.show()
fig = plt.subplots.make_subplots(rows = 3, cols = 2, specs = [[{"type":"table"}] * 2]*3,

                                subplot_titles = ['Action & Adventure', 'Animation', 'Comedy', 'Documentary','Drama','Reality'],

                                horizontal_spacing=0.03,

                                vertical_spacing=0.1)

for i, genre in enumerate(major_genres):

    df = webSeries[webSeries[genre] == 1].sort_values(by = 'R Rating', ascending = False)

    trace = go.Table(header = dict(values = ['<b>Series Title</b>','<b>R Rating</b>']), cells = dict(values = [df['Series Title'][:3], df['R Rating'][:3]]))

    fig.add_trace(trace, row = (i//2)+1 , col = (i%2)+1)

fig.update_layout(height = 500,margin=dict(l=80, r=80, t=100, b=20),

                  title = { 'text' : '<b>Top 3 Rotten Tomatoes Rated Web Series for Major Genres</b>', 'x' : 0.5},)

fig.show()