import pandas as pd

spotify=pd.read_csv('/kaggle/input/top50spotify2019/top50.csv',encoding='latin_1') #we need the latin encoding for pandas to read it correctly

spotify.head()
#Basic information

spotify.info()
#baisc statistics about the dataset

spotify.describe()
#features of the dataset

spotify.columns
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
#plotting the most popular songs

plt.figure(figsize=(9,25))

sns.barplot(data=spotify,x='Popularity',y='Track.Name')

plt.show()
#plotting the most songs by genre

sns.catplot(y='Genre',kind='count',

           edgecolor="0.5",data=spotify)

plt.show()
#which artist has the most songs in the Top 50 playlist?

sns.catplot(y='Artist.Name',kind='count',

           edgecolor="0.6",data=spotify)

plt.show()
#what is the optimal bpm for a song?

plt.figure(figsize=(25,9))

sns.countplot(data=spotify,x='Beats.Per.Minute')

plt.title('Beats/min')

plt.show()
#Energy, Danceability and Loudness

#2D Plotting



#Energy

plt.figure(figsize=(8,4))

sns.distplot(spotify['Energy'],kde=False,bins=15,color='red')

plt.title('Energy')

plt.show()



#Danceability

plt.figure(figsize=(8,4))

sns.distplot(spotify['Danceability'],kde=False,bins=15,color='blue')

plt.title('Danceability')

plt.show()



#Loudness

plt.figure(figsize=(8,4))

sns.distplot(spotify['Loudness..dB..'],kde=False,bins=15,color='green')

plt.title('Loudness')

plt.xlabel('Loudness(dB)')

plt.show()

#3D Plotting of Energy, Danceability and Loudness (with Plotly)

import plotly.graph_objects as go



#generate charts

fig = go.Figure(data = [go.Scatter3d(

    x = spotify['Energy'],

    y = spotify['Danceability'],

    z = spotify['Loudness..dB..'],

    text = spotify['Track.Name'],  ## Additional texts which will be shown

    mode = 'markers',

    marker = dict(

    color = spotify['Popularity'],

    colorbar_title = 'Popularity',

    colorscale = 'blues'

    )

)])



#set variables and size

fig.update_layout(width=800, height=800, title = 'Energy, Danceability & Loudness of Songs',

                  scene = dict(xaxis=dict(title='Energy'),

                               yaxis=dict(title='Danceability'),

                               zaxis=dict(title='Loudness')

                               )

                 )



fig.show()

#relating speechiness and acousticness with popularity

plt.figure(figsize=(12,6))

sns.violinplot(x='Speechiness.',y='Popularity',data=spotify)

plt.xlabel('Speechiness')

plt.ylabel('Popularity')

plt.title('Speechiness vs Popularity')

plt.show()



plt.figure(figsize=(10,10))

sns.despine(offset=10,left=True)

sns.jointplot(data=spotify,

             x='Acousticness..',

             y='Popularity',

             kind='kde',

             space=1)



plt.title('Acousticness vs Popularity')

plt.show()
#looking at the length of songs

import plotly.express as px



fig=px.line(spotify,y='Length.',x='Track.Name',title='Length distribution of songs')

fig.show()
#plotting correlation between all variables

plt.figure(figsize=(15,15))

plt.title('Correlation between all variables')

sns.heatmap(data=spotify.corr(),

           annot=True,

           square=True,

           linewidths=1)

plt.show()
#all histograms with data columns

sns.pairplot(spotify)

plt.plot()

plt.show()