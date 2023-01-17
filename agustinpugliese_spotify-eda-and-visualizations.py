import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.figure_factory as ff
import plotly.express as px
from PIL import Image
from wordcloud import WordCloud

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

%matplotlib inline
dataset = pd.read_csv('../input/top50spotify2019/top50.csv', encoding = "ISO-8859-1")
dataset.head()
dataset = dataset.drop(['Unnamed: 0'], axis = 1)
dataset.columns = ['Track_Name', 'Artist_Name', 'Genre', 'BPM', 'Energy', 'Danceability', 'dB', 'Liveness','Valence',
                  'Length(s)', 'Acousticness','Speechiness','Popularity']
hist_data = [dataset['Danceability'], dataset['Popularity'], dataset['Acousticness']]
group_labels = [ 'Danceability','Popularity','Acousticness']

fig = ff.create_distplot(hist_data, group_labels, bin_size=[10, 10,1], colors = ['#F66095', '#2BCDC1', '#393E46'])
fig.update_layout(title_text='Popularity, Danceability and Acousticness')
fig.show()
Artists = pd.DataFrame(dataset['Artist_Name'].value_counts()).reset_index()
Artists.columns = ['Artist','Total songs']
fig = px.bar(Artists, x = 'Artist', y = 'Total songs', color = 'Artist', title='Artist vs amount of top songs')        
fig.show()
Genre = pd.DataFrame(dataset['Genre'].value_counts()).reset_index()
Genre.columns = ['Genre','Total songs']
fig = px.bar(Genre, x = 'Genre', y = 'Total songs', color = 'Genre', title='Genre vs amount of top songs')        
fig.show()
plt.figure(figsize=(12,8))

sns.boxplot(x = "Popularity", y = "Genre", data=dataset, palette = 'Set1')

plt.xlabel('Popularity', fontsize=16)
plt.ylabel('Genre', fontsize=16)
plt.yticks(rotation='0')
plt.title("Genre vs popularity", fontsize=20)
plt.show()
x = dataset.columns[3:]
heat = go.Heatmap(z =dataset.corr(),
                  x = x,
                  y = x,
                  xgap=1, ygap=1,
                  colorbar_thickness=20,
                  colorbar_ticklen=3,
                  hovertext = dataset.corr(),
                  hoverinfo='text',colorscale=[[0.0, '#F5FFFA'], 
                         [0.2, '#ADD8E6'], 
                         [0.4, '#87CEEB'],
                         [0.6, '#87CEFA'], 
                         [0.8, '#40E0D0'], 
                         [1.0, '#00CED1']]
                   )

title = 'Correlation Matrix'               

layout = go.Layout(title_text=title, title_x=0.5, 
                   width=600, height=600,
                   xaxis_showgrid=False,
                   yaxis_showgrid=False,
                   yaxis_autorange='reversed')
   
fig=go.Figure(data=[heat], layout=layout)        
fig.show() 
fig = px.scatter(dataset, x = "Speechiness", y = "BPM", size='Popularity', color = "Genre", title = 'BPM vs. Speechiness')
fig.show()
fig = px.scatter(dataset, x = "Energy", y = "dB", size='Popularity', color = "Genre", title = 'Energy vs. dB')
fig.show()
plt.figure(figsize=(12,8))
sns.jointplot(x = dataset["Energy"], y = dataset['Danceability'], height=10 ,kind="kde", color='purple')

plt.ylabel('Danceability', fontsize=12)
plt.xlabel("Energy", fontsize=12)
plt.title("Energy Vs Danceability", fontsize=15)

plt.show()
hist_data = [dataset['Energy'], dataset['Danceability']]
group_labels = [ 'Energy','Danceability']

fig = ff.create_distplot(hist_data, group_labels, bin_size=[10, 10], colors = ['#D4323E', '#3466D4'])
fig.update_layout(title_text='Energy vs Danceability')
fig.show()
plt.figure(figsize=(12,8))
sns.jointplot(x = dataset["Speechiness"], y = dataset['BPM'], height=10 ,kind="kde", color='turquoise')

plt.ylabel('BPM', fontsize=12)
plt.xlabel("Speechiness", fontsize=12)
plt.title("Speechiness Vs BPM", fontsize=15)

plt.show()
hist_data = [dataset['Speechiness'], dataset['BPM']]
group_labels = [ 'Speechiness','BPM']

fig = ff.create_distplot(hist_data, group_labels, bin_size=[12, 12], colors = ['#2462ab', '#2ba323'])
fig.update_layout(title_text='Speechiness vs BPM')
fig.show()
plt.subplots(figsize=(12,8))
wordcloud = WordCloud(
                          background_color='White',max_words = 30,
                           contour_color='black', contour_width=1, 
                          width=1500, margin=10,
                          height=1080
                         ).generate(" ".join(dataset.Track_Name))
plt.imshow(wordcloud)
plt.axis('off')
plt.show()