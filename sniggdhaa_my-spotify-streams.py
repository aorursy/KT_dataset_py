import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re
import matplotlib.style as style
style.use('seaborn-poster')
sns.set_style('darkgrid')
#Getting the data
mydata = pd.read_csv('../input/mydata/spotify_data.csv', index_col=0,parse_dates=['Date'])
btsdata = pd.read_csv('../input/mydata/bts.csv', index_col='index')
mydata['trackName']= mydata['trackName'].map(lambda x: x.lower())
mydata['minPlayed']= mydata['msPlayed'].map(lambda x: x/60000)
mydata['hrsPlayed']= mydata['msPlayed'].map(lambda x: x/3600000)
mydata.head()
btsdata.head()
#Dropping unnecessary columns
btsdata.drop(columns={'time_signature','key','artist_name'}, inplace=True)
btsdata['song_name']= btsdata['song_name'].map(lambda x: x.lower())
btsdata['duration_min']= btsdata['duration_ms'].map(lambda x: x/60000)
btsdata.head()
#Number or artists discovered during Sept 2019-Sept 2020
len(mydata.artistName.unique())
#Number or artists discovered during March 2020-Sept 2020
pre_artist = mydata[mydata.Date < '2020-03-01'].artistName.unique()
len(mydata[mydata.Date > '2020-03-01'][~mydata.artistName.isin(pre_artist)].artistName.unique())
def plotMean(data, mycolor, mylinestyle):
    plt.axvline(np.mean(data), color=mycolor, linestyle=mylinestyle, linewidth=1.5, label='Mean({})'.format(round(np.mean(data),2)))
    plt.legend(loc='best')
def plotMedian(data, mycolor, mylinestyle):
    plt.axvline(np.median(data), color=mycolor, linestyle=mylinestyle, linewidth=1.5, label='Median({})'.format(round(np.median(data),2)))
    plt.legend(loc='best')
def plotLabel(data,x):
    plt.annotate("Count: {}".format(round(data.max(),2)), (x, data.max()),bbox=dict(fc='yellow'))
def plotBar(data,palette):
    sns.barplot(x=data,y=data.keys(),palette = palette)
    plotMean(data,'r','-')
    plotMedian(data,'g','--')
    plt.show()
data = mydata.groupby(['Date','trackName'], as_index = False).size().groupby('Date').size()
data.plot.line()
plotLabel(data,'2020-01-19')
plt.title('Number of tracks streamed over time', fontweight='bold')
plt.ylabel('Number of tracks')
plt.show()
data = mydata.groupby(['Date','hrsPlayed'])['hrsPlayed'].sum().groupby('Date').sum()
data.plot.line()
plotLabel(data,'2020-01-19')
plt.title('Number of hours streamed over time', fontweight='bold')
plt.ylabel('Hours played')
plt.show()
plotBar(mydata[mydata.Date < '2020-03-01'].trackName.value_counts()[:15],'inferno')
plotBar(mydata[mydata.Date > '2020-03-01'].trackName.value_counts()[:15], 'inferno')
mydata.groupby(['artistName'])['hrsPlayed'].sum().sort_values(ascending=False)[:15].plot.pie(figsize=(10,10), autopct='%1.0f%%')
plt.title('Top 15 artists based on hours played in percentage', fontweight='bold')
plt.ylabel('')
plt.show()
plt.figure(figsize=(15,8))
data = mydata[mydata.artistName.isin(['BTS','V','RM','BTSYOUNG4EVER'])].groupby(['Date','minPlayed'])['minPlayed'].sum().groupby('Date').sum()
plt.scatter(x=data.keys(), y=data,c=data, cmap='autumn_r',s= 250, edgecolors='black')
plt.ylabel('Minutes played')
plt.show()
plotBar(mydata[mydata.Date < '2020-07-01'].groupby(['artistName'])['hrsPlayed'].sum().sort_values(ascending=False)[:20],'viridis')
plotBar(mydata[mydata.Date > '2020-07-01'].groupby(['artistName'])['hrsPlayed'].sum().sort_values(ascending=False)[:20],'viridis')
plotBar(mydata[mydata.artistName.isin(['BTS','V','RM','BTSYOUNG4EVER'])].groupby(['trackName'])['minPlayed'].sum().sort_values(ascending=False)[:20],'nipy_spectral')
from wordcloud import WordCloud 
cloud=''
for x in mydata['artistName'].unique():
    x= x.replace(" ", "")
    cloud+= ''.join(x) +' '
plt.figure(figsize=(12,8))
wordcloud = WordCloud(background_color='white',max_font_size=50).generate(cloud)
plt.imshow(wordcloud)
plt.axis('off')
plt.title('ARTISTS', fontweight='bold')
plt.show()
#My preferred values of audio features
preference = pd.DataFrame()
def plotFeatures(feature):
    data = btsdata[btsdata.song_name.isin(mydata[mydata.msPlayed>1].trackName.tolist())].groupby(['song_name'])[feature].mean().sort_values(ascending=False)
    sns.barplot(x=data,y=data.keys(),palette = 'gnuplot')
    plotMean(data,'r','--') 
    preference['{}'.format(feature)]= [round(np.mean(data),2)]
    plt.show()
sns.heatmap(btsdata.corr(), annot=True, center=1)
plt.show()
sns.lmplot(x='acousticness',y='energy',data=btsdata, height=7,line_kws={'color': 'red'})
plt.title('Acousticness - Energy', fontweight='bold')
plt.show()
plotFeatures('danceability')
plotFeatures('energy')
plotFeatures('speechiness')
plotFeatures('acousticness')
plotFeatures('instrumentalness')
plotFeatures('liveness')
plotFeatures('tempo')
plotFeatures('valence')
preference
kpop = pd.read_csv('../input/mydata/kpop.csv', index_col=0)
kpop['song_name']= kpop['song_name'].map(lambda x: x.strip().lower())
kpop.head()
#Features
kpop_features = kpop.loc[:,['acousticness','danceability','energy','instrumentalness','liveness','speechiness','tempo', 'valence']]
kpop_features.head()
from sklearn.metrics.pairwise import euclidean_distances
kpop['Similarity'] = euclidean_distances(kpop_features, preference.to_numpy()).squeeze()
kpop.sort_values(by= 'Similarity', inplace= True)
similar = kpop[['artist_name', 'song_name', 'Similarity']]
similar = similar.drop_duplicates(subset=['artist_name', 'song_name'])
similar.head(20)