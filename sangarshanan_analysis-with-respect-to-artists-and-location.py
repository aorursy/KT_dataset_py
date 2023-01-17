import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
data=pd.read_csv("../input/data.csv",encoding = 'utf8')
data.head()
data['Region'].unique()
dt_artist = data.loc[data['Artist'] == 'Shakira',:]
dt_loction = dt_artist.loc[dt_artist['Region'] == 'es',:]
da = dt_artist.loc[dt_artist['Region'] == 'ec',:]
fig, ax =plt.subplots(1,2)
dt_l = dt_loction.loc[dt_loction['Position'] < 11,:]
dt_r = da.loc[da['Position'] < 11,:]
sns.countplot(x = 'Position',
              data = dt_l,
              order = dt_l['Position'].value_counts().index,ax=ax[0])
sns.countplot(x = 'Position',
              data = dt_r,
              order = dt_r['Position'].value_counts().index,ax=ax[1])
ax[0].set_title('With respect to Spain')
ax[1].set_title('With respect to Ecuador')

plt.show()
Top_song = data.loc[data['Streams'] == data['Streams'].max() ,:]
Top_song
Top = data.loc[data['Position'] == 1,:]
Best = Top.groupby('Artist').size()
Best = Best[(Best.values>200)]
sns.barplot(Best.index, Best.values, alpha=0.8)
plt.xticks(rotation='vertical')
plt.xlabel('Name of the Artist', fontsize=14)
plt.ylabel('Number of #1 Hits on Chart', fontsize=14)
plt.title("Top Rated Artists", fontsize=16)
plt.show()
#POSITION - 2
Top = data.loc[data['Position'] == 2,:]
Best = Top.groupby('Artist').size()
Best = Best[(Best.values>200)]
sns.barplot(Best.index, Best.values, alpha=0.8 )
plt.xticks(rotation='vertical')
plt.xlabel('Name of the Artist', fontsize=14)
plt.ylabel('Number of #2 Hits on Chart', fontsize=14)
plt.title("Top Rated Artists with most #2 hits", fontsize=16)
plt.show()
#POSITION - 3
Top = data.loc[data['Position'] == 3,:]
Best = Top.groupby('Artist').size()
Best = Best[(Best.values>200)]
sns.barplot(Best.index, Best.values, alpha=0.8)
plt.xticks(rotation='vertical')
plt.xlabel('Name of the Artist', fontsize=14)
plt.ylabel('Number of #3 Hits on Chart', fontsize=14)
plt.title("Top Rated Artists with most #3 hits", fontsize=16)
plt.show()
sheeran = data.loc[data['Artist'] == 'Ed Sheeran',:]
colors = ['gold', 'orange','yellowgreen', 'lightcoral', 'lightskyblue']
top = sheeran['Track Name'].value_counts()[:5].index.tolist()
value =sheeran['Track Name'].value_counts()[:5].values.tolist()
plt.pie(value, labels=top, colors=colors, autopct='%1.1f%%',startangle=140)
plt.axis('equal')
plt.show()
songs = sheeran[(sheeran['Region'] == 'global') & (sheeran['Track Name'] == 'Shape of You') ]
songs['Position'].value_counts()
songs['Streams'].value_counts()
sns.lmplot(x='Streams', y='Position', data=songs,hue='Track Name')
plt.ylim(0, 10)
plt.xlim(0, None)
plt.show()
plt.figure(figsize=(10,6))
songs = sheeran[(sheeran['Track Name'] == 'Perfect') | (sheeran['Track Name'] == 'Shape of You') |  (sheeran['Track Name'] == 'Photograph') |  (sheeran['Track Name'] == 'Thinking Out Loud') | (sheeran['Track Name'] == 'Castle on the Hill') ]
sns.violinplot(x='Track Name', y='Position', data=songs)
plt.show()
cor = data.drop('URL',axis =1)
cor = cor.drop('Date',axis=1)
# Calculate correlations
corr = cor.corr()
# Heatmap
sns.heatmap(corr)
plt.show()
from datetime import datetime
data["Date2"]=data["Date"].apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))
plt.figure(figsize=(10,6))
poss = data[(data['Region'] == 'global') & (data['Track Name'] == 'Havana') ]
plt.plot(poss['Date2'],poss['Position'])
plt.title('Havana Position with respect to time')
plt.xlabel('Time')
plt.ylabel('Position')
plt.show()
plt.figure(figsize=(10,6))
poss = data[(data['Region'] == 'us') & (data['Position'] == 1)]
dat = poss['Track Name'].value_counts()
Best = dat[(dat.values>8)]
sns.barplot(Best.index, Best.values, alpha=0.8)
plt.xticks(rotation='vertical')
plt.xlabel('Song Name', fontsize=14)
plt.ylabel('Number of weeks in #1', fontsize=14)
plt.title("Hit songs in the United States", fontsize=16)
plt.show()