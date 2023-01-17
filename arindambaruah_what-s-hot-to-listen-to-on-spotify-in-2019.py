import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import plotly.express as px

import plotly.graph_objects as go
df=pd.read_csv('../input/top50spotify2019/top50.csv',encoding='latin_1',index_col=0)

df.head()
df['Count']=1

df_artist=df.groupby('Artist.Name')['Count'].sum().reset_index().sort_values(by='Count',ascending=False)

df_artist.head(10)
sns.catplot('Artist.Name','Count',data=df_artist.head(10),kind='bar',height=8,aspect=2,palette='winter')

plt.title('Top 10 artists of 2019',size=25)

plt.xlabel('Artist name',size=15)

plt.ylabel('Number of songs in top 50',size=15)

plt.xticks(size=15,rotation=45)
df_genre=df.groupby('Genre')['Count'].sum().reset_index().sort_values(by='Count',ascending=False)

df_genre.head()
fig=px.pie(df_genre,values='Count',names='Genre',hole=0.4)

fig.update_layout(title='Top genres of 2019',title_x=0.45,annotations=[dict(text='Genre',font_size=20, showarrow=False,height=800,width=700)])

fig.update_traces(textfont_size=15,textinfo='percent')

fig.show()
df_top_tracks=df.sort_values(by='Popularity',ascending=False).head(10).reset_index(drop=True)

df_top_tracks.index=df_top_tracks.index + 1

df_top_tracks
sns.set()

df_top_tracks.plot.area(y=['Beats.Per.Minute', 'Energy',

       'Danceability', 'Loudness..dB..', 'Liveness', 'Valence.',

       'Acousticness..', 'Speechiness.'],alpha=0.4,figsize=(15,12),stacked=True)

plt.title('Technical variations of top 10 songs',size=20)

plt.xlabel('Track number')



sns.set(style='darkgrid')

fig2=plt.figure(figsize=(10,8))

ax1=fig2.add_subplot(221)

sns.boxplot('Beats.Per.Minute',data=df_top_tracks,orient='v',ax=ax1)



ax2=fig2.add_subplot(222)

sns.boxplot('Valence.',data=df_top_tracks,orient='v',ax=ax2,color='indianred')



ax3=fig2.add_subplot(223)

sns.boxplot('Acousticness..',data=df_top_tracks,orient='v',ax=ax3,color='green')



ax4=fig2.add_subplot(224)

sns.boxplot('Speechiness.',data=df_top_tracks,orient='v',ax=ax4,color='yellow')
plt.figure(figsize=(10,8))

corr_songs=df.iloc[:,:-1].corr()

sns.heatmap(corr_songs,annot=True,fmt='.2f',cmap='rocket')
fig3=px.line(df,y='Length.',x='Track.Name',height=800, width=1000)

fig3.update_layout(title='Length of top songs',title_x=0.5,plot_bgcolor="black")



fig3.update_xaxes(visible=False)

fig3.show()
sns.jointplot(x='Loudness..dB..',y='Energy',data=df,kind='kde',color='green')
sns.jointplot(x='Beats.Per.Minute',y='Speechiness.',data=df,kind='kde',color='red')