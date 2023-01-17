import numpy as np

import pandas as pd 



df=pd.read_csv('/kaggle/input/top-spotify-songs-from-20102019-by-year/top10s.csv',encoding='ISO-8859-1')

df=df.iloc[:,1:]

df.head()
df.isnull().any()


number_songs=df.title.nunique()

number_artists=df.artist.nunique()

number_genres=df['top genre'].nunique()



print('There are', number_songs,'songs,', number_artists,'artists and',number_genres,'genres in the dataset.')

import matplotlib.pyplot as plt

import seaborn as sb

from matplotlib import rcParams

rcParams['figure.figsize'] =12,12



sb.countplot(y=df['artist'],order=df.artist.value_counts().iloc[:10].index);



genres_piechart=plt.pie(df['top genre'].value_counts().iloc[:5],explode=[0.1,0,0,0,0] ,labels=df['top genre'].value_counts().iloc[:5].index,

autopct='%1.1f%%', shadow=True, startangle=50)
bpm_=df['bpm']

dur=df['dur']



rcParams['figure.figsize'] =17,7



#BPM

fig, (ax1,ax2) =plt.subplots(1,2)

ax1.hist(bpm_,bins=np.arange(40, 210, step=20));

ax1.set_title('Frequency plot of Beats per Minute (BPM)')

plt.sca(ax1)

plt.xticks(np.arange(40, 210, step=20))

plt.yticks(np.arange(0, 250, step=20))

ax1.set_xlabel('Beats per Minute (BPM)');

ax1.set_ylabel('Counts');



#Duration

ax2.hist(dur,bins=np.arange(120, 450, step=30));

ax2.set_title('Frequency plot of Song Duration')

plt.sca(ax2)

plt.xticks(np.arange(120, 450, step=30),('2:00','2:30','3:00','3:30','4:00','4:30','5:00','5:30','6:00','6:30','7:00'))

plt.yticks(np.arange(0, 250, step=20))

ax2.set_xlabel('Song Duration');

ax2.set_ylabel('Counts');



bpm_mean=df['bpm'].groupby(df['year']).mean()

length_mean=df['dur'].groupby(df['year']).mean()



bpm_med=df['bpm'].groupby(df['year']).median()

length_med=df['dur'].groupby(df['year']).median()





rcParams['figure.figsize'] =19,9





fig, axs =plt.subplots(2,2);

#Mean

axs[0,0].plot(bpm_mean);

axs[0,0].set_title('Mean BPM of songs for each year',fontsize=15);

axs[0,0].set_ylabel('Beats per Minute',fontsize=12)

plt.sca(axs[0,0])

plt.xticks(np.arange(2010, 2020, step=1));



axs[0,1].plot(length_mean);

axs[0,1].set_title('Mean Length of songs for each year',fontsize=15);

axs[0,1].set_ylabel('Minutes:econds',fontsize=12)

plt.sca(axs[0,1])

plt.xticks(np.arange(2010, 2020, step=1));

plt.yticks(np.arange(180, 260, step=10),('3:00','3:10','3:20','3:30','3:40','3:50','4:00','4:10','4:20'));



#Median

axs[1,0].plot(bpm_med);

axs[1,0].set_title('Median BPM of songs for each year',fontsize=15);

axs[1,0].set_ylabel('Beats per Minute,',fontsize=12)

plt.sca(axs[1,0])

plt.xticks(np.arange(2010, 2020, step=1));





axs[1,1].plot(length_med);

axs[1,1].set_title('Median Length of songs for each year',fontsize=15);

axs[1,1].set_ylabel('Minutes:Seconds',fontsize=12)

plt.sca(axs[1,1])

plt.xticks(np.arange(2010, 2020, step=1));

plt.yticks(np.arange(180, 260, step=10),('3:00','3:10','3:20','3:30','3:40','3:50','4:00','4:10','4:20'));

corr_matrix=df.corr()

corr_matrix

sb.heatmap(corr_matrix, annot=True);

#val and dnce

sb.regplot(x=df.val,y=df.dnce).set_title('Valence (Positive mood) vs Danceability',fontsize=15)

plt.xlabel('Valence (Positive mood)',fontsize=12);

plt.ylabel('Danceability',fontsize=12);

#val and nrgy

sb.regplot(x=df.val,y=df.nrgy).set_title('Valence (Positive mood) vs Energy',fontsize=15);

plt.xlabel('Valence (Positive mood)',fontsize=12);

plt.ylabel('Energy',fontsize=12);