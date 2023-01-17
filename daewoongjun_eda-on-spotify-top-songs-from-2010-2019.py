# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

from plotly import __version__

import plotly.graph_objs as go

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

import cufflinks as cf

cf.go_offline()

init_notebook_mode(connected=True)

%matplotlib inline

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
songs = pd.read_csv('../input/top-spotify-songs-from-20102019-by-year/top10s.csv',encoding='ISO-8859-1')

print(songs.head())
print(songs.columns)
#Clearing Data

print(songs.isnull().sum())


top10songs = []

for i in range(0,10):

    year = str(201)+str(i)

    yearint = int(year)

    idx = songs[songs['year']==yearint]['title'].head(10)

    top10songslist = songs[songs['year']== yearint]['title'].tolist()[0:10]#Removing the index .to_string(index=False) works too

    top10str = "\n".join(top10songslist)

    print("The Top10 Songs in year {} are \n{} ".format(year,top10str),'\n')

    


popularsongs =[]

popsonggenre = []

for i in range(0,10):

    year = str(201)+str(i)

    yearint = int(year)

    idx = songs[songs['year']==yearint]['pop'].idxmax()

    popularsongs.append(songs.iloc[idx]['title'])

    popsonggenre.append(songs.iloc[idx]['top genre'])

dic1 = {"Year":['2010','2011','2012','2013','2014','2015','2016','2017','2018','2019'],

        "Songs": popularsongs,

        "Genre": popsonggenre}

mpsbyyear = pd.DataFrame.from_dict(dic1)

topgenre = mpsbyyear['Genre'].mode()

topgenrelis = topgenre.tolist()

topgenrelis = "".join(topgenrelis)

print(mpsbyyear)

print("The most frequent genre of Top1 by year is : {} ".format(topgenrelis))
print(songs['top genre'].value_counts())

plt.figure(figsize = (10,10))

plt.xticks(rotation=90)

sns.countplot(x='top genre',data = songs,order = songs['top genre'].value_counts(ascending=True).tail(10).index,palette='dark')

plt.title('Total Number of the Songs of Each Genre')
# The Longest Title from 2010-2019 (Excluding the spaces, featuring names and the letters after "-")

max = 0



ans = ""

for i in songs['title']:

    length = 1

    for k in i:

        if k == '(':

            break

        elif k == '-':

            break

        length = length + 1

    if length - i.count(" ") > max:

        max = length

        ans = i

print("The longest title from 2010-2019 is \"{}\"".format(ans))
#Let's look at the top singer of some genres

#dance pop

top_dp = songs.iloc[songs[songs['top genre'] == 'dance pop']['pop'].idxmax()]

dic_tdp = {"Top Dance Pop Music":['pop','nrgy','dnce','val','acous', 'spch','live'],

           "Values":[top_dp['pop'],top_dp['nrgy'],top_dp['dnce'],top_dp['val'],top_dp['acous'],top_dp['spch'],top_dp['live']]}

top_dp_df = pd.DataFrame.from_dict(dic_tdp)

top_dp_df.iplot(kind='bar',x='Top Dance Pop Music',y='Values',color='blue')
#pop

top_popm = songs.iloc[songs[songs['top genre'] == 'pop']['pop'].idxmax()]

print(top_popm.loc[['title','artist','pop','year']])



dic_popm = {"Top Pop Music":['pop','nrgy','dnce','val','acous', 'spch','live'],

           "Values":[top_popm['pop'],top_popm['nrgy'],top_popm['dnce'],top_popm['val'],top_popm['acous'],top_popm['spch'],top_popm['live']]}

top_popm_df = pd.DataFrame.from_dict(dic_popm)

top_popm_df.iplot(kind='bar',x='Top Pop Music',y='Values',color='red')

#edm

top_edm = songs.iloc[songs[songs['top genre'] == 'edm']['pop'].idxmax()]

print(top_edm.loc[['title','artist','pop','year']])



dic_edm = {"Top EDM Music":['pop','nrgy','dnce','val','acous', 'spch','live'],

           "Values":[top_edm['pop'],top_edm['nrgy'],top_edm['dnce'],top_edm['val'],top_edm['acous'],top_edm['spch'],top_edm['live']]}

top_edm_df = pd.DataFrame.from_dict(dic_edm)

top_edm_df.iplot(kind='bar',x='Top EDM Music',y='Values',color = 'yellow')
#hip hop

top_hh = songs.iloc[songs[songs['top genre'] == 'hip hop']['pop'].idxmax()]

print(top_hh.loc[['title','artist','pop','year']])



dic_hh = {"Top Hip Hop Music":['pop','nrgy','dnce','val','acous', 'spch','live'],

           "Values":[top_hh['pop'],top_hh['nrgy'],top_hh['dnce'],top_hh['val'],top_hh['acous'],top_hh['spch'],top_hh['live']]}

top_hh_df = pd.DataFrame.from_dict(dic_hh)

top_hh_df.iplot(kind='bar',x='Top Hip Hop Music',y='Values',color = 'black')
plt.figure(figsize=(20,10))

plt.xticks(size=10,rotation=90,color = 'red')

plt.xlabel('Artist',color='red')

plt.ylabel('Count',color = 'red')

plt.yticks(color='red')

sns.countplot(x='artist',data = songs,order = songs['artist'].value_counts(ascending=True).tail(10).index, palette='viridis')

plt.title("Most Frequently Appeared Artists",size=20)
Mostfreqartist = songs['artist'].mode().tolist()

Mostfreqartist = "".join(Mostfreqartist)

print("The most frequently appeared artist from 2010-2019 is {}".format(Mostfreqartist))

dic = {"Title" : songs[songs['artist']=='Katy Perry']['title'],

       "Genre" : songs[songs['artist']=='Katy Perry']['top genre'],

       "Popularity" : songs[songs['artist']=='Katy Perry']['pop'],

       "Year": songs[songs['artist']=='Katy Perry']['year']}

df_mfa = pd.DataFrame(dic).reset_index(drop=True)

print(df_mfa)
mostbpm = songs['bpm'].value_counts(ascending=True).keys().tolist()

mostbpmused =  songs['bpm'].value_counts(ascending=True).tolist()

print('The most frequently used bpm is : {} and there were {} songs with this BPM from 2010-2019'.format(mostbpm[-1],mostbpmused[-1]))

plt.figure(figsize=(15,4))

plt.title('Most Frequently used bpm')

sns.countplot(x='bpm',data=songs,order = songs['bpm'].value_counts(ascending=True).tail(15).index,palette='Set2')

#Lets see the features of some top genre:

genre =('dance pop', 'canadian pop','electropop', 'edm','pop')

pop_dist = songs.loc[songs['top genre'].isin(genre) & songs['pop']]

#pop_dist.boxplot(column=['pop'],by=['top genre'],figsize=(12,8),figsize=(30,7),figsize=(30,7),fontsize=15)

plt.style.use('fivethirtyeight')

plt.figure(figsize=(10,7))

sns.violinplot(x=pop_dist['top genre'],y=pop_dist['pop'],palette = 'viridis').set_title('The Popularity Distribtion of Top Genre')
 #Lets see the features of some top genre:

dB_dist = songs.loc[(songs['top genre'].isin(genre)) & (songs['dB'])]

plt.figure(figsize=(10,4))

sns.boxenplot(x=pop_dist['top genre'],y=pop_dist['dB'],palette = 'Set2').set_title('The dB Distribtion of Top Genre')
speech_dist = songs.loc[(songs['top genre'].isin(genre)) & (songs['spch'])]

plt.figure(figsize=(10,4))

sns.boxplot(x=pop_dist['top genre'],y=pop_dist['spch'],palette = 'Set2_r').set_title('The Speech Distribtion of Top Genre')
energy_dist = songs.loc[(songs['top genre'].isin(genre)) & (songs['nrgy'])]

plt.figure(figsize=(10,4))

sns.boxenplot(x=pop_dist['top genre'],y=pop_dist['nrgy'],palette = 'Set1').set_title('The Energy Distribtion of Top Genre')
songs.columns

# Distribution among the years

year = ('2010','2011','2012','2013','2014','2015','2016','2017','2018','2019')

dnce_distribution = songs.loc[songs['year'].isin(year) & songs['dnce']]

plt.figure(figsize=(15,4))

plt.style.use('fivethirtyeight')

plt.style.use('_classic_test')

sns.violinplot(x=dnce_distribution['year'],y=dnce_distribution['dnce']).set_title('Danceability Distribtuion among 10 years')
duration_distribution = songs.loc[songs['year'].isin(year) & songs['dur']]

plt.figure(figsize=(15,4))

plt.style.use('Solarize_Light2')

sns.violinplot(x=duration_distribution['year'],y=duration_distribution['dur']).set_title('Song Duration Distribtuion among 10 years')
bpm_distribution = songs.loc[songs['year'].isin(year) & songs['bpm']]

plt.figure(figsize=(15,4))

plt.style.use('bmh')

sns.boxenplot(x=bpm_distribution['year'],y=bpm_distribution['bpm']).set_title('BPM Distribtuion among 10 years')
val_distribution = songs.loc[songs['year'].isin(year) & songs['val']]

plt.figure(figsize=(15,4))

plt.style.use('ggplot')

sns.boxenplot(x=val_distribution['year'],y=val_distribution['val']).set_title('Valence Distribtuion among 10 years')
songs.columns

sns.heatmap(songs[['bpm','nrgy','dnce','dB','live','val','dur','acous','spch','pop']].corr(),annot=True)
Maroon5 = songs[songs['artist']=='Maroon 5']

Maroon5


Maroon5 = songs[songs['artist']=='Maroon 5']

g= sns.FacetGrid(Maroon5,col='year')

g = g.map(sns.countplot,'top genre')
sns.barplot(x=Maroon5['year'],y=Maroon5['pop'],palette='Set3_r',color='blue').set_title('Popularity of the Songs of Maroon 5 Among the 10 years')

fig,ax = plt.subplots(nrows= 3,ncols=2,figsize=(15,7))

fig.tight_layout(pad=3.0)    # Adjusting the space gap between the subplots

Maroon5_bpm = Maroon5.groupby(['year'],as_index=False)['bpm'].mean().apply(np.int64)      #apply is used to make the floats to int

#mean_bpm                            #by putting the as_index it removes the empty index and make bpm as a col name

Maroon5_nrgy = Maroon5.groupby(['year'],as_index=False)['nrgy'].mean().apply(np.int64)  

Maroon5_dnce = Maroon5.groupby(['year'],as_index=False)['dnce'].mean().apply(np.int64)  

Maroon5_val = Maroon5.groupby(['year'],as_index=False)['val'].mean().apply(np.int64)  

Maroon5_dur = Maroon5.groupby(['year'],as_index=False)['dur'].mean().apply(np.int64) 

Maroon5_acous = Maroon5.groupby(['year'],as_index=False)['acous'].mean().apply(np.int64)  



plt.style.use('Solarize_Light2')

ax[0][0].plot(Maroon5_bpm['year'],Maroon5_bpm['bpm'])

ax[0][0].set_title('Avg BPM over 10 years')

ax[0][0].set_xlabel('Years')

ax[0][0].set_ylabel('BPM')



ax[0][1].plot(Maroon5_nrgy['year'],Maroon5_nrgy['nrgy'])

ax[0][1].set_title('Avg Energy level of the Songs over 10 years')

ax[0][1].set_xlabel('Years')

ax[0][1].set_ylabel('Energy')



ax[1][0].plot(Maroon5_dnce['year'],Maroon5_dnce['dnce'])

ax[1][0].set_title('Avg Danceability over 10 years')

ax[1][0].set_xlabel('Years')

ax[1][0].set_ylabel('dnce')



ax[1][1].plot(Maroon5_val['year'],Maroon5_val['val'])

ax[1][1].set_title('Avg Valence over 10 years')

ax[1][1].set_xlabel('Years')

ax[1][1].set_ylabel('val')



ax[2][0].plot(Maroon5_dur['year'],Maroon5_dur['dur'])

ax[2][0].set_title('Avg Duration of the Songs over 10 years')

ax[2][0].set_xlabel('Years')

ax[2][0].set_ylabel('dur')



ax[2][1].plot(Maroon5_acous['year'],Maroon5_acous['acous'])

ax[2][1].set_title('Avg Acoustic level over 10 years')

ax[2][1].set_xlabel('Years')

ax[2][1].set_ylabel('acous')

#Do people like to listen to the songs which is recorded on live

sns.kdeplot(songs['live'],songs['pop'],shade=True)

plt.title('Do Songs Recorded Live are Prone to be More Popular? ')
sns.jointplot(songs['dnce'],songs['val'],kind='reg',color='b')
sns.jointplot(kind='kde',x=songs['bpm'],y=songs['nrgy'])
plt.style.use('ggplot')

sns.kdeplot(songs['bpm'],songs['val'],shade=True,cmap="Purples")

plt.title('Which BPM Gives the Highest Valence?')
plt.style.use('fivethirtyeight')

sns.jointplot(kind='hex',x=songs['dur'],y=songs['pop'])
plt.style.use('ggplot')

sns.kdeplot(songs['spch'],songs['pop'],shade=True)

plt.title('Longer the Speech, the Better Song Popularity ?')
plt.style.use('ggplot')

sns.kdeplot(songs['spch'],songs['dnce'],shade=True,color='y').set_title('Correlation Between Danceability and Speech')
def letterlength(a):           # Counting the length of the letters

    length = 0 

    idx = 0

    for i in a:    

        if i == '(':

           

            break

        elif i == '-':

           

            break

        length = length + 1

        idx = idx + 1        # index when it breaks the for loop

    if (('(' in a) | ('-' in a)):

        length = length - a[0:idx].count(" ")

    else :

        length = length - a.count(" ")

    return(length)

        
songs['lengthoftitle'] = songs['title'].apply(letterlength)

sns.countplot(x=songs['lengthoftitle'],palette='viridis')

#plt.figure(figsize=(10,4))

#sns.distplot(songs['lengthofsongs'])

groupbylen = songs.groupby(['lengthoftitle'],as_index=False)['pop'].mean().apply(np.int64)

sns.jointplot(kind='hex',x=songs['lengthoftitle'],y=songs['pop'],cmap='Purples')