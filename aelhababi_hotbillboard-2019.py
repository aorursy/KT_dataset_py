# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
hs = pd.read_csv('../input/Hot Stuff.csv')
hs.info()
## return Better Position obtained by the Song in  the Chart

def Better_Position(Song,df):

    return df[df['SongID']==Song]['Peak Position'].min()
## return whether the Song has been a Number One Sonf



def NumberOneSong(Song,df):

    if Better_Position(Song,df) == 1:

        return 1

    return 0

hs['NumberOneWeek']=hs['Peak Position'].where(hs['Peak Position']==1,0)
def find_top_5(s):

    s = s.sort_values(ascending=False)

    top5 = s.iloc[:5]

    top5_total = top5.sum()

    # total = s.sum()

    return  top5_total # / total
top10Performers = hs[hs['NumberOneWeek']==1].groupby('Performer').agg({'NumberOneWeek':sum }).sort_values('NumberOneWeek',ascending=False)[:10]


top10Performers.plot.bar(width=0.5,figsize=(18,5),color='green',fontsize=20,rot=80)


top10Songs = hs[hs['NumberOneWeek']==1].groupby('SongID').agg({'NumberOneWeek':sum }).sort_values('NumberOneWeek',ascending=False)[:10]
top10Songs.plot.bar(width=0.5,figsize=(18,5),color='purple',fontsize=20,rot=80)
hs.info()
NumberOnehs = hs[hs['Peak Position']==1]
topSongs = hs[hs['NumberOneWeek']==1].groupby(['Performer','SongID']).agg({'NumberOneWeek':sum })

topSongs.reindex()

topSongs['count']=1





PerformerNumberOneHit=topSongs.groupby('Performer').agg({'count':sum }).sort_values('count',ascending=False)
fig, ax = plt.subplots(figsize=(15,7))

ax.set_title('10 top performers on NumberOne Hits',fontsize=20,color='blue')

PerformerNumberOneHit[:10].plot(ax=ax,kind='barh',color='#FFD200',fontsize=20,rot=15,width=0.8,edgecolor='red',linewidth=1)
hs['Year']=hs['WeekID'].str[-4:]
hs['Month']=hs['WeekID'].str.extract(r'/(\w+)',expand=True)
OnlytopSongs = hs[hs['NumberOneWeek']==1][['Performer','SongID','Song','Year','NumberOneWeek']]
GroupedSongsYear = OnlytopSongs.groupby(['Year','Performer','SongID','Song']).agg({'NumberOneWeek':min})


GroupedSongsYear = GroupedSongsYear.reset_index()
GroupedSongsYear.head()
HitsByYearperPerformer=GroupedSongsYear.groupby(['Performer','Year']).agg({'NumberOneWeek':sum})

HitsByYearperPerformer=HitsByYearperPerformer.rename(columns={'NumberOneWeek':'NumberOnes'})
HitsByYearperPerformer=HitsByYearperPerformer.reset_index()
HitsByYearperPerformer
fig, ax = plt.subplots(figsize=(12,8))

ax.set_title('10 top Years for NumberOne Hits',fontsize=20,color='blue')

HitsByYearperPerformer.groupby('Year').sum().sort_values('NumberOnes',ascending=False)[:10].plot(ax=ax,kind='bar',fontsize=20,width=0.8, 

                                                           color='brown',edgecolor='red',linewidth=2)
fig, ax = plt.subplots(figsize=(17,8))

ax.set_title('10 top Years/performer for NumberOne Hits',fontsize=15,color='blue')

HitsByYearperPerformer.groupby(['Year','Performer']).sum().sort_values('NumberOnes',ascending=False)[:50].plot(

    ax=ax,kind='bar',fontsize=10,rot=85,width=0.5)
import squarify
PerformerNumberOneHit=PerformerNumberOneHit.reset_index()


def Hot100squarif(df,Size,Label,Slice):

    fig, ax = plt.subplots(figsize=(17,15))

    squarify.plot(ax=ax,sizes=df[:Slice][Size],label=df[:Slice][Label])
#squarify.plot(ax=ax,sizes=PerformerNumberOneHit[:20]['count'],label=PerformerNumberOneHit[:20]['Performer'])

Hot100squarif(PerformerNumberOneHit,'count','Performer',100)
#GroupedSongsYear[GroupedSongsYear['Year']=='1958']

def GiveMeAllHitSongs():

    year=input('For which year do you want to list the Number One hits?\n')

    print("\n",GroupedSongsYear[GroupedSongsYear['Year']== year][['Performer','Song']].to_string(index=False))


GiveMeAllHitSongs()