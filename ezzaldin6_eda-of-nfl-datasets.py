import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
injury_rec=pd.read_csv('/kaggle/input/nfl-playing-surface-analytics/InjuryRecord.csv')

player_trackdata=pd.read_csv('/kaggle/input/nfl-playing-surface-analytics/PlayerTrackData.csv')

play=pd.read_csv('/kaggle/input/nfl-playing-surface-analytics/PlayList.csv')
plt.style.use('fivethirtyeight')

sns.set_style('whitegrid')

injury_rec['Surface'].value_counts().plot.pie(autopct='%.1f%%',

                                              shadow=True)

plt.title('Comparing injuries in different fields')

plt.ylabel('')

plt.show()
injury_rec['injury_type']=injury_rec[['DM_M1','DM_M7','DM_M28','DM_M42']].sum(axis=1)

def change_vals(i):

    if i==1:

        return 'Light Injury'

    elif i==2:

        return 'Medium Injury'

    elif i==3:

        return 'Almost Serious Injury'

    else:

        return 'Serious Injury'

injury_rec['injury_type']=injury_rec['injury_type'].apply(change_vals)
injury_rec['injury_type'].value_counts(normalize=True).plot.pie(autopct='%.1f%%',

                                                                shadow=True,

                                                               colors=['gold', 'yellowgreen', 'lightcoral', 'lightskyblue'])

plt.title('Injury-Types of NFL players')

plt.ylabel('')

plt.show()
sns.countplot(x='Surface',hue='BodyPart',data=injury_rec)

plt.legend(loc='best')

plt.show()
sns.countplot(x='Surface',hue='injury_type',data=injury_rec)

plt.legend(loc='best')

plt.show()
body_part=injury_rec['BodyPart'].unique().tolist()

fig=plt.figure(figsize=(15,3))

for i in range(len(body_part)-1):

    ax=fig.add_subplot(1,5,i+1)

    c=injury_rec[(injury_rec['BodyPart']==body_part[i])&(injury_rec['Surface']=='Synthetic')]['injury_type'].value_counts(normalize=True)*100

    c_lst=c.tolist()

    ax.bar(c.index,c_lst,width=0.4)

    ax.set_title('{} in Synthetic'.format(body_part[i]))

    ax.set_ylabel('')

    xlabels=[i for i in c.index]

    ax.set_xticklabels(xlabels, rotation=90)

plt.show()
body_part=injury_rec['BodyPart'].unique().tolist()

fig=plt.figure(figsize=(15,3))

for i in range(len(body_part)):

    ax=fig.add_subplot(1,5,i+1)

    c=injury_rec[(injury_rec['BodyPart']==body_part[i])&(injury_rec['Surface']=='Natural')]['injury_type'].value_counts(normalize=True)*100

    c_lst=c.tolist()

    ax.bar(c.index,c_lst,width=0.4,color='red')

    ax.set_title('{} in Natural'.format(body_part[i]))

    ax.set_ylabel('')

    xlabels=[i for i in c.index]

    ax.set_xticklabels(xlabels, rotation=90)

plt.show()
injured_full_data=injury_rec.merge(play)

len(injured_full_data)
injured_full_data['PlayType'].value_counts().plot.barh(figsize=(15,5))

plt.title('play-type of injured players')

plt.xlabel('frequency')

plt.ylabel('play-type')

plt.show()
plt.figure(figsize=(16,5))

plt.subplot(1,2,1)

injured_full_data['Position'].value_counts().plot.barh()

plt.title('Position of injured players')

plt.xlabel('frequency')

plt.ylabel('')

plt.subplot(1,2,2)

play['Position'].value_counts().plot.barh()

plt.title('Position of all players')

plt.xlabel('frequency')

plt.ylabel('')

plt.show()
injured_full_data['Temperature'].plot.kde(label='injured players')

play['Temperature'].plot.kde(color='red',label='all players')

plt.axvline(injured_full_data['Temperature'].mean(),c='green',label='inj_mean')

plt.axvline(play['Temperature'].mean(),c='yellow',label='all_mean')

plt.legend(loc='best')

plt.show()
injured_full_data['Weather'].value_counts().plot.barh(figsize=(15,10))

plt.title('Weather')

plt.xlabel('frequency')

plt.ylabel('Weather-type')

plt.show()
plt.figure(figsize=(6,16))

plt.subplot(2,1,1)

injured_full_data[injured_full_data['Surface']=='Synthetic']['Weather'].value_counts().plot.barh(figsize=(15,10),color='red')

plt.title('Weather in Synthetic')

plt.ylabel('')

plt.subplot(2,1,2)

injured_full_data[injured_full_data['Surface']=='Natural']['Weather'].value_counts().plot.barh(figsize=(15,10))

plt.title('Weather in Natural')

plt.xlabel('frequency')

plt.ylabel('')

plt.show()
player_trackdata['s'].plot.hist(bins=20)

plt.axvline(player_trackdata['s'].mean(),color='red',label='mean')

plt.axvline(player_trackdata['s'].median(),color='green',label='median')

plt.axvline(player_trackdata['s'].std(),color='blue',label='std')

plt.legend()

plt.show()
inj_players=injury_rec['PlayKey'].tolist()

player_trackdata.query('PlayKey in @inj_players')['s'].plot.hist(bins=20)

plt.title('Distribution of speed of injured players')

plt.xlabel('speed')

plt.ylabel('frequency')

plt.axvline(player_trackdata.query('PlayKey in @inj_players')['s'].mean(),label='mean',color='red')

plt.axvline(player_trackdata.query('PlayKey in @inj_players')['s'].std(),label='std',color='blue')

plt.legend()

plt.show()
player_trackdata.query('PlayKey in @inj_players')['o'].plot.hist(bins=30)

plt.title('Distribution of Orientation of injured players')

plt.xlabel('Orientation')

plt.ylabel('frequency')

plt.axvline(player_trackdata.query('PlayKey in @inj_players')['o'].mean(),label='mean',color='red')

plt.axvline(player_trackdata.query('PlayKey in @inj_players')['o'].std(),label='std',color='blue')

plt.legend(loc='best')

plt.show()
player_trackdata.query('PlayKey in @inj_players')['dir'].plot.hist(bins=30)

plt.title('Distribution of Direction of injured players')

plt.xlabel('Direction')

plt.ylabel('frequency')

plt.axvline(player_trackdata.query('PlayKey in @inj_players')['dir'].mean(),label='mean',color='red')

plt.axvline(player_trackdata.query('PlayKey in @inj_players')['dir'].std(),label='std',color='blue')

plt.legend(loc='best')

plt.show()
inj_synth=injury_rec[injury_rec['Surface']=='Synthetic']['PlayKey'].tolist()

inj_natural=injury_rec[injury_rec['Surface']=='Natural']['PlayKey'].tolist()

player_trackdata.query('PlayKey in @inj_synth')['s'].plot.hist(bins=20,alpha=0.4,label='Synthetic')

player_trackdata.query('PlayKey in @inj_natural')['s'].plot.hist(bins=20,color='red',alpha=0.4,label='natural')

plt.title('injured players in synthetic VS Natural')

plt.xlabel('Speed')

plt.ylabel('frequency')

plt.legend()
print('mean of injured players in synthetic field: ',player_trackdata.query('PlayKey in @inj_synth')['s'].mean())

print('mean of injured players in natural field: ',player_trackdata.query('PlayKey in @inj_natural')['s'].mean())
print('std of injured players in synthetic field: ',player_trackdata.query('PlayKey in @inj_synth')['s'].std())

print('std of injured players in natural field: ',player_trackdata.query('PlayKey in @inj_natural')['s'].std())
print('skewness of injured players in synthetic field: ',player_trackdata.query('PlayKey in @inj_synth')['s'].skew())

print('skewness of injured players in natural field: ',player_trackdata.query('PlayKey in @inj_natural')['s'].skew())
print('kurtosis of injured players in synthetic field: ',player_trackdata.query('PlayKey in @inj_synth')['s'].kurtosis())

print('kurtosis of injured players in natural field: ',player_trackdata.query('PlayKey in @inj_natural')['s'].kurtosis())
player_trackdata.query('PlayKey in @inj_synth')['o'].plot.hist(bins=20,alpha=0.4,label='Synthetic')

player_trackdata.query('PlayKey in @inj_natural')['o'].plot.hist(bins=20,color='red',alpha=0.4,label='natural')

plt.title('injured players in synthetic VS Natural')

plt.xlabel('Orientation')

plt.ylabel('frequency')

plt.legend()

plt.show()
print('mean of injured players in synthetic field: ',player_trackdata.query('PlayKey in @inj_synth')['o'].mean())

print('mean of injured players in natural field: ',player_trackdata.query('PlayKey in @inj_natural')['o'].mean())
print('std of injured players in synthetic field: ',player_trackdata.query('PlayKey in @inj_synth')['o'].std())

print('std of injured players in natural field: ',player_trackdata.query('PlayKey in @inj_natural')['o'].std())
print('skewness of injured players in synthetic field: ',player_trackdata.query('PlayKey in @inj_synth')['o'].skew())

print('skewness of injured players in natural field: ',player_trackdata.query('PlayKey in @inj_natural')['o'].skew())
print('kurtosis of injured players in synthetic field: ',player_trackdata.query('PlayKey in @inj_synth')['o'].kurtosis())

print('kurtosis of injured players in natural field: ',player_trackdata.query('PlayKey in @inj_natural')['o'].kurtosis())
player_trackdata.query('PlayKey in @inj_synth')['dir'].plot.hist(bins=20,alpha=0.4,label='Synthetic')

player_trackdata.query('PlayKey in @inj_natural')['dir'].plot.hist(bins=20,color='red',alpha=0.4,label='natural')

plt.title('injured players in synthetic VS Natural')

plt.xlabel('Direction')

plt.ylabel('frequency')

plt.legend()

plt.show()
print('mean of injured players in synthetic field: ',player_trackdata.query('PlayKey in @inj_synth')['dir'].mean())

print('mean of injured players in natural field: ',player_trackdata.query('PlayKey in @inj_natural')['dir'].mean())
print('std of injured players in synthetic field: ',player_trackdata.query('PlayKey in @inj_synth')['dir'].std())

print('std of injured players in natural field: ',player_trackdata.query('PlayKey in @inj_natural')['dir'].std())
print('skewness of injured players in synthetic field: ',player_trackdata.query('PlayKey in @inj_synth')['dir'].skew())

print('skewness of injured players in natural field: ',player_trackdata.query('PlayKey in @inj_natural')['dir'].skew())
print('kurtosis of injured players in synthetic field: ',player_trackdata.query('PlayKey in @inj_synth')['dir'].kurtosis())

print('kurtosis of injured players in natural field: ',player_trackdata.query('PlayKey in @inj_natural')['dir'].kurtosis())