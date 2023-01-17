# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
play_list = pd.read_csv('/kaggle/input/nfl-playing-surface-analytics/PlayList.csv')

play_list.head()
injury_record = pd.read_csv('/kaggle/input/nfl-playing-surface-analytics/InjuryRecord.csv')

injury_record.head()
player_track_record = pd.read_csv('/kaggle/input/nfl-playing-surface-analytics/PlayerTrackData.csv')

player_track_record.head()


print(play_list.shape,injury_record.shape,player_track_record.shape)
#lower limb injury's count based on Playerkey

import matplotlib                  

import matplotlib.pyplot as plt

import seaborn as sns  

lower_inj=injury_record.groupby('BodyPart').count()['PlayerKey']

print(lower_inj)

injury_record.groupby('BodyPart')['PlayerKey'].count().plot(kind='bar', figsize = (10,8), title='Count of Injuries based on body parts')

plt.xticks(rotation=100)

plt.show()
count_inj=injury_record.groupby(['Surface']).count()['PlayerKey']

print(count_inj)

injury_record.groupby(['Surface']).count()['PlayerKey'].plot(kind='bar', figsize = (5,5), title='Count of Injuries based on Surface')

plt.xticks(rotation=100)

plt.show()

inj_pivot = injury_record.pivot_table(index='BodyPart', columns='Surface', aggfunc='size',fill_value=0)

inj_pivot.head()
#Visualising the pivot table data in a bar chart

inj_pivot.plot(kind='barh', figsize=[10,5], stacked=False, colormap='autumn')


play_inj_together = injury_record.merge(play_list, on='PlayKey')

play_inj_together.head()
#play_inj_together.isnull().sum()
play_inj_together['PlayKey'].nunique()

#play_inj_together['PlayKey'].value_counts()

#play_list['Position'].value_counts()

play_list['Position'].value_counts().plot(kind='bar', figsize=(10,5), color='orange', title='Position of All Players')

#play_list.info()
#play_inj_together['Position'].nunique()

play_inj_together['Position'].value_counts().plot(kind='bar', figsize=(10,5), color='orange', title='Position of Injured Players')

plt.legend()


play_inj_pivot = play_inj_together.pivot_table(index=['Position','Surface'], columns='BodyPart', aggfunc='size',fill_value=0)

play_inj_pivot.head()


play_inj_pivot.plot(kind='barh', figsize=[15,15],stacked=True)

plt.title('Injuries based on Position and Surface')
play_inj_together.columns
#Injuries occurred in Natural Surface when the weather varies from sunny to cloudy.

play_inj_stad_pivot = play_inj_together[play_inj_together['Surface'] == 'Natural']['Weather'].value_counts()

#play_inj_stad_pivot

play_inj_stad_pivot.plot(kind='bar', figsize = (10,5))

plt.show()
#displaying the Injuries happened in Synthetic Surface when the weather varies from sunny, cloudy and mostly sunny.

play_inj_stad_pivot1 = play_inj_together[play_inj_together['Surface'] == 'Synthetic']['Weather'].value_counts()

#play_inj_stad_pivot

play_inj_stad_pivot1.plot(kind='bar', figsize = (10,5), color ='orange')

plt.show()
#Analysing the attributes Surface, Weather and StadiumType together to visualise the facts. 

#To analyse whether the characteristics of stadium and weather having impact on the surface and injuries.

comb_result = play_inj_together.pivot_table(index=['StadiumType','Weather'], columns='Surface', aggfunc='size',fill_value=0)

comb_result
comb_result.plot(kind='bar', figsize=[15,15], stacked=True, colormap='autumn')

plt.ylabel('Number of Injuries')
#player_track_record.columns

player_track_record.head()
#To visualise the speed of all the players during the game.

x=player_track_record['s']

import pandas as pd

y = pd.Series(x)

ax = sns.distplot(y,hist=True,color='brown')
#Plotting the players movement (speed) during the game. The value displayed in the graph has its mean, 

#minimum and maximum speed of the injured players. One can visualise the number of the injuries have been diminished

#when the speed increases.



injury_rec=injury_record['PlayKey'].tolist()

player_track_record.query('PlayKey in @injury_rec')['s'].plot.hist(bins=30)

plt.axvline(player_track_record.query('PlayKey in @injury_rec')['s'].mean(),label='mean',color='green')

#The below given script shows the minimum and maximum speed of players who got injured. 

plt.axvline(player_track_record.query('PlayKey in @injury_rec')['s'].max(), label='max_speed',color='Orange')

plt.axvline(player_track_record.query('PlayKey in @injury_rec')['s'].min(), label='min_speed',color='brown')

plt.legend()

plt.xlabel('Speed')

plt.ylabel('Frequency')



#The graph shows the maximum speed of players who got injured both in Natural and Synthetic Surface.

injury_Nat=injury_record[injury_record['Surface'] == 'Natural']['PlayKey'].tolist()

injury_Syn=injury_record[injury_record['Surface'] == 'Synthetic']['PlayKey'].tolist()

player_track_record.query('PlayKey in @injury_Nat')['s'].plot.hist(bins=30,color='blue', label='Natural')

player_track_record.query('PlayKey in @injury_Syn')['s'].plot.hist(bins=30,color='orange', label='Synthetic',)

plt.axvline(player_track_record.query('PlayKey in @injury_Nat')['s'].max(), label='max_speed_N',color='green')

plt.axvline(player_track_record.query('PlayKey in @injury_Syn')['s'].max(), label='max_speed_S',color='red')

plt.legend()

plt.xlabel('Speed')

plt.ylabel('Frequency')
player_track_record.query('PlayKey in @injury_Nat')['s'].plot.hist(bins=30,color='blue', label='Natural')

player_track_record.query('PlayKey in @injury_Syn')['s'].plot.hist(bins=30,color='orange', label='Synthetic',)

plt.axvline(player_track_record.query('PlayKey in @injury_Nat')['s'].skew(),color='red', label='Natural_sk')

plt.axvline(player_track_record.query('PlayKey in @injury_Syn')['s'].skew(),color='green', label='Synthetic_sk')

plt.xlabel('Speed')

plt.legend()
player_track_record.query('PlayKey in @injury_Nat')['dir'].plot.hist(bins=30,color='blue', label='Natural')

player_track_record.query('PlayKey in @injury_Syn')['dir'].plot.hist(bins=30,color='brown', label='Synthetic',)

plt.axvline(player_track_record.query('PlayKey in @injury_Nat')['dir'].skew(),color='yellow')

plt.axvline(player_track_record.query('PlayKey in @injury_Syn')['dir'].skew(),color='green')

plt.legend()

plt.xlabel('direction')

player_track_record.query('PlayKey in @injury_Nat')['o'].plot.hist(bins=30,color='blue', label='Natural')

player_track_record.query('PlayKey in @injury_Syn')['o'].plot.hist(bins=30,color='brown', label='Synthetic',)

plt.axvline(player_track_record.query('PlayKey in @injury_Nat')['o'].skew(),color='yellow')

plt.axvline(player_track_record.query('PlayKey in @injury_Syn')['o'].skew(),color='green')

plt.legend()

plt.xlabel('O')