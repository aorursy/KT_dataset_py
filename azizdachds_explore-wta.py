import pandas as pd
import matplotlib.pyplot as plt

players = pd.read_csv('../input/players/players.csv', encoding='latin1', index_col=0)

# Top column is misaligned.
players.index.name = 'ID'
players.columns = ['First' , 'Last', 'Handidness', 'DOB', 'Country']

# Parse date data to dates.
players = players.assign(DOB=pd.to_datetime(players['DOB'], format='%Y%m%d'))

# Handidness is reported as U if unknown; set np.nan instead.
import numpy as np
players = players.assign(Handidness=players['Handidness'].replace('U', np.nan))

players.head()

matches = pd.read_csv('../input/player/matches.csv', encoding='latin1', index_col=0)
matches.head(5)
import pandas as pd
rankings = pd.read_csv('../input/player/rankings.csv', encoding='latin1', index_col=0)
rankings.head(5)
#Data manipulation libraries : 
import numpy as np  #numpy
import pandas as pd  #pandas

#System libraries
import glob #The glob module finds all the pathnames matching a specified pattern according to the rules used by the Unix shell



#Plotting
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

#math operations lib 
import math
from math import pi

#date manipulation 
import datetime as dt

#Impute missing data
from sklearn.preprocessing import Imputer 



#Splitting data to test and train 
from sklearn.model_selection import train_test_split

import datetime

import os

#countrys with moste nomber of players 
players.Country.value_counts().head(20).plot.bar(
    figsize=(12, 6),
    title='WTA Player Country Representing'
)

matches['winner_ioc'].value_counts().head(20).plot.bar(
    figsize=(12, 4),
    title='WTA country with Most Matches Win'
)
#WTA country with Most Matches lost

matches['loser_ioc'].value_counts().head(20).plot.bar(
    figsize=(12, 4),
    title='WTA country with Most Matches lost'
)
pd.concat([matches['winner_name'], matches['loser_name']]).value_counts().head(20).plot.bar(
    figsize=(12, 4),
    title='WTA Players with Most Matches Played'
)
matches['winner_name'].value_counts().head(20).plot.bar(
    figsize=(12, 4),
    title='WTA Players with Most Matches Won'
)
matches['loser_name'].value_counts().head(20).plot.bar(
    figsize=(12, 4),
    title='WTA Players with Most Matches Lost'
)
import pandas as pd

t11 = pd.read_csv('../input/player/wta_matches_2011.csv', encoding='latin1', index_col=0)
t12 = pd.read_csv('../input/player/wta_matches_2012.csv', encoding='latin1', index_col=0)
t13 = pd.read_csv('../input/player/wta_matches_2013.csv', encoding='latin1', index_col=0)
t14 = pd.read_csv('../input/player/wta_matches_2014.csv', encoding='latin1', index_col=0)
t15 = pd.read_csv('../input/player/wta_matches_2015.csv', encoding='latin1', index_col=0)
t16 = pd.read_csv('../input/player/wta_matches_2016.csv', encoding='latin1', index_col=0)

wta = pd.concat([t11, t12,t13,t14,t15,t16])
wta.head(5)
wta.duplicated().sum()
players.duplicated().sum()
rankings.shape
rankings.duplicated().sum()

rankings.drop_duplicates(keep='first').shape
wta.isnull().sum(axis=0)
wta.shape

           
wta['surface'].value_counts(dropna=False)
wta['surface'].fillna(value='Hard', inplace=True)
wta['winner_hand'].fillna(value='R', inplace=True)

wta['loser_hand'].fillna(value='R', inplace=True)

wta['loser_entry'].value_counts(dropna=False)

wta['loser_entry'].fillna(value='S', inplace=True)

wta['winner_entry'].fillna(value='S', inplace=True)

wta = wta.drop('l_SvGms', 1)

wta = wta.drop('w_SvGms', 1)

wta.isnull().sum(axis=0)
wta['l_bpFaced'].fillna((wta['l_bpFaced'].mean()), inplace=True)

wta['l_bpSaved'].fillna((wta['l_bpSaved'].mean()), inplace=True)

wta['l_2ndWon'].fillna((wta['l_2ndWon'].mean()), inplace=True)

wta['l_1stWon'].fillna((wta['l_1stWon'].mean()), inplace=True)

wta['l_1stIn'].fillna((wta['l_1stIn'].mean()), inplace=True)

wta['l_svpt'].fillna((wta['l_svpt'].mean()), inplace=True)

wta['l_df'].fillna((wta['l_df'].mean()), inplace=True)

wta['l_ace'].fillna((wta['l_ace'].mean()), inplace=True)

wta['w_bpFaced'].fillna((wta['w_bpFaced'].mean()), inplace=True)

wta['w_bpSaved'].fillna((wta['w_bpSaved'].mean()), inplace=True)

wta['w_ace'].fillna((wta['w_ace'].mean()), inplace=True)

wta['w_df'].fillna((wta['w_df'].mean()), inplace=True)

wta['w_1stIn'].fillna((wta['w_1stIn'].mean()), inplace=True)

wta['w_svpt'].fillna((wta['w_svpt'].mean()), inplace=True)

wta['w_1stWon'].fillna((wta['w_1stWon'].mean()), inplace=True)

wta['w_2ndWon'].fillna((wta['w_2ndWon'].mean()), inplace=True)

matches.isnull().sum(axis=0)
matches.shape

matches['loser_rank'].fillna(value='40.0', inplace=True)

matches['year'].value_counts(dropna=False)

matches['year'].fillna(value='2002.0', inplace=True)

matches['round'].value_counts(dropna=False)

matches['round'].fillna(value='R32', inplace=True)

matches['surface'].fillna(value='Hard', inplace=True)

matches['winner_hand'].fillna(value='R', inplace=True)

matches['loser_hand'].fillna(value='R', inplace=True)

matches['loser_rank_points'].value_counts(dropna=False)

matches['loser_rank_points'].fillna(value='400.0', inplace=True)

matches['minutes'].fillna((matches['minutes'].mean()), inplace=True)

matches['winner_ht'].value_counts(dropna=False)

matches['winner_ht'].fillna(value='170.0', inplace=True)

matches['winner_entry'].fillna(value='S', inplace=True)


matches = matches.drop('Unnamed: 32', 1)

winners = list(np.unique(wta.winner_name))
losers = list(np.unique(wta.loser_name))

all_players = winners + losers
players = np.unique(all_players)

players_wta = pd.DataFrame()
players_wta['Name'] = players
players_wta['Wins'] = players_wta.Name.apply(lambda x: len(wta[wta.winner_name == x]))
players_wta['Losses'] = players_wta.Name.apply(lambda x: len(wta[wta.loser_name == x]))

players_wta['PCT'] = np.true_divide(players_wta.Wins,players_wta.Wins + players_wta.Losses)
players_wta['Games'] = players_wta.Wins + players_wta.Losses
#%%
plt.style.use('fivethirtyeight')
wta['Year'] = wta.tourney_date.apply(lambda x: str(x)[0:4])
wta['Sets'] = wta.score.apply(lambda x: x.count('-'))
wta['Rank_Diff'] =  wta['loser_rank'] - wta['winner_rank']
wta['ind'] = range(len(wta))
wta['Rank_Diff_Round'] = wta.Rank_Diff.apply(lambda x: 10*round(np.true_divide(x,10)))
wta = wta.set_index('ind')

surfaces = ['Hard','Grass','Clay','Carpet']
for surface in surfaces:
    players_wta[surface + '_wins'] = players_wta.Name.apply(lambda x: len(wta[(wta.winner_name == x) & (wta.surface == surface)]))
    players_wta[surface + '_losses'] = players_wta.Name.apply(lambda x: len(wta[(wta.loser_name == x) & (wta.surface == surface)]))
    players_wta[surface + 'PCT'] = np.true_divide(players_wta[surface + '_wins'],players_wta[surface + '_losses'] + players_wta[surface + '_wins'])
    
serious_players = players_wta[players_wta.Games>40]
serious_players['Height'] = serious_players.Name.apply(lambda x: list(wta.winner_ht[wta.winner_name == x])[0])
serious_players['Best_Rank'] = serious_players.Name.apply(lambda x: min(wta.winner_rank[wta.winner_name == x]))
serious_players['Win_Aces'] = serious_players.Name.apply(lambda x: np.mean(wta.w_ace[wta.winner_name == x]))
serious_players['Lose_Aces'] = serious_players.Name.apply(lambda x: np.mean(wta.l_ace[wta.loser_name == x]))
serious_players['Aces'] = (serious_players['Win_Aces']*serious_players['Wins'] + serious_players['Lose_Aces']*serious_players['Losses'])/serious_players['Games']
wta.surface.value_counts(normalize=True).plot(kind='bar')
wta['Aces'] = wta.l_ace + wta.w_ace

plt.bar(1,np.mean(wta.Aces[wta.surface == 'Hard']))
plt.bar(2,np.mean(wta.Aces[wta.surface == 'Grass']), color = 'g')
plt.bar(3,np.mean(wta.Aces[wta.surface == 'Clay']), color ='r')
plt.bar(4,np.mean(wta.Aces[wta.surface == 'Carpet']), color ='y')
plt.ylabel('Aces per Match')
plt.xticks([1,2,3,4], ['Hard','Grass','Clay','Carpet'])
plt.title('More Aces on Grass')
wta['df'] = wta.l_df + wta.w_df

plt.bar(1,np.mean(wta.df[wta.surface == 'Hard']))
plt.bar(2,np.mean(wta.df[wta.surface == 'Grass']), color = 'g')
plt.bar(3,np.mean(wta.df[wta.surface == 'Clay']), color ='r')
plt.bar(4,np.mean(wta.df[wta.surface == 'Carpet']), color ='y')
plt.ylabel('Aces per Match')
plt.xticks([1,2,3,4], ['Hard','Grass','Clay','Carpet'])
plt.title('More double faults on  Hard ')
wta['bps'] = wta.l_bpFaced + wta.w_bpFaced

plt.bar(1,np.mean(wta.bps[wta.surface == 'Hard']))
plt.bar(2,np.mean(wta.bps[wta.surface == 'Grass']), color = 'g')
plt.bar(3,np.mean(wta.bps[wta.surface == 'Clay']), color ='r')
plt.bar(4,np.mean(wta.bps[wta.surface == 'Carpet']), color ='y')
plt.ylabel('break point saved  per surface')
plt.xticks([1,2,3,4], ['Hard','Grass','Clay','Carpet'])
plt.title('easier to break serve on Clay ')
plt.bar(1,np.mean(matches.minutes[matches.surface == 'Hard']))
plt.bar(2,np.mean(matches.minutes[matches.surface == 'Grass']), color = 'g')
plt.bar(3,np.mean(matches.minutes[matches.surface == 'Clay']), color ='r')
plt.bar(4,np.mean(matches.minutes[matches.surface == 'Carpet']), color ='y')
plt.ylabel('Aces per Match')
plt.xticks([1,2,3,4], ['Hard','Grass','Clay','Carpet'])
plt.title('less time  on Grass')
print('Average time on HARD courts ', np.mean(matches.minutes[matches.surface == 'Hard']))

print('Average time on Clay courts ', np.mean(matches.minutes[matches.surface == 'Clay']))

print('Average time on Grass courts ', np.mean(matches.minutes[matches.surface == 'Grass']))

avg_height = []
years = np.arange(2011,2016)
for year in years:
    avg_winner = np.mean(wta.winner_ht[wta.Year == str(year)])
    avg_loser = np.mean(wta.winner_ht[wta.Year == str(year)])
    avg_height.append(np.mean([avg_winner,avg_loser]))

plt.bar(years,avg_height)
plt.ylim([165,175])
plt.xlabel('Year')
plt.ylabel('Average Height')
plt.title('Are tennis players getting taller?')
pd.concat([wta['winner_ht'], wta['loser_ht']]).value_counts().head(20).plot.bar(
    figsize=(12, 4),
    title='WTA Players height '
)

wta['loser_ht'].value_counts().head(20).plot.bar(
    figsize=(12, 4),
    title='WTA hieght with Most Matches Lost'
)
wta['winner_ht'].value_counts().head(20).plot.bar(
    figsize=(12, 4),
    title='WTA Players height with Most Matches win'
)
print('Average winner height on HARD courts ', np.mean(wta.winner_ht[wta.surface == 'Hard']))
print('Average winner height on CLAY courts ', np.mean(wta.winner_ht[wta.surface == 'Clay']))
print('Average winner height on GRASS courts ', np.mean(wta.winner_ht[wta.surface == 'Grass']))
print('Average winner height on CARPET courts ', np.mean(wta.winner_ht[wta.surface == 'Carpet']))
print('Average Player Height', np.mean(serious_players.Height))
print('Average Height of no.1 Rank ',np.mean(serious_players.Height[serious_players.Best_Rank == 1]))

import numpy as np

players.set_index('DOB').resample('Y').count().Country.plot.line(
    linewidth=1, 
    figsize=(12, 4),
    title='WTA Player Year of Birth'
)
wta.winner_age.plot(kind='hist')
matchesWon = matches[(matches['round'] == 'F') & 
(matches['tourney_name'] =='US Open')| \
(matches['tourney_name'] =='French Open')|(matches['tourney_name'] =='Wimbledon')| \
                     (matches['tourney_name'] =='Australian Open')]
matchesWon.winner_age.plot(kind='hist',color='gold',label='Age',bins = 150 \
                           ,linewidth=0.01,grid=True,figsize = (12,10))
plt.xlabel('Age')              # label = name of label
plt.ylabel('No of Wins')
plt.suptitle('Number of Grandslams wins based on Age of the player', x=0.5, y=.9, ha='center', fontsize='xx-large')
df70 = wta[(wta.loser_age - wta.winner_age > 0)]
df71 = wta[(wta.loser_age - wta.winner_age < 0)]
labels = 'the winner is the youngest', 'the winner is the oldest'
values = [6589, 7419]
colors = ['red', 'pink']
explode = (0.1, 0)  # explode 1st slice
 
# Plot
plt.pie(sizes, explode=explode, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=140)
 
plt.axis('equal')
plt.show()

plt.bar(1,np.mean(matches.winner_age[matches['round'] == 'R128']))
plt.bar(2,np.mean(matches.winner_age[matches['round'] == 'R64']), color = 'g')
plt.bar(3,np.mean(matches.winner_age[matches['round'] == 'QF']), color ='r')
plt.bar(4,np.mean(matches.winner_age[matches['round'] == 'F']), color ='y')


plt.ylabel('Aces per Match')
plt.xticks([1,2,3,4], ['R128','R64','qF','F'])
plt.title('age / rounds ')
plt.style.use('fivethirtyeight')

(matches
     .assign(
         winner_seed = matches.winner_seed.fillna(0).map(lambda v: v if str.isdecimal(str(v)) else np.nan),
         loser_seed = matches.loser_seed.fillna(0).map(lambda v: v if str.isdecimal(str(v)) else np.nan)
     )
     .loc[:, ['winner_seed', 'loser_seed']]
     .pipe(lambda df: df.winner_seed.astype(float) >= df.loser_seed.astype(float))
     .value_counts()
).plot.bar(title='Higher Ranked Seed Won Match')
import re
wta['Set_1'], wta['Set_2'], wta['Set_3'] = wta['score'].str.split(' ',2).str
comeback = 0
for item,row in wta.iterrows():
	if 'R' not in str(row['Set_2']):
		if 'R' not in str(row['Set_3']) and str(row['Set_3']) != 'nan' and 'u' not in str(row['Set_3']) and str(row['Set_3']) != '6-0 6-1' and 'D' not in str(row['Set_3']):
			set_score_Set_2 = re.sub("\(\d+\)"," ",row['Set_2'])
			set_score_Set_3 = re.sub("\(\d+\)"," ",row['Set_3'])
			Set_3 = float(set_score_Set_3.split('-')[0]) - float(set_score_Set_3.split('-')[1])
			Set_2 = float(set_score_Set_2.split('-')[0]) - float(set_score_Set_2.split('-')[1])
			if Set_3 * Set_2 > 0:
				comeback += 1

print ('Comeback %% = %f'%(100*float(comeback)/float(len(wta))))

plt.bar(1,np.sum([(wta.surface == 'Hard') & ( 'RET' in str(row['score'])  ) ] ))
plt.bar(2,np.sum([(wta.surface == 'Clay') & ( 'RET' in str(row['score'])  ) ] ), color = 'r')
plt.bar(3,np.sum([(wta.surface == 'Grass') & ( 'RET' in str(row['score'])  ) ] ), color ='g')
plt.bar(4,np.sum([(wta.surface == 'Carpet') & ( 'RET' in str(row['score'])  ) ] ), color ='y')


plt.ylabel('retirement per surface')
plt.xticks([1,2,3,4], ['Hard','Clay','Grass','Carpet'])
plt.title('retirement / surface ')
plt.bar(1,np.sum([(wta['round'] == 'R128') & ( 'RET' in str(row['score'])  ) ] ))
plt.bar(2,np.sum([(wta['round'] == 'R64') & ( 'RET' in str(row['score'])  ) ] ))
plt.bar(3,np.sum([(wta['round'] == 'R32') & ( 'RET' in str(row['score'])  ) ] ), color = 'r')
plt.bar(4,np.sum([(wta['round'] == 'R16') & ( 'RET' in str(row['score'])  ) ] ))
plt.bar(5,np.sum([(wta['round'] == 'QF') & ( 'RET' in str(row['score'])  ) ] ))
plt.bar(6,np.sum([(wta['round'] == 'SF') & ( 'RET' in str(row['score'])  ) ] ), color ='g')
plt.bar(7,np.sum([(wta['round'] == 'F') & ( 'RET' in str(row['score'])  ) ] ), color ='y')


plt.ylabel('retirement per rounds')
plt.xticks([1,2,3,4,5,6], ['R128','R64','R32','R16','QF','SF','F'])
plt.title('retirement / round ')
plt.bar(1,np.sum([(wta['loser_rank'] < 30 ) & ( 'RET' in str(row['score'])  ) ] ))
plt.bar(2,np.sum([(wta['loser_rank'] >30   ) & (wta['loser_rank'] <50 ) & ( 'RET' in str(row['score'])  ) ] ))
plt.bar(3,np.sum([(wta['loser_rank'] >50   ) & (wta['loser_rank'] <100 ) & ( 'RET' in str(row['score'])  ) ] ))

plt.bar(4,np.sum([(wta['loser_rank'] > 100 ) & ( 'RET' in str(row['score'])  ) ] ))


plt.ylabel('retirement per rounds')
plt.xticks([1,2,3,4], ['<30','[30 .. 50]','[50 .. 100]','>100'])
plt.title('retirement / rank ')
plt.bar(1,np.sum (wta['winner_entry'] == 'S'  )  )  
plt.bar(2,np.sum (wta['winner_entry'] == 'Q'  )  ) 
plt.bar(3,np.sum (wta['winner_entry'] == 'WC'  )  )
plt.bar(4,np.sum (wta['winner_entry'] == 'LL'  )  ) 
plt.bar(5,np.sum (wta['winner_entry'] == 'ALT'  )  ) 



plt.ylabel('retirement per rounds')
plt.xticks([1,2,3,4,5], ['S','Q','WC','LL','ALT'])
plt.title('S , Q , WC and LL matche win ')
plt.bar(1,np.sum([(wta['round'] == 'F') & ( (wta['winner_entry'] == 'S'  )  ) ] ))
plt.bar(2,np.sum([(wta['round'] == 'F') & ( (wta['winner_entry'] == 'Q'  )  ) ] ))
plt.bar(3,np.sum([(wta['round'] == 'F') & ( (wta['winner_entry'] == 'WC'  )  ) ] ))
plt.bar(4,np.sum([(wta['round'] == 'F') & ( (wta['winner_entry'] == 'LL'  )  ) ] ))
plt.bar(5,np.sum([(wta['round'] == 'F') & ( (wta['winner_entry'] == 'ALT'  )  ) ] ))



plt.ylabel('retirement per rounds')
plt.xticks([1,2,3,4,5], ['S','Q','WC','LL','ALT'])
plt.title('S , Q , WC and LL in finals ')
plt.bar(1,np.sum([(wta['tourney_level'] == 'G') & ( (wta['winner_entry'] == 'S'  )  ) ] ))
plt.bar(2,np.sum([(wta['tourney_level'] == 'G') & ( (wta['winner_entry'] == 'Q'  )  ) ] ))
plt.bar(3,np.sum([(wta['tourney_level'] == 'G') & ( (wta['winner_entry'] == 'WC'  )  ) ] ))
plt.bar(4,np.sum([(wta['tourney_level'] == 'G') & ( (wta['winner_entry'] == 'LL'  )  ) ] ))
plt.bar(5,np.sum([(wta['tourney_level'] == 'G') & ( (wta['winner_entry'] == 'ALT'  )  ) ] ))



plt.ylabel('retirement per rounds')
plt.xticks([1,2,3,4,5], ['S','Q','WC','LL','ALT'])
plt.title('S , Q , WC and LL in GS ')