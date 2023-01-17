import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
champion = pd.read_csv('../input/champs_and_runner_ups_series_averages.csv')
champion.head()
champion.shape
len(champion)
champion.columns
champion.size
champion[['FGP','TPP','FTP']].round(2).head()
champion.isnull().sum().sum()
champion.isnull().sum().idxmax()
champion['Team'].unique().tolist()
champion.groupby(['Year','Team','Game', 'Home'], as_index=True).agg({"FG": "sum", "FGA": "sum","Win": "max", "PTS": "sum"})
champion.groupby('Team')[['Game','Win','PTS']].sum()
# Getting specific rows where the values in column 'Home' = 1,

champion[(champion.Home == 1) & (champion.index.isin([0,2,4,6,8,10]))]
# inverse operator, returns all rows where values Home =1 is not present in 0,2,4,6,8,10

champion[~((champion.Home == 1) & (champion.index.isin([0,2,4,6,8,10])))].head()
# using loc and select 1:4 will get a different result than using iloc to select rows 1:4.

champion.loc[1:4]
champion.iloc[1:4]
champion.iloc[1, 3]
champion.loc[0, ['Team', 'Year', 'Game']]
champion.loc[[0, 10, 19], :] 

# returns 1st,11th & 20th row and all columns
# iloc[row slicing, column slicing]

champion.iloc[1:6, 1:4]
# idxmin() to get the index of the min of PTS

champion.loc[champion.groupby("Team")["PTS"].idxmin()]
# alternate method to idxmin()

champion.sort_values(by="PTS").groupby("Team", as_index=False).first()
#  idxmax() to get the index of the max of PTS

champion.loc[champion.groupby("Team")["PTS"].idxmax()]
# alternate method to idxmax()

champion.sort_values(by="PTS").groupby("Team", as_index=False).last()
champion.sort_values(by=(['Team', 'PTS']), ascending=False).head()
# select all rows that have a Win = 1

champion[champion.Win == 1].head()
# select all rows that do not contain Win = 1, ie. returns all rows with Win = 0. 

# Win = 0 are the losers

champion[champion.Win != 1].head()
# finds all rows with Team = 'Lakers'

champion[champion['Team'].isin(['Lakers'])].head()
champion.groupby('Year')['Win'].count().sum()
year_wins = pd.crosstab(champion.Win, champion.Year, margins=True)

year_wins.T
dfhome_1 = champion[champion['Home'] == 1 ]

dfhome_1.head()
champion.groupby('Team')['Win'].agg(np.sum).plot(kind = 'bar')
over = champion.groupby('Team', as_index=True).agg({"PTS": "sum"})

over['PTS'].plot(kind='bar')
champion[(champion['Home']>0) & (champion['Year'] == 2000) & (champion['Team'] == 'Lakers')]
champion[(champion['Home']>0) & (champion['Team'] == 'Lakers') | (champion['Team'] == 'Bulls')].head()
champion[champion["Team"] == "Lakers"]["PTS"].value_counts().plot(kind="bar")
champion.loc[10:15]
champion.iloc[5:10,0:7] 
champion.Win.nlargest(5)
# check the type of an object like this

type(champion.Win.nlargest(5))
champion['Win'].nlargest(5).dtype
champion.index
champion.loc[6]
champion_pieces = [champion[:3], champion[3:7], champion[7:5]]

champion_pieces
champion[champion['TPP'].notnull()].head()
champion[champion['TPP'].isnull()]
champion.ix[2, 'Team']
champion.Team.ix[2]
champion.describe()   
champion.info()   
champion.Team.str.len()
champion.groupby('Team').agg(['min', 'max'])
# Min PTS by all teams in Home = 0 (away matches) and 1(home matches)

table = pd.pivot_table(champion,values=['PTS'],index=['Home'],columns=['Team'],aggfunc=np.min,margins=True)

table.T
# Max PTS by all teams in Home = 0 (away matches) and 1(home matches)

table = pd.pivot_table(champion,values=['PTS'],index=['Home'],columns=['Team'],aggfunc=np.max,margins=True)

table.T
champion.groupby(['Team','Year']).sum()
champion.groupby(['Year']).groups.keys()
len(champion.groupby(['Year']).groups[1980])
champ_runnerup = pd.read_csv('../input/champs_and_runner_ups_series_averages.csv')

champ_runnerup.head()
champ_runnerup.shape
champ_runnerup.columns
champ_runnerup[['Year', 'Status','Team','PTS']]
# convert the PTS field from float to an integer 

champ_runnerup['PTS'] = champ_runnerup['PTS'].astype('int64')

champ_runnerup['PTS'].dtype
champ_runnerup[['Year', 'Status','Team','PTS']].head()
champs = champ_runnerup[champ_runnerup['Status'] == 'Champion'].groupby('Year') ['Team'].sum()

champs.head()
ch = champs.value_counts()

ch1 = ch.to_frame().reset_index()

ch1
type(ch1)
runnerup = champ_runnerup[champ_runnerup['Status'] == 'Runner Up'].groupby('Year') ['Team'].sum()

runnerup.head()
ru = runnerup.value_counts()

ru1 = ru.to_frame().reset_index()

ru1
finalteams = pd.merge(ch1,ru1, on = 'index', how = 'outer')

finalteams
# Finalists in the tournament

final_teams = pd.concat([ch1, ru1], axis=1, ignore_index=True)

final_teams
# Finding Nan values 

final_teams[final_teams.isnull().any(axis=1)]
final_teams[pd.isnull(final_teams).any(axis=1)]
gc = champ_runnerup.groupby(['Status'])

gc.get_group('Runner Up').head().round()