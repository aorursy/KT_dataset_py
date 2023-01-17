# workspace prep 
import pandas as pd 
import matplotlib.pyplot as plt
%matplotlib inline
# import data & look at data
play = pd.read_csv('../input/NFL-Punt-Analytics-Competition/play_information.csv')
player_role =pd.read_csv("../input/NFL-Punt-Analytics-Competition/play_player_role_data.csv") 
player = pd.read_csv('../input/NFL-Punt-Analytics-Competition/player_punt_data.csv')
# data quality check
dfs = ([play, player_role, player])

for df in dfs:
    
    print("dataframe information")
    nan_count = df.apply(lambda x: x.count(), axis=0)
    if sum(nan_count) == len(df)*len(df.columns):
        print('No Missing Values')
    elif nan_count != len(df):
        print(nan_count)
    
    print(df.shape)
    print(df.info())
    print(df.head())
# join on proper keys - no missing data
# first two on GSISID
# then on GameKey and PlayID to get the full data set for each player    
full_players = player.merge(player_role, left_on='GSISID',right_on='GSISID',how = 'left')
full_set = full_players.merge(play, left_on=['GameKey','PlayID'],
                              right_on = ['GameKey','PlayID'],
                              how = 'left')
print(full_set.info())
print(full_set.isna().sum())
full_set.head()
#drop the null values
df=full_set.dropna()
# need to split the score column and also the home and away column into 4 
# diff columns 
df['Home_Team_Visit_Team'] = df['Home_Team_Visit_Team'].astype(str)
df['Score_Home_Visiting'] = df['Score_Home_Visiting'].astype(str)
# splits
df=df.join(df['Home_Team_Visit_Team'].str.split('-', 1, expand=True).rename(columns={0:'Home',1:'Away'}))
df=df.join(df['Score_Home_Visiting'].str.split(' - ', 1, expand=True).rename(columns={0:'Home_score',1:'Away_score'}))

# Date
df["Game_Date"] = pd.to_datetime(df["Game_Date"], format = '%m/%d/%Y')

# drop columns that were split
df = df.drop(['Home_Team_Visit_Team'], axis = 1)
df = df.drop(['Score_Home_Visiting'], axis = 1)
df.head()
# Extract key information from the Play Description string variable using re
df['PlayDescription'] = df['PlayDescription'].astype(str)

# punt length 
import re 
punt_length = []
for row in df['PlayDescription']:
    match = re.search('punts (\d+)', row)
    if match:
        punt_length.append(match.group(1))
    elif match is None:
        punt_length.append(0)
        
# return length
return_length = []
for row in df['PlayDescription']:
    match = re.search('for (\d+)', row)
    if match:
        return_length.append(match.group(1))
    elif match is None:
        return_length.append(0)
            
# fair catch
fair_catch = []
for row in df['PlayDescription']:
    match = re.search('fair catch', row)
    if match:
        fair_catch.append(1)
    elif match is None:
        fair_catch.append(0)

# injury
injury = []
for row in df['PlayDescription']:
    match = re.search('injured', row)
    if match:
        injury.append(1)
    elif match is None:
            injury.append(0)

# penalty         
penalty = []
for row in df['PlayDescription']:
    if 'Penalty' in row.split():
        penalty.append(1)
    elif 'PENALTY' in row.split():
        penalty.append(1)
    elif 'Penalty' not in row.split():
        penalty.append(0)
    elif 'PENALTY' not in row.split():
        penalty.append(0)
        

# downed
downed = []
for row in df['PlayDescription']:
    match = re.search('downed', row)
    if match:
        downed.append(1)
    elif match is None:
        downed.append(0)

# fumble
fumble = []
for row in df['PlayDescription']:
    match = re.search('FUMBLES', row)
    if match:
        fumble.append(1)
    elif match is None:
        fumble.append(0)

# muff
muff = []
for row in df['PlayDescription']:
    match = re.search('MUFFS', row)
    if match:
        muff.append(1)
    elif match is None:
        muff.append(0)

# Touchback
touchback = []
for row in df['PlayDescription']:
    match = re.search('Touchback', row)
    if match:
        touchback.append(1)
    elif match is None:
        touchback.append(0)

# Touchdown
touchdown = []
for row in df['PlayDescription']:
    match = re.search('TOUCHDOWN', row)
    if match:
        touchdown.append(1)
    elif match is None:
        touchdown.append(0)

# add new columns to the df 
df["punt_length"] = punt_length
df["return_length"] = return_length
df["fair_catch"] = fair_catch
df["injury"] = injury
df["penalty"] = penalty
df["downed"] = downed
df["fumble"] = fumble
df['muff'] = muff
df['touchback'] = touchback
df['touchdown'] = touchdown
df.head()
import feather
df_final = feather.read_dataframe('../input/feathered-ngs/ngs.feather')
print(df_final.shape)
df_final.head()
new_df = df.merge(df_final.drop_duplicates(subset=['GSISID','GameKey','PlayID']), how='left',
                  left_on=['GSISID','GameKey','PlayID','Season_Year_x'], right_on = ['GSISID','GameKey','PlayID','Season_Year'])
del df_final
new_df.head()
# game data
game = pd.read_csv('../input/NFL-Punt-Analytics-Competition/game_data.csv')

game_with_new = new_df.merge(game, how = 'left', left_on = "GameKey", 
                             right_on = "GameKey")
# columns to keep 
keep = ['GSISID', 'Number', 'Position','Season_Year_x', 'GameKey', 'PlayID',
       'Role', 'Game_Date_x', 'Week_x',
       'Game_Clock', 'YardLine', 'Quarter', 'Play_Type', 'Poss_Team',
       'Home', 'Away', 'Home_score', 'Away_score',
       'punt_length', 'return_length', 'fair_catch', 'injury', 'penalty',
       'downed', 'fumble', 'muff', 'touchback','touchdown','x',
       'y', 'dis', 'o', 'dir', 'Event', 'Season_Type_y',
       'Game_Day', 'Game_Site', 'Start_Time',
       'Home_Team', 'Visit_Team', 'Stadium',
       'StadiumType', 'Turf', 'GameWeather', 'Temperature', 'OutdoorWeather'
       ]
df_clean = game_with_new[keep]
del game_with_new

# rename columns
headers = ['GSISID', 'Number', 'Position', 'Season_Year','Season_Year_x', 'GameKey', 'PlayID',
       'Role', 'Game_Date', 'Week',
       'Game_Clock', 'YardLine', 'Quarter', 'Play_Type', 'Poss_Team',
       'Home', 'Away', 'Home_score', 'Away_score',
       'punt_length', 'return_length', 'fair_catch', 'injury', 'penalty',
       'downed', 'fumble', 'muff', 'touchback','touchdown','x',
       'y', 'dis', 'o', 'dir', 'Event', 'Season_Type',
       'Game_Day', 'Game_Site', 'Start_Time',
       'Home_Team', 'Visit_Team', 'Stadium',
       'StadiumType', 'Turf', 'GameWeather', 'Temperature', 'OutdoorWeather'
       ]
df_clean.columns = headers
print(df_clean.dtypes)
df_clean[["punt_length", "return_length"]] = df_clean[["punt_length", "return_length"]].apply(pd.to_numeric)
df_clean = df_clean.drop(columns='Season_Year_x')
df_clean.head()
# lets look at how many games and punts there are 
games = len(df_clean['GameKey'].unique().tolist())
print('There are ' + str(games) + ' games in the dataset.')
punts = len(df_clean['PlayID'].unique().tolist())
print('There are ' + str(punts) + ' punts in the dataset.')
print('On average, there are ' + str(punts/games) + ' punts per game.')
# let's start with the injury field
no_injuries = df_clean.loc[df_clean['injury'] == 0]
injuries = df_clean.loc[df_clean['injury'] == 1]
# average function 
def avg(lst):
    return sum(lst)/len(lst)

# Number of injuries
print('There are ' + str(len(injuries['PlayID'].unique().tolist())) + ' injuries in the dataset.')

# lets look at the average punt length and return lenth for both new dfs
print('The average punt length for a play with an injury is ' + str(avg(injuries['punt_length'].unique().tolist())))
print('The average punt length for a play without an injury is ' + str(avg(no_injuries['punt_length'].unique().tolist())))
print('The average punt return for a play with an injury is ' + str(avg(injuries['return_length'].unique().tolist())))
print('The average punt return for a play without an injury is ' + str(avg(no_injuries['return_length'].unique().tolist())))
#injuries by gameday
total_injuries = injuries.groupby('Game_Day')['PlayID'].nunique()
total_no_injuries = no_injuries.groupby('Game_Day')['PlayID'].nunique()
print('On Fridays, injuires occured on ' + str(3/203) + ' percent of punt plays.')
print('On Mondays, injuires occured on ' + str(4/312) + ' percent of punt plays.')
print('On Saturdays, injuires occured on ' + str(7/602) + ' percent of punt plays.')
print('On Sundays, injuires occured on ' + str(56/2648) + ' percent of punt plays.')
print('On Thursdays, injuires occured on ' + str(16/871) + ' percent of punt plays.')
# injuries by game site
injuries.groupby('Game_Site')['PlayID'].nunique().plot(kind='bar',figsize=(18, 16))
plt.xlabel('Week')
plt.ylabel('Injuries')
plt.title('Injuries per Location')
plt.show()
# injuries by season year
injuries.groupby('Season_Year')['PlayID'].nunique().plot(kind='bar',figsize=(12, 10))
plt.xlabel('Year')
plt.ylabel('Injuries')
plt.title('Injuries (2016-2017)')
plt.show()
# injuries by muff
data = injuries.groupby('muff')['PlayID'].nunique().plot(kind='bar', figsize=(12, 10))
plt.xlabel('muff')
plt.ylabel('Injuries')
plt.title('Injuries on Muffs')
plt.show()
# injuries by fumble
data = injuries.groupby('fumble')['PlayID'].nunique().plot(kind='bar', figsize=(12, 10))
plt.xlabel('Fumble')
plt.ylabel('Injuries')
plt.title('Injuries on Fumbles')
plt.show()
# injuries by touchdown
data = injuries.groupby('touchdown')['PlayID'].nunique().plot(kind='bar',figsize=(12, 10))
plt.xlabel('Touchdowns')
plt.ylabel('Injuries')
plt.title('Injuries on Touchdowns')
plt.show()

# injuries by week
data = injuries.groupby('Week')['PlayID'].nunique().plot(kind='bar',figsize=(18, 16))
plt.xlabel('Week')
plt.ylabel('Injuries')
plt.title('Injuries per Week')
plt.show()
# injuries by quarter
injuries.groupby('Quarter')['PlayID'].nunique().plot(kind='bar',figsize=(12, 10))
plt.xlabel('Week')
plt.ylabel('Injuries')
plt.title('Injuries per Quarter')
plt.show()
# injuries per season type
injuries.groupby('Season_Type')['PlayID'].nunique().plot(kind='bar', figsize=(12, 10))
plt.xlabel('Season Type')
plt.ylabel('Injuries')
plt.title('Injuries in Pre, Post, and Regular Season Games')
plt.show()
# lets look at teams who have the most injuries
fig, axes = plt.subplots(nrows=1, ncols=2, sharey = True)

injuries.groupby('Home')['PlayID'].nunique().plot(figsize=(18, 16),ax=axes[0],kind='bar')
plt.ylabel('Injuries')
plt.suptitle('Frequency of Injuries by Home (Left) and Away (Right)')
injuries.groupby('Away')['PlayID'].nunique().plot(figsize=(18, 16),ax=axes[1],kind='bar')
plt.show()
# lets look at punt length
cols = ['GameKey', 'PlayID','punt_length','injury']
punt_length = df_clean[cols]
punt_length = punt_length.drop_duplicates()

# histogram for punt length on injuires
fig, axes = plt.subplots(nrows=1, ncols=2)

punt_length['punt_length'].loc[punt_length['injury']==1].plot(ax=axes[0],kind='hist', bins = 10, color = 'red', edgecolor = 'black', figsize=(18, 16))
punt_length['punt_length'].loc[punt_length['injury']==0].plot(ax=axes[1],kind='hist', bins = 10, edgecolor = 'black', figsize=(18, 16))
plt.suptitle('Frequency of Injuries (Red) and Non-Injuries (Blue) by Return length')
plt.show()
# same process for return length
cols = ['GameKey', 'PlayID','return_length','injury']
return_length= df_clean[cols]
return_length= return_length.drop_duplicates()

# histogram for return length on injuires
fig, axes = plt.subplots(nrows=1, ncols=2)

return_length['return_length'].loc[return_length['injury']==1].plot(ax=axes[0],kind='hist', bins = 15, color = 'red', edgecolor='black',figsize=(18, 16))
return_length['return_length'].loc[return_length['injury']==0].plot(ax=axes[1],kind='hist', bins = 15, edgecolor='black',figsize=(18, 16))
plt.suptitle('Frequency of Injuries (Red) and Non-Injuries (Blue) by Return length')
plt.show()
# injuries by fair catch
injuries.groupby('fair_catch')['PlayID'].nunique().plot(kind='bar', figsize=(12, 10))
plt.xlabel('Fair Catch')
plt.ylabel('Injuries')
plt.title('Injuries on Fair Catches')
plt.show()