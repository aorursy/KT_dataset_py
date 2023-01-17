# linear algebra

import numpy as np

# dataframes

import pandas as pd

# ploting

import matplotlib.pyplot as plt

# math functions

import math

# visualisation

import seaborn as sns

# visualisation

import plotly.express as px

# visualisation

import plotly.graph_objects as go



# disable warnings

# pd.set_option('mode.chained_assignment', None)
# detailed tournament games data 

tourney_results = pd.read_csv('/kaggle/input/march-madness-analytics-2020/MDataFiles_Stage2/MNCAATourneyDetailedResults.csv')

# detailed tournament games data (for infographics)

all_tourney_results = tourney_results.copy()

# compact tournament games data 

tourney_compact_results = pd.read_csv('/kaggle/input/march-madness-analytics-2020/MDataFiles_Stage2/MNCAATourneyCompactResults.csv')

# seeds data

seeds = pd.read_csv('/kaggle/input/march-madness-analytics-2020/MDataFiles_Stage2/MNCAATourneySeeds.csv')

# regular season data (cut it since there was no tournament in 2020)

regular_results = pd.read_csv('/kaggle/input/march-madness-analytics-2020/MDataFiles_Stage2/MRegularSeasonDetailedResults.csv')

regular_results = regular_results[regular_results.Season<2020]

# game cities data

game_cities = pd.read_csv('/kaggle/input/march-madness-analytics-2020/MDataFiles_Stage2/MGameCities.csv')

# city names

city_names = pd.read_csv('/kaggle/input/march-madness-analytics-2020/MDataFiles_Stage2/Cities.csv')

# team names

team_names = pd.read_csv('/kaggle/input/march-madness-analytics-2020/MDataFiles_Stage2/MTeams.csv')

# city coordinates

lat_long_df = pd.read_csv('/kaggle/input/simplemaps-us-cities-database/uscities.csv')
# 1

# "teams with their hometown coordinates" dataframe (location_data)



# a little tweak to merge cities and coordinates

city_names = city_names.replace('Mt. Pleasant', 'Mount Pleasant')



#get all teams that won at least a game in Years 2010-2019 and merge it with Cities ID, city and team names

location_data = pd.merge(regular_results[(regular_results['Season']>2009)],game_cities[game_cities['CRType'] == 'Regular'], on=['Season','DayNum','WTeamID','LTeamID'])

location_data = location_data[location_data['WLoc'] == 'H'].drop_duplicates(subset = ['WTeamID','CityID'], keep = 'last').drop(['DayNum', 'LTeamID','LTeamID','Season','LScore','WScore','WLoc','CRType','NumOT'],axis = 1)

location_data = pd.merge(pd.merge(location_data,city_names, on = 'CityID'),team_names, left_on = 'WTeamID',right_on = 'TeamID').drop(['FirstD1Season','WTeamID','LastD1Season'], axis = 1)

location_data = location_data[['TeamID','TeamName','CityID','City','State']]



#leave only needed information in lat_long_df

lat_long_df = lat_long_df[['city','state_id','lat','lng']]



#merge cities and coordinates

location_data = pd.merge(location_data, lat_long_df, left_on = ['City','State'], right_on = ['city','state_id'], how = 'left')



#add missing data

location_data.at[5,'lat'] = 44.8834

location_data.at[5,'lng'] = -68.6708

location_data.at[15,'lat'] = 40.6960

location_data.at[15,'lng'] = -73.9932

location_data.at[23,'lat'] = 41.1414

location_data.at[23,'lng'] = -73.2637

location_data.at[25,'lat'] = 41.1414

location_data.at[25,'lng'] = -73.2637

location_data.at[26,'lat'] = 40.5080

location_data.at[26,'lng'] = -80.2060

location_data.at[30,'lat'] = 34.4587

location_data.at[30,'lng'] = -82.8687

location_data.at[32,'lat'] = 40.0349

location_data.at[32,'lng'] = -75.3373

location_data.at[34,'lat'] = 40.7489

location_data.at[34,'lng'] = -74.2609

location_data.at[39,'lat'] = 34.7867

location_data.at[39,'lng'] = -86.5698

location_data.at[63,'lat'] = 42.0841

location_data.at[63,'lng'] = -76.0542

location_data.at[85,'lat'] = 40.8870

location_data.at[85,'lng'] = -73.9012

location_data.at[92,'lat'] = 43.7023

location_data.at[92,'lng'] = -72.2895

location_data.at[99,'lat'] = 41.3858

location_data.at[99,'lng'] = -72.9042

location_data.at[148,'lat'] = 38.9904

location_data.at[148,'lng'] = -104.8606

location_data.at[153,'lat'] = 41.9221

location_data.at[153,'lng'] = -71.5496

location_data.at[199,'lat'] = 40.5608

location_data.at[199,'lng'] = -74.4655

location_data.at[202,'lat'] = 31.8190

location_data.at[202,'lng'] = -91.0500

location_data.at[211,'lat'] = 42.3233

location_data.at[211,'lng'] = -71.1423

location_data.at[215,'lat'] = 34.2379

location_data.at[215,'lng'] = -118.5301

location_data.at[288,'lat'] = 43.1338

location_data.at[288,'lng'] = -70.9269

location_data.at[298,'lat'] = 42.3403

location_data.at[298,'lng'] = -72.4968

location_data.at[331,'lat'] = 41.7620

location_data.at[331,'lng'] = -72.7420

location_data.at[362,'lat'] = 40.8121

location_data.at[362,'lng'] = -77.8561



#drop not needed columns

location_data.drop(['city','state_id'], axis = 1, inplace = True)



#add data of 'no teams' cities 

location_data = location_data.append([{'TeamID': 0,'TeamName':'none','CityID':4008,'City':'Anaheim','State':'CA','lat':33.839,'lng':-117.8573}], ignore_index = True)

location_data = location_data.append([{'TeamID': 0,'TeamName':'none','CityID':4019,'City':'Auburn Hills','State':'MI','lat':42.6735,'lng':-83.2447}], ignore_index = True)

location_data = location_data.append([{'TeamID': 0,'TeamName':'none','CityID':4130,'City':'Glendale','State':'AZ','lat':33.5797,'lng':-112.2258}], ignore_index = True)

location_data = location_data.append([{'TeamID': 0,'TeamName':'none','CityID':4254,'City':'Oklahoma City','State':'OK','lat':35.4676,'lng':-97.5137}], ignore_index = True)



location_data.to_csv('location_data.csv',index = None)
# 2

# "tournament games with calculated distances between colleges' hometowns and game cities" dataframe (tourney_results)



# since game location data stays consistently known since 2010, we cut our analysis to 2010-2019 period

tourney_results = tourney_results[tourney_results['Season'] > 2009]



# merge tournament results with game city coordinates

tourney_results = pd.merge(tourney_results,game_cities[game_cities['CRType'] != 'Regular'].drop(['LTeamID','CRType'], axis = 1), on = ['Season','DayNum','WTeamID'], how = 'left')

tourney_results = tourney_results.rename(columns = {'CityID':'GameCityID'})

tourney_results = pd.merge(tourney_results, location_data.drop_duplicates(subset = 'CityID').drop('TeamID', axis = 1), left_on = 'GameCityID', right_on = 'CityID', how = 'left')

tourney_results.drop(['CityID','State','TeamName','City'], axis = 1, inplace = True)

tourney_results.rename(columns = {'lat':'GameCityLat','lng':'GameCityLng'}, inplace = True)



# merge tournament results with winning and losing teams' hometown coordinates

tourney_results = pd.merge(tourney_results, location_data.drop_duplicates(subset = 'TeamID'), left_on = 'WTeamID', right_on = 'TeamID', how = 'left')

tourney_results.drop(['TeamID','CityID','State','TeamName','City'], axis = 1, inplace = True)

tourney_results.rename(columns = {'lat':'WCityLat','lng':'WCityLng'}, inplace = True)



tourney_results = pd.merge(tourney_results, location_data.drop_duplicates(subset = 'TeamID'), left_on = 'LTeamID', right_on = 'TeamID', how = 'left')

tourney_results.drop(['TeamID','CityID','State','TeamName','City'], axis = 1, inplace = True)

tourney_results.rename(columns = {'lat':'LCityLat','lng':'LCityLng'}, inplace = True)



# save data in "detailed_tourney_results" and "games_and_coordinates" dataframes

detailed_tourney_results = tourney_results.copy()

games_and_coordinates = tourney_results[['Season','DayNum','WTeamID','LTeamID','GameCityID','GameCityLat','GameCityLng','WCityLat','WCityLng','LCityLat','LCityLng']]



# saving infromation about all of game cities in "neutral_loc" dataframe

neutral_loc = detailed_tourney_results[['GameCityID','GameCityLat','GameCityLng']]

neutral_loc = pd.merge(neutral_loc, city_names, left_on = 'GameCityID', right_on = 'CityID')

neutral_loc.drop('GameCityID', axis = 1, inplace = True)

neutral_loc.rename(columns = {'GameCityLat':'lat', 'GameCityLng':'lng'}, inplace = True)



# get every coordinate to radians

tourney_results['GameCityLat'] = np.radians(tourney_results['GameCityLat'])

tourney_results['GameCityLng'] = np.radians(tourney_results['GameCityLng'])

tourney_results['WCityLat'] = np.radians(tourney_results['WCityLat'])

tourney_results['WCityLng'] = np.radians(tourney_results['WCityLng'])

tourney_results['LCityLat'] = np.radians(tourney_results['LCityLat'])

tourney_results['LCityLng'] = np.radians(tourney_results['LCityLng'])



# counting distances in miles



wdlon = tourney_results['GameCityLng'] - tourney_results['WCityLng']

wdlat = tourney_results['GameCityLat'] - tourney_results['WCityLat']

ldlon = tourney_results['GameCityLng'] - tourney_results['LCityLng']

ldlat = tourney_results['GameCityLat'] - tourney_results['LCityLat']



wtemp = 2 * np.arctan2(np.sqrt(np.sin(wdlat / 2)**2 + np.cos(tourney_results['WCityLat']) * np.cos(tourney_results['GameCityLat']) * np.sin(wdlon / 2)**2), np.sqrt(1 - np.sin(wdlat / 2)**2 + np.cos(tourney_results['WCityLat']) * np.cos(tourney_results['GameCityLat']) * np.sin(wdlon / 2)**2))

ltemp = 2 * np.arctan2(np.sqrt(np.sin(ldlat / 2)**2 + np.cos(tourney_results['LCityLat']) * np.cos(tourney_results['GameCityLat']) * np.sin(ldlon / 2)**2), np.sqrt(1 - np.sin(ldlat / 2)**2 + np.cos(tourney_results['LCityLat']) * np.cos(tourney_results['GameCityLat']) * np.sin(ldlon / 2)**2))



R = 3958.8 # Earth radius in miles

tourney_results['WDistance'] = R * wtemp

tourney_results['LDistance'] = R * ltemp



# get rid of coordinates in "tourney results" dataframe



tourney_results.drop(['GameCityID','GameCityLat','GameCityLng','WCityLat','WCityLng','LCityLat','LCityLng'],axis = 1, inplace = True)

tourney_results = tourney_results.dropna()



# count Relative Proximity



tourney_results['Relative_proximity'] = tourney_results['LDistance'] - tourney_results['WDistance'] 



# save detailed data

detailed_tourney_results = tourney_results.copy()



# cut not needed

tourney_results = tourney_results[['Season','DayNum','WTeamID','LTeamID','WDistance','LDistance','Relative_proximity']]
# 3



# reset "tourney_results" dataframe

tourney_results_reset = pd.read_csv('/kaggle/input/march-madness-analytics-2020/MDataFiles_Stage2/MNCAATourneyDetailedResults.csv')

tourney_results_reset = tourney_results_reset[tourney_results_reset['Season']>2009]



# get information about last loss for every team

last_loss = regular_results.drop_duplicates(['Season','LTeamID'], keep = 'last')[['Season','DayNum','LTeamID']]

last_loss.rename(columns = {'LTeamID':'TeamID', 'DayNum':'LastLoss'}, inplace = True)

last_loss = last_loss[last_loss.Season > 2009]



# get information about last win for every team

last_win = regular_results.drop_duplicates(['Season','WTeamID'], keep = 'last')[['Season','DayNum','WTeamID']]

last_win.rename(columns = {'WTeamID':'TeamID', 'DayNum':'LastWin'}, inplace = True)

last_win = last_win[last_win.Season > 2009]



# get information about last game for every team

last_games = pd.merge(last_loss, last_win)

last_games['LastGame'] = last_games[['LastLoss','LastWin']].max(axis = 1)

last_games = last_games[['Season','TeamID','LastGame']]



# rename columns of "games_and_coordinates" dataframe

games_and_coordinates = games_and_coordinates.rename(columns = {'WCityLat':'WTeamLat','WCityLng':'WTeamLng','LCityLat':'LTeamLat','LCityLng':'LTeamLng'})



# merge "games_and_coordinates" with data of last games

games_and_coordinates = pd.merge(games_and_coordinates, last_games, left_on = ['Season','WTeamID'], right_on = ['Season', 'TeamID'])

games_and_coordinates.rename(columns = {'LastGame':'WLastGame'}, inplace = True)

games_and_coordinates.drop('TeamID', axis = 1, inplace = True)



games_and_coordinates = pd.merge(games_and_coordinates, last_games, left_on = ['Season','LTeamID'], right_on = ['Season', 'TeamID'])

games_and_coordinates.rename(columns = {'LastGame':'LLastGame'}, inplace = True)

games_and_coordinates.drop('TeamID', axis = 1, inplace = True)







# update data of current location

for i in range(2010,2020):

    #list of teams

    teams_list = set(games_and_coordinates[games_and_coordinates['Season'] == i].LTeamID.unique()).union(set(games_and_coordinates[games_and_coordinates['Season'] == i].WTeamID.unique()))

    for j in teams_list:

        #list of games by team

        temp = games_and_coordinates[((games_and_coordinates.WTeamID == j)|(games_and_coordinates.LTeamID == j)) & (games_and_coordinates.Season == i)].sort_values('DayNum')

        if temp.shape[0] > 1:

            k = 0

            while k < temp.shape[0]-1:

                # if k game is in "First Four" period (DayNum 134 or 135)

                if temp.iloc[k].DayNum in [134,135]:

                    next_game_row = temp.iloc[k+1].name

                    current_day = temp.iloc[k].DayNum

                    gcLat = temp.iloc[k].GameCityLat

                    gcLng = temp.iloc[k].GameCityLng

                    # rewrite location of winning team in next round and last game day date

                    if (temp.iloc[1].WTeamID == j):

                        games_and_coordinates.at[next_game_row, 'WLastGame'] = current_day

                        games_and_coordinates.at[next_game_row, 'WTeamLat'] = gcLat

                        games_and_coordinates.at[next_game_row, 'WTeamLng'] = gcLng

                    # rewrite location of losing team in next round and last game day date

                    else:

                        games_and_coordinates.at[next_game_row, 'LLastGame'] = current_day

                        games_and_coordinates.at[next_game_row, 'LTeamLat'] = gcLat

                        games_and_coordinates.at[next_game_row, 'LTeamLng'] = gcLng

                    k = k + 1

                # rewrite last game day date

                else:

                    next_game_row = temp.iloc[k+1].name

                    current_day = temp.iloc[k].DayNum

                    if (temp.iloc[k+1].WTeamID == j):

                        games_and_coordinates.at[next_game_row, 'WLastGame'] = current_day

                    else:

                        games_and_coordinates.at[next_game_row, 'LLastGame'] = current_day

                    k = k + 1



# calculate distances

games_and_coordinates['GameCityLat'] = np.radians(games_and_coordinates['GameCityLat'])

games_and_coordinates['GameCityLng'] = np.radians(games_and_coordinates['GameCityLng'])

games_and_coordinates['WTeamLat'] = np.radians(games_and_coordinates['WTeamLat'])

games_and_coordinates['WTeamLng'] = np.radians(games_and_coordinates['WTeamLng'])

games_and_coordinates['LTeamLat'] = np.radians(games_and_coordinates['LTeamLat'])

games_and_coordinates['LTeamLng'] = np.radians(games_and_coordinates['LTeamLng'])



wdlon = games_and_coordinates['GameCityLng'] - games_and_coordinates['WTeamLng']

wdlat = games_and_coordinates['GameCityLat'] - games_and_coordinates['WTeamLat']

ldlon = games_and_coordinates['GameCityLng'] - games_and_coordinates['LTeamLng']

ldlat = games_and_coordinates['GameCityLat'] - games_and_coordinates['LTeamLat']



wtemp = 2 * np.arctan2(np.sqrt(np.sin(wdlat / 2)**2 + np.cos(games_and_coordinates['WTeamLat']) * np.cos(games_and_coordinates['GameCityLat']) * np.sin(wdlon / 2)**2), np.sqrt(1 - np.sin(wdlat / 2)**2 + np.cos(games_and_coordinates['WTeamLat']) * np.cos(games_and_coordinates['GameCityLat']) * np.sin(wdlon / 2)**2))

ltemp = 2 * np.arctan2(np.sqrt(np.sin(ldlat / 2)**2 + np.cos(games_and_coordinates['LTeamLat']) * np.cos(games_and_coordinates['GameCityLat']) * np.sin(ldlon / 2)**2), np.sqrt(1 - np.sin(ldlat / 2)**2 + np.cos(games_and_coordinates['LTeamLat']) * np.cos(games_and_coordinates['GameCityLat']) * np.sin(ldlon / 2)**2))



games_and_coordinates['WDistance'] = R * wtemp

games_and_coordinates['LDistance'] = R * ltemp



# count recovery time for winning and losing team and their difference

games_and_coordinates['WRecoveryTime'] = games_and_coordinates.DayNum - games_and_coordinates.WLastGame

games_and_coordinates['LRecoveryTime'] = games_and_coordinates.DayNum - games_and_coordinates.LLastGame

games_and_coordinates['Time_advantage'] = games_and_coordinates['WRecoveryTime'] - games_and_coordinates['LRecoveryTime']



# merge "games_and_coordinates" with seeds data of every team

games_and_coordinates = pd.merge(games_and_coordinates , seeds,left_on = ['Season','WTeamID'], right_on=['Season','TeamID'])

games_and_coordinates['WSeed'] = pd.to_numeric(games_and_coordinates ['Seed'].str[1:3])

games_and_coordinates.drop(['TeamID', 'Seed'],axis = 1, inplace = True)



games_and_coordinates  = pd.merge(games_and_coordinates , seeds,left_on = ['Season','LTeamID'], right_on=['Season','TeamID'])

games_and_coordinates ['LSeed'] = pd.to_numeric(games_and_coordinates ['Seed'].str[1:3])

games_and_coordinates.drop(['TeamID','Seed'],axis = 1, inplace = True)



# count total distance traveled

teams_travel_distances = pd.merge(games_and_coordinates[games_and_coordinates.DayNum.isin([134,135,136,137,143,144,152])].groupby('LTeamID')['LDistance'].sum().reset_index(),games_and_coordinates[games_and_coordinates.DayNum.isin([134,135,136,137,143,144,152])].groupby('WTeamID')['WDistance'].sum().reset_index(), left_on='LTeamID', right_on='WTeamID', how = 'left').fillna(0)

teams_travel_distances['Distance'] = teams_travel_distances['LDistance'] + teams_travel_distances['WDistance']

teams_travel_distances.drop(['LDistance','WTeamID','WDistance'], axis = 1, inplace = True)

teams_travel_distances.rename(columns = {'LTeamID':'TeamID','LTeamLng':'lng', 'LTeamLat':'lat'}, inplace = True)



# count number of wins in after travel games

team_tourney_wins = games_and_coordinates[games_and_coordinates.DayNum.isin([134,135,136,137,143,144,152])].groupby('WTeamID').count().Season.rename(columns = {'Season':'Wins'})

team_tourney_wins.name = 'Wins'

team_tourney_wins = team_tourney_wins.reset_index().rename(columns = {'WTeamID':'TeamID'})



# count number of loses in after travel games

team_tourney_loses = games_and_coordinates[games_and_coordinates.DayNum.isin([134,135,136,137,143,144,152])].groupby('LTeamID').count().Season.rename(columns = {'Season':'Loses'})

team_tourney_loses.name = 'Loses'

team_tourney_loses = team_tourney_loses.reset_index().rename(columns = {'LTeamID':'TeamID'})



# create dataframe of loses and wins for every team

team_tourney_games = pd.merge(team_tourney_loses, team_tourney_wins, how = 'left')

team_tourney_games = team_tourney_games.fillna(0)



# save full dataframe for model

X = games_and_coordinates.copy()



# leave only games after travel

games_and_coordinates = games_and_coordinates[games_and_coordinates.DayNum.isin([134,135,136,137,143,144,152])]
# count amount of wins for each team

team_tourney_wins = tourney_compact_results.groupby('WTeamID').count().Season.rename(columns = {'Season':'Wins'})

team_tourney_wins.name = 'Wins'

team_tourney_wins = team_tourney_wins.reset_index().rename(columns = {'WTeamID':'TeamID'})



# count amount of loses for each team

team_tourney_loses = tourney_compact_results.groupby('LTeamID').count().Season.rename(columns = {'Season':'Loses'})

team_tourney_loses.name = 'Loses'

team_tourney_loses = team_tourney_loses.reset_index().rename(columns = {'LTeamID':'TeamID'})



# merge wins and loses data

team_tourney_games = pd.merge(team_tourney_loses, team_tourney_wins, how = 'left')

team_tourney_games = team_tourney_games.fillna(0)



# merge wins and loses data with each teams location

map1_data = pd.merge(team_tourney_games, location_data).drop_duplicates(subset='TeamID')



fig = px.scatter_geo(map1_data, title = 'Map of teams by wins since 1985<br>(zoom in and out)',

                     lon = map1_data.lng,

                     lat = map1_data.lat,

                     color = 'Wins',

                     size = 'Wins',

                     hover_name='TeamName',

                     projection="albers usa")



fig.update_traces(hovertemplate="<b>%{text}</b><br><br>Total number of wins: %{customdata}",text = map1_data.TeamName, customdata = map1_data.Wins , hoverinfo='skip')



fig.show()
# count number of championships

champ_count = tourney_compact_results[tourney_compact_results.DayNum == 154].groupby('WTeamID').count().Season.rename(columns = {'Season':'Wins'})

champ_count.name = 'Championships'

champ_count.dropna()

champ_count = champ_count.reset_index().rename(columns = {'WTeamID':'TeamID'})



# merge data about championship teams and location 

map2_data = pd.merge(champ_count, location_data).drop_duplicates(subset='TeamID')



fig = px.scatter_geo(map2_data, title = 'Map of teams by Championships since 1985<br>(zoom in and out)',

                     lon = map2_data.lng,

                     lat = map2_data.lat,

                     size = 'Championships',

                     color = 'Championships',

                     hover_name='TeamName',

                     projection="albers usa",

                     color_continuous_scale = 'Rainbow')



fig.update_traces(hovertemplate="<b>%{text}</b><br><br>Championships won: %{customdata}",text = map2_data.TeamName, customdata = map2_data.Championships , hoverinfo='skip')



fig.show()
# count games in every city

count_games_in_every_city = neutral_loc.groupby('City').State.count().reset_index()

count_games_in_every_city.rename(columns = {'State':'Games played'}, inplace = True)



# merge data about game cities and their location

map3_data = pd.merge(neutral_loc.drop_duplicates(),count_games_in_every_city, on = 'City')



fig = px.scatter_geo(map3_data, title = 'Map of cities where March Madness games were played since 2010 <br>(zoom in and out)',

                     lon = map3_data.lng,

                     lat = map3_data.lat,

                     size = 'Games played',

                     color = 'Games played',

                     hover_name='City',

                     projection="albers usa")



fig.update_traces(hovertemplate="<b>%{text}</b><br><br>Games played since 2010: %{customdata}",text = map3_data.City, customdata = map3_data['Games played'] , hoverinfo='skip')



fig.show()
# create dataframe, where each game is watched from both teams sides

all_tourney_results.drop(['WLoc'], axis = 1, inplace = True)

all_tourney_results_swap = all_tourney_results[[

    'Season', 'DayNum', 'LTeamID', 'LScore', 'WTeamID', 'WScore', 'NumOT', 

    'LFGM', 'LFGA', 'LFGM3', 'LFGA3', 'LFTM', 'LFTA', 'LOR', 'LDR', 'LAst', 'LTO', 'LStl', 'LBlk', 'LPF', 

    'WFGM', 'WFGA', 'WFGM3', 'WFGA3', 'WFTM', 'WFTA', 'WOR', 'WDR', 'WAst', 'WTO', 'WStl', 'WBlk', 'WPF']]

all_tourney_results.columns = [x.replace('W','Team ').replace('L','Opponent ') for x in list(all_tourney_results.columns)]

all_tourney_results_swap.columns = [x.replace('L','Team ').replace('W','Opponent ') for x in list(all_tourney_results.columns)]



all_tourney_results = pd.concat((all_tourney_results,all_tourney_results_swap))



# count number of games of each team

tourney_games_count = all_tourney_results.groupby(['Team TeamID']).count().reset_index()[['Team TeamID', 'Season']]



# get total stats

total_data = all_tourney_results.groupby(['Team TeamID']).sum().reset_index()

total_data.drop(['Season', 'DayNum'], axis = 1, inplace = True)

total_data = pd.merge(total_data, tourney_games_count)

total_data.rename(columns = {'Season':'Games'}, inplace = True)

total_data = pd.merge(total_data, location_data, left_on = 'Team TeamID', right_on = 'TeamID')

total_data.drop_duplicates(subset = 'Team TeamID', inplace = True)



#count total games

season_tourney_games_count = all_tourney_results.groupby(['Season','Team TeamID']).count().reset_index()[['Season','Team TeamID','DayNum']]

season_tourney_games_count.rename(columns = {'DayNum':'Games'}, inplace = True)





map5_data = total_data[['Games', 'TeamID', 'TeamName', 'CityID', 'City', 'State', 'lat', 'lng']]



map5_data = pd.concat((map5_data,total_data[['Team TeamID', 'Team Score', 'Opponent TeamID', 'Opponent Score', 'NumOT', 'Team FGM',

       'Team FGA', 'Team FGM3', 'Team FGA3', 'Team FTM', 'Team FTA', 'Team OR', 'Team DR',

       'Team Ast', 'Team TO', 'Team Stl', 'Team Blk', 'Team PF', 'Opponent FGM', 'Opponent FGA',

       'Opponent FGM3', 'Opponent FGA3', 'Opponent FTM', 'Opponent FTA', 'Opponent OR', 'Opponent DR', 'Opponent Ast',

       'Opponent TO', 'Opponent Stl', 'Opponent Blk', 'Opponent PF']].div(total_data.Games, axis = 0)), axis = 1)



map5_data = map5_data[map5_data.Games > 5]

map5_data.iloc[:,9:] = map5_data.iloc[:,9:].round(1)



stat_names = ['Team Score', 'Team FGM','Team FGA','Team FGM3','Team FGA3','Team FTM','Team FTA','Team OR','Team DR', 'Team Ast','Team TO','Team Stl','Opponent Score', 'Opponent FGM','Opponent FGA','Opponent FGM3','Opponent FGA3','Opponent FTM','Opponent FTA','Opponent OR','Opponent DR', 'Opponent Ast','Opponent TO','Opponent Stl']

fig = go.Figure()

but_list = list()



for x in stat_names:

    map5_data = map5_data.assign(text = map5_data['TeamName'] + ', Average ' + x + ': ' + map5_data[x].astype(str))



    fig.add_trace(go.Scattergeo(

        lon = map5_data.lng,

        lat = map5_data.lat,

        text = map5_data.text,

        hovertemplate = '%{text}',

        mode = 'markers',

        marker=dict(

        size= np.interp(map5_data[x], (map5_data[x].min(),map5_data[x].max()), (3,25)),

        cmax=0,

        cmin=0,

        color=map5_data[x],

        colorbar=dict(

            title= x + " average"

        ),

        colorscale="Bluered"

        ),

        marker_color = map5_data[x],

        name = x + ' average',

        visible = (x == 'Team Score')

        ))

    

    ###

    

    list_of_visible = [False]*len(stat_names)

    list_of_visible[stat_names.index(x)] = True

    

    d = dict(label=x, method = 'update', args=[{"visible": list_of_visible},{"title": x + " average<br>(scroll to zoom; choose stats in dropdown menu)"}])

    but_list.append(d)



fig.update_layout(

    updatemenus=[

        dict(

            active=0,

            buttons=but_list,

        )

    ])



fig.update_layout(

        title = 'Team Scoring average in March Madness<br>(scroll to zoom; choose stats in dropdown menu)',

        title_text="Team Scoring average in March Madness<br>(scroll to zoom; choose stats in dropdown menu)",

        geo_scope='usa',

    )



fig.show()
# copy regular season statistics

all_regular_results = regular_results.copy()



# create dataframe, where each game is watched from both teams sides

all_regular_results.drop(['WLoc'], axis = 1, inplace = True)

all_regular_results_swap = all_regular_results[['Season', 'DayNum', 'LTeamID', 'LScore', 'WTeamID', 'WScore', 'NumOT', 'LFGM', 'LFGA', 'LFGM3', 'LFGA3', 'LFTM', 'LFTA', 'LOR', 'LDR', 'LAst', 'LTO', 'LStl', 'LBlk', 'LPF', 'WFGM', 'WFGA', 'WFGM3', 'WFGA3', 'WFTM', 'WFTA', 'WOR', 'WDR', 'WAst', 'WTO', 'WStl', 'WBlk', 'WPF']]

all_regular_results.columns = [x.replace('W','Team ').replace('L','Opponent ') for x in list(all_regular_results.columns)]

all_regular_results_swap.columns = [x.replace('L','Team ').replace('W','Opponent ') for x in list(all_regular_results.columns)]



all_regular_results = pd.concat((all_regular_results,all_regular_results_swap))



# count regular season games by year

season_regular_games_count = all_regular_results.groupby(['Season','Team TeamID']).count().reset_index()[['Season','Team TeamID','DayNum']]

season_regular_games_count.rename(columns = {'DayNum':'Games'}, inplace = True)



# get team statistics by season

season_regular_total_data = all_regular_results.groupby(['Season','Team TeamID']).sum().reset_index()

season_regular_total_data.drop(['DayNum'], axis = 1, inplace = True)

season_regular_total_data = pd.merge(season_regular_total_data, season_regular_games_count, left_on = ['Season','Team TeamID'], right_on=['Season','Team TeamID'])

season_regular_total_data = pd.merge(season_regular_total_data, location_data, left_on = 'Team TeamID', right_on = 'TeamID')

season_regular_total_data.drop_duplicates(subset = ['Season','Team TeamID'], inplace = True)



map4_data = season_regular_total_data[['Season','Games', 'TeamID', 'TeamName', 'CityID', 'City', 'State', 'lat', 'lng']]



# get average team statistics by season

map4_data = pd.concat((map4_data,season_regular_total_data[['Team TeamID', 'Team Score', 'Opponent TeamID', 'Opponent Score', 'NumOT', 'Team FGM',

       'Team FGA', 'Team FGM3', 'Team FGA3', 'Team FTM', 'Team FTA', 'Team OR', 'Team DR',

       'Team Ast', 'Team TO', 'Team Stl', 'Team Blk', 'Team PF', 'Opponent FGM', 'Opponent FGA',

       'Opponent FGM3', 'Opponent FGA3', 'Opponent FTM', 'Opponent FTA', 'Opponent OR', 'Opponent DR', 'Opponent Ast',

       'Opponent TO', 'Opponent Stl', 'Opponent Blk', 'Opponent PF']].div(map4_data.Games, axis = 0)), axis = 1)



map4_data = map4_data.round(1)



map4_data = pd.merge(map4_data, season_tourney_games_count.rename(columns={'Games':'Tourney_played'}), left_on = ['Season','TeamID'], right_on = ['Season','Team TeamID'])

map4_data = map4_data.sort_values(['Season','TeamID'])



map4_data.drop(['Team TeamID_x', 'Team TeamID_y'], axis = 1, inplace = True)



fig = px.scatter_geo(map4_data, title = 'Map of teams by points scored in regular season<br>(scroll to zoom; choose season by slider)',

                     lon = map4_data.lng,

                     lat = map4_data.lat,

                     size = np.interp(map4_data['Team Score'], (map4_data['Team Score'].min(),map4_data['Team Score'].max()), (1,14)),

                     color = 'Team Score',

                     hover_name='TeamName',

                     text = 'Games',

                     projection="albers usa",

                     animation_frame = 'Season',

                     animation_group = 'TeamName',

                     color_continuous_scale = "Bluered")



fig.show()
#plot

plt.figure(figsize = (17,5));

plt.title('Wins distribution', fontsize = 14);

sns.distplot(tourney_results['Relative_proximity'], axlabel = "Winning team Relative Proximity", bins = 24, norm_hist = False, kde = None);



# PRINTS

print1 = 'We have Relative Proximity average of ' + str(int(tourney_results['Relative_proximity'].mean())) + ' miles across all of the wins.'
maximum = 0 # variable for maximum value

p = 0 # variable for critical value

for i in range(0,1700):

    # count wins in closer/neutral/farther groups for each i

    i_count_farther = tourney_results[tourney_results['Relative_proximity'] < -i].shape[0]

    i_count_neutral = tourney_results[(tourney_results['Relative_proximity'] >= -i) & (tourney_results['Relative_proximity'] < i)].shape[0]

    i_count_closer = tourney_results[tourney_results['Relative_proximity'] >= i].shape[0]

    # check first condition

    if (i_count_neutral > i_count_closer) or (i_count_neutral > i_count_farther):

        break

    # check second condition

    if (i_count_closer/(i_count_closer + i_count_farther) > maximum):

        p = i

        maximum = i_count_closer/(i_count_closer + i_count_farther)

        

print("The critical value is: " + str(p))
# create 'no neutral(nn)' dataframe of tournament games named tourney_results_nn

tourney_results_nn = tourney_results[(tourney_results['Relative_proximity'] < -p) | (tourney_results['Relative_proximity'] > p)];

# 'swing' distance data, so it starts from zero

tourney_results_nn = tourney_results_nn.assign(Relative_proximity = tourney_results_nn['Relative_proximity']  - np.sign(tourney_results_nn['Relative_proximity'])*p)



# count wins in each category

count_farther = tourney_results[tourney_results['Relative_proximity'] < -p].shape[0];

count_closer = tourney_results[tourney_results['Relative_proximity'] >= p].shape[0];



#plot

plt.figure(figsize=(12,9));

plt.xticks(np.arange(2), ('teams that are based ' + str(p) + ' miles closer than opponent', 'teams that are based ' + str(p) + ' miles farther than opponent'));

fig = plt.bar(np.arange(2), (count_closer,count_farther), color=sns.color_palette("Blues",2));

plt.title('Total games won in significant matches since 2010', fontsize = 14);

plt.show();



#PRINT

print1 = 'Teams that are based ' + str(p) + ' miles closer than opponent won ' + str(count_closer) + ' games and have ' + str(int(round(count_closer/(count_closer+count_farther),2)*100)) + '% winning percentage.'

print2 = str(count_farther) + ' games were won by the team ' + str(p) + ' miles farther.'
plt.figure(figsize = (17,5));

plt.title('Wins distribution of significant games', fontsize = 14);

sns.distplot(tourney_results_nn['Relative_proximity'], axlabel = " Significant Relative Proximity", bins = 24);
# get original distances back (+ or - 160 miles to every value)

tourney_results_nn['Relative_proximity'] = tourney_results_nn['Relative_proximity'] + np.sign(tourney_results_nn['Relative_proximity'])*p;



# count wins and loses for every team in every significant games

team_tourney_wins = tourney_results_nn.groupby('WTeamID').count().Season.rename(columns = {'Season':'Wins'});

team_tourney_wins.name = 'Wins';

team_tourney_wins = team_tourney_wins.reset_index().rename(columns = {'WTeamID':'TeamID'});



team_tourney_loses = tourney_results_nn.groupby('LTeamID').count().Season.rename(columns = {'Season':'Loses'});

team_tourney_loses.name = 'Loses'

team_tourney_loses = team_tourney_loses.reset_index().rename(columns = {'LTeamID':'TeamID'});



team_tourney_games = pd.merge(team_tourney_loses, team_tourney_wins, how = 'left');

team_tourney_games = team_tourney_games.fillna(0);



# count total and average distance traveled for every team in every significant game

team_distance_rank = pd.merge(tourney_results_nn.groupby('WTeamID').sum().reset_index()[['WTeamID','Relative_proximity','WDistance']],tourney_results_nn.groupby('LTeamID').sum().reset_index()[['LTeamID','Relative_proximity','LDistance']], left_on = 'WTeamID', right_on = 'LTeamID');

team_distance_rank['Relative_proximity'] = team_distance_rank['Relative_proximity_x'] - team_distance_rank['Relative_proximity_y'];

team_distance_rank.drop(['Relative_proximity_x', 'Relative_proximity_y','LTeamID'], axis = 1, inplace = True);

#team_distance_rank['Distance_total'] = team_distance_rank['WDistance'] + team_distance_rank['LDistance'];

team_distance_rank.drop(['LDistance', 'WDistance'], axis = 1, inplace = True);

team_distance_rank.rename(columns = {'WTeamID':'TeamID'}, inplace = True);

team_distance_rank = pd.merge(team_distance_rank, team_names);

team_distance_rank.drop(['FirstD1Season', 'LastD1Season'], axis = 1, inplace = True);



team_dist_and_games = pd.merge(team_distance_rank, team_tourney_games);

all_team_dist_and_games = team_dist_and_games;

#team_dist_and_games['Distance_average'] = team_dist_and_games['Distance_total']/(team_dist_and_games['Loses'] + team_dist_and_games['Wins']);



#more than 3 significant games played

team_dist_and_games = team_dist_and_games[team_dist_and_games['Loses']+team_dist_and_games['Wins'] > 5];



# count average Relative Proximity and winning percentage for every team

team_dist_and_games = team_dist_and_games.assign(Relative_proximity_average = team_dist_and_games['Relative_proximity']/(team_dist_and_games['Loses'] + team_dist_and_games['Wins']))

team_dist_and_games['Winning_percentage'] = team_dist_and_games['Wins']/(team_dist_and_games['Loses'] +team_dist_and_games['Wins'])



# Get

# Top 20 Winning teams in significant matches

top_20_winning_teams = team_dist_and_games.sort_values('Winning_percentage', ascending = False).iloc[0:20]

# Bottom 20 Winning teams in significant matches

bottom_20_winning_teams = team_dist_and_games.sort_values('Winning_percentage', ascending = True).iloc[0:20]

# Top 20 team by Relative Proximity in significant matches

top_20_Geographical_av = team_dist_and_games.sort_values('Relative_proximity_average', ascending = False).iloc[0:20];

# Bottom 20 team by Relative Proximity in significant matches

bottom_20_Geographical_av = team_dist_and_games.sort_values('Relative_proximity_average').iloc[0:20];



plt.figure(figsize=(17,7));

plot = sns.barplot(x='TeamName', y = 'Relative_proximity_average', data = top_20_winning_teams, palette = 'Blues_d');

plot.set_xticklabels(top_20_winning_teams['TeamName'], rotation = 90);

plot.set(xlabel = "Better team are on the left", ylabel = "Relative Proximity average", title = "Top 20 winning teams' Relative Proximity (average)");

         

# PRINTS

print1 = 'In significant games, winning teams tend to be ' + str(int(top_20_winning_teams['Relative_proximity_average'].mean())) + ' miles closer to the game point'

print2 = top_20_winning_teams[top_20_winning_teams.TeamName == 'Connecticut'].Relative_proximity_average.iloc[0]
plt.figure(figsize=(17,7));

plot = sns.barplot(bottom_20_winning_teams['TeamName'][::-1], bottom_20_winning_teams['Relative_proximity_average'], palette = 'Blues_d');

plot.set_xticklabels(bottom_20_winning_teams['TeamName'][::-1], rotation = 90);

plot.set(xlabel = "Better teams are on the left", ylabel = "Relative Proximity average", title = "Bottom 20 winning teams' Relative Proximity (average)");



# PRINTS

print1 = 'In significant games, losing teams tend to be ' + str(int(bottom_20_winning_teams['Relative_proximity_average'].mean())) + ' miles farther to the game point'
fig, (ax1,ax2) = plt.subplots(1,2,sharey='row', figsize=(17,7))



plot1 = sns.barplot(-top_20_Geographical_av['Relative_proximity_average'],top_20_Geographical_av['Wins'], palette = 'Blues_d', ax = ax1)

plot1.set_xticklabels(top_20_Geographical_av['TeamName'], rotation = 90)

plot1.set(xlabel = "", ylabel = "Wins", title = 'Top 20 teams by Relative Proximity');



plot2 = sns.barplot(-bottom_20_Geographical_av['Relative_proximity_average'], bottom_20_Geographical_av['Wins'], palette = 'Blues_d', ax = ax2)

plot2.set_xticklabels(bottom_20_Geographical_av['TeamName'][::-1], rotation = 90)

plot2.set(xlabel = "", ylabel = "Wins", title = 'Bottom 20 teams by Relative Proximity');



### PRINTS

print1 = 'Average team from top 20 list of teams by Relative Proximity has ' + str(top_20_Geographical_av['Wins'].mean()) + ' wins in 2010-2019 championships'

print2 = 'Average team from bottom 20 list of teams by Relative Proximity has ' + str(bottom_20_Geographical_av['Wins'].mean()) + ' wins in 2010-2019 championships'
fig, (ax1,ax2) = plt.subplots(1,2,sharey='row', figsize=(17,7))



plot1 = sns.barplot(-top_20_Geographical_av['Relative_proximity_average'],top_20_Geographical_av['Winning_percentage'], palette = 'Blues_d', ax = ax1)

plot1.set_xticklabels(top_20_Geographical_av['TeamName'], rotation = 90)

plot1.set(xlabel = "", ylabel = "Winning percentage", title = 'Top 20 teams by Relative Proximity');



plot2 = sns.barplot(-bottom_20_Geographical_av['Relative_proximity_average'], bottom_20_Geographical_av['Winning_percentage'], palette = 'Blues_d', ax = ax2)

plot2.set_xticklabels(bottom_20_Geographical_av['TeamName'][::-1], rotation = 90)

plot2.set(xlabel = "", ylabel = "Winning percentage", title = 'Bottom 20 teams by Relative Proximity');



# PRINTS

print1 = 'Average team from top 20 list of teams by Relative Proximity has ' + str(top_20_Geographical_av['Winning_percentage'].mean()- bottom_20_Geographical_av['Winning_percentage'].mean()) + ' wins in 2010-2019 championships'

print2 = 'Average team from bottom 20 list of teams by Relative Proximity has ' + str() + ' wins in 2010-2019 championships'
# set of winning teams

wt = set(top_20_winning_teams.TeamID)

# set of losing teams

lt = set(bottom_20_winning_teams.TeamID)

# set of top teams by Relative Proximity

advt = set(top_20_Geographical_av.TeamID)

# set of bottom teams by Relative Proximity

disadvt = set(bottom_20_Geographical_av.TeamID)

pd.concat((top_20_winning_teams[top_20_winning_teams.TeamID.isin(wt.intersection(disadvt))], bottom_20_winning_teams[bottom_20_winning_teams.TeamID.isin(lt.intersection(advt))]), axis = 0).reset_index(drop = True)
# set color and text for game cities

neutral_loc.drop_duplicates(inplace = True)

neutral_loc = neutral_loc.dropna()

neutral_loc['Teams by Relative Proximity average'] = 'Game City'

neutral_loc['text'] = neutral_loc['City'] + ', ' + neutral_loc['State']



# set color and text for top teams by Relative Proximity

advantagers_loc = pd.merge(top_20_Geographical_av,location_data)

advantagers_loc = advantagers_loc.drop_duplicates(subset = 'TeamName')

advantagers_loc['Teams by Relative Proximity average'] = 'Top teams'

advantagers_loc['text'] = advantagers_loc['TeamName'] + ', City: ' + advantagers_loc['City'] + ', ' + advantagers_loc['State']



# set color and text for bottom teams by Relative Proximity

disadvantagers_loc = pd.merge(bottom_20_Geographical_av,location_data)

disadvantagers_loc = disadvantagers_loc.drop_duplicates(subset = 'TeamName')

disadvantagers_loc['Teams by Relative Proximity average'] = 'Bottom teams'

disadvantagers_loc['text'] = disadvantagers_loc['TeamName'] + ', City: ' + disadvantagers_loc['City'] + ', ' + disadvantagers_loc['State']



fig0 = go.Figure()



fig0.add_trace(go.Scattergeo(

        lon = disadvantagers_loc.lng,

        lat = disadvantagers_loc.lat,

        hovertemplate = "<b>%{text}</b><br><br>Relative Proximity average: %{customdata:.2f}",

        text = disadvantagers_loc.text,

        customdata = disadvantagers_loc.Relative_proximity_average,

        mode = 'markers',

        hoverinfo='skip',

        marker=dict(

        #size= np.interp(disadvantagers_loc['Relative_proximity_average'], (disadvantagers_loc['Relative_proximity_average'].max(),disadvantagers_loc['Relative_proximity_average'].min()), (5,8)),

        color = "#ff0000",

        ),

        name = 'Bottom teams',

        visible = True

        ))



fig0.add_trace(go.Scattergeo(

        lon = advantagers_loc.lng,

        lat = advantagers_loc.lat,

        hovertemplate = "<b>%{text}</b><br><br>Relative Proximity average: %{customdata:.2f}",

        text = advantagers_loc.text,

        customdata = advantagers_loc.Relative_proximity_average,

        mode = 'markers',

        hoverinfo='skip',

        marker=dict(

        #size= np.interp(advantagers_loc['Relative_proximity_average'], (advantagers_loc['Relative_proximity_average'].max(),advantagers_loc['Relative_proximity_average'].min()), (5,8)),

        color = "#0000ff",

        ),

        name = 'Top teams',

        visible = True

        ))

    



fig0.update_layout(

        title = 'Teams by Relative Proximity<br>(scroll to zoom)',

        geo_scope='usa',

    )



fig0.show()
# set color and text for top teams by Relative Proximity

winners_loc = pd.merge(top_20_winning_teams,location_data)

winners_loc = winners_loc.drop_duplicates(subset = 'TeamName')

winners_loc['text'] = winners_loc['TeamName'] + ', City: ' + winners_loc['City'] + ', ' + winners_loc['State']



# set color and text for bottom teams by Relative Proximity

losers_loc = pd.merge(bottom_20_winning_teams,location_data)

losers_loc = losers_loc.drop_duplicates(subset = 'TeamName')

losers_loc['text'] = losers_loc['TeamName'] + ', City: ' + losers_loc['City'] + ', ' + losers_loc['State']

fig1 = go.Figure()



fig1.add_trace(go.Scattergeo(

        lon = losers_loc.lng,

        lat = losers_loc.lat,

        hovertemplate = "<b>%{text}</b><br><br>Winning percentage: %{customdata:%.2f}",

        text = losers_loc.text,

        customdata = losers_loc.Winning_percentage,

        mode = 'markers',

        hoverinfo='skip',

        marker=dict(

        #size= np.interp(disadvantagers_loc['Relative_proximity_average'], (disadvantagers_loc['Relative_proximity_average'].max(),disadvantagers_loc['Relative_proximity_average'].min()), (5,8)),

        color = "#ff0000",

        ),

        name = 'Bottom teams',

        visible = True

        ))



fig1.add_trace(go.Scattergeo(

        lon = winners_loc.lng,

        lat = winners_loc.lat,

        hovertemplate = "<b>%{text}</b><br><br>Winning_percentage in significant games: %{customdata:%.2f}",

        text = winners_loc.text,

        customdata = winners_loc.Winning_percentage,

        mode = 'markers',

        hoverinfo='skip',

        marker=dict(

        #size= np.interp(advantagers_loc['Relative_proximity_average'], (advantagers_loc['Relative_proximity_average'].max(),advantagers_loc['Relative_proximity_average'].min()), (5,8)),

        color = "#0000ff",

        ),

        name = 'Top teams',

        visible = True

        ))

    



fig1.update_layout(

        title = 'Teams by winning percentage<br>(scroll to zoom)',

        geo_scope='usa'

    )



fig1.show()


dists_and_seeds = tourney_results_nn[tourney_results_nn.DayNum.isin(range(136,155))]



dists_and_seeds = pd.merge(dists_and_seeds , seeds,left_on = ['Season','WTeamID'], right_on=['Season','TeamID'])

dists_and_seeds['WSeed'] = pd.to_numeric(dists_and_seeds['Seed'].str[1:3])

dists_and_seeds.drop(['TeamID', 'Seed'],axis = 1, inplace = True)





dists_and_seeds = pd.merge(dists_and_seeds, seeds,left_on = ['Season','LTeamID'], right_on=['Season','TeamID'])

dists_and_seeds['LSeed'] = pd.to_numeric(dists_and_seeds['Seed'].str[1:3])

dists_and_seeds.drop(['TeamID','Seed'],axis = 1, inplace = True)



seed_ga_data = list()

for i in range(1,17):

    seed_i_data = list()

    

    seed_i_wins = dists_and_seeds[(dists_and_seeds.WSeed == i)]

    seed_i_loses = dists_and_seeds[(dists_and_seeds.LSeed == i)]

    seed_i_loses = seed_i_loses.assign(Relative_proximity = -1*seed_i_loses.Relative_proximity)

    seed_i = pd.concat((seed_i_wins, seed_i_loses))

    

    seed_i_data.append(i)

    seed_i_data.append(seed_i_wins.Relative_proximity.mean())

    seed_i_data.append(seed_i_loses.Relative_proximity.mean())

    seed_i_data.append(seed_i.Relative_proximity.mean())

    seed_i_data.append(seed_i_wins.shape[0])

    seed_i_data.append(seed_i_loses.shape[0])

    

    seed_ga_data.append(seed_i_data)



seeds_ga_df = pd.DataFrame(seed_ga_data, columns = ['Seed','Relative Proximity in wins','Relative Proximity in loses','Relative Proximity', 'Total number of wins', 'Total number of loses'])



from IPython.display import display, HTML

display(HTML(seeds_ga_df.to_html(index=False)))
# count wins in each category

no_high_low_seeds = dists_and_seeds[((~dists_and_seeds['WSeed'].isin([1,2,3,4]))) & ((~dists_and_seeds['LSeed'].isin([1,2,3,4])))]



# count wins in each category

count_farther = no_high_low_seeds[no_high_low_seeds['Relative_proximity'] < 0].shape[0];

count_closer = no_high_low_seeds[no_high_low_seeds['Relative_proximity'] >= 0].shape[0];



#plot

plt.figure(figsize=(12,9));

plt.xticks(np.arange(2), ('teams that are based ' + str(p) + ' miles closer than opponent', 'teams that are based ' + str(p) + ' miles farther than opponent'));

fig = plt.bar(np.arange(2), (count_closer,count_farther), color=sns.color_palette("Blues",2));

plt.title('Total games won in significant matches since 2010 (no 1st, 2nd, 3rd, 4th seeds)', fontsize = 14);

plt.show();
detailed_tourney_results.drop(['WLoc'], axis = 1, inplace = True)



plt.figure(figsize = (3,10))

sns.heatmap(detailed_tourney_results.corr()[['Relative_proximity']], center = 0);
# count closer wins

closer_wins = tourney_results_nn[tourney_results_nn['Relative_proximity'] > 0].groupby('WTeamID').count().reset_index()[['WTeamID','Season']]

closer_wins.rename(columns = {'WTeamID':'TeamID', 'Season':'Wins when relatively closer'}, inplace = True)



# count closer loses

closer_loses = tourney_results_nn[tourney_results_nn['Relative_proximity'] > 0].groupby('LTeamID').count().reset_index()[['LTeamID','Season']]

closer_loses.rename(columns = {'LTeamID':'TeamID', 'Season':'Loses when relatively closer'}, inplace = True)



# count farther wins

farther_wins = tourney_results_nn[tourney_results_nn['Relative_proximity'] < 0].groupby('WTeamID').count().reset_index()[['WTeamID','Season']]

farther_wins.rename(columns = {'WTeamID':'TeamID', 'Season':'Wins when relatively farther'}, inplace = True)



# count closer loses

farther_loses = tourney_results_nn[tourney_results_nn['Relative_proximity']< 0].groupby('LTeamID').count().reset_index()[['LTeamID','Season']]

farther_loses.rename(columns = {'LTeamID':'TeamID', 'Season':'Loses when relatively farther'}, inplace = True)



# merge loses and wins for every team

closer_games = pd.merge(closer_wins, closer_loses, how = 'left', on = 'TeamID')

closer_games = closer_games.fillna(0)

farther_games = pd.merge(farther_wins, farther_loses, how = 'left', on = 'TeamID')

farther_games = farther_games.fillna(0)

games = pd.merge(closer_games, farther_games)

games = games.fillna(0)



# get statistics

games = pd.merge(games, team_names)

games['Relatively closer games'] = games['Wins when relatively closer'] + games['Loses when relatively closer']

games['Relatively farther games'] = games['Wins when relatively farther'] + games['Loses when relatively farther']

games.drop(['FirstD1Season','LastD1Season'], axis = 1,inplace = True)

games = games[games['Relatively closer games'] + games['Relatively farther games'] > 5]

games['Winning % when relatively closer'] = games['Wins when relatively closer']/games['Relatively closer games']

games['Winning % when relatively farther'] = games['Wins when relatively farther']/games['Relatively farther games']

games = games[['TeamName', 'Winning % when relatively closer','Winning % when relatively farther', 'Wins when relatively closer', 'Loses when relatively closer','Wins when relatively farther', 'Loses when relatively farther']]



games['Loses when relatively closer'] = games['Loses when relatively closer'].astype('Int32')

games['Loses when relatively farther'] = games['Loses when relatively farther'].astype('Int32')



games
# merge teams' travel distances and data about wins and loses

map7_data = pd.merge(teams_travel_distances, team_tourney_games)



# count average distance

map7_data['Average_distance'] = map7_data['Distance']/(map7_data['Loses']+map7_data['Wins'])



# merge with location data

map7_data = pd.merge(map7_data, location_data)



# merge with team names

map7_data = pd.merge(map7_data, team_names)



# round distances for better display

map7_data = map7_data.assign(Average_distance = map7_data.Average_distance.round(1))



# minimum 5 games

map7_data = map7_data[map7_data['Wins'] + map7_data['Loses'] > 4]



# drop duplicated teams

map7_data.drop_duplicates(subset = 'TeamName', inplace = True)



# count winning percetage

map7_data['Winning%'] = (map7_data['Wins']/(map7_data['Wins']+map7_data['Loses'])).round(2)



# assign text for display

map7_data['text'] = map7_data['TeamName'] + '<br><br>Average distance: ' + map7_data['Average_distance'].astype(str) + '<br> Winning %: ' + map7_data['Winning%'].astype(str)



fig2 = px.scatter_geo(map7_data, title = 'Map of teams by distance average<br>(winning percentage in games after travel = size)',

                     lon = map7_data.lng,

                     lat = map7_data.lat,

                     color= 'Average_distance',

                     size = 'Winning%',

                     hover_name="TeamName",

                     projection="albers usa",

                     color_continuous_scale = 'YlOrRd')



fig2.update_traces(hovertemplate='%{text}',text = map7_data.text, hoverinfo='skip')



fig2.update_layout(

        coloraxis_colorbar=dict(title = 'Average distnace traveled')

    )

        

fig2.show()
plt.figure(figsize = (17,5))

plt.title('Relative frequency distribution of games', fontsize = 14)

sns.distplot(games_and_coordinates['WDistance'], bins = 24, hist = False,label="Distance traveled by winning team");



sns.distplot(games_and_coordinates['LDistance'], axlabel = "Distance traveled", label="Distance traveled by losing team",bins = 24, hist = False);

plt.legend()



# PRINTS

print1 = 'Average distance in loses: ' + str(int(games_and_coordinates['LDistance'].mean())) + ' miles.'

print2 = 'Average distance in wins: ' + str(int(games_and_coordinates['WDistance'].mean())) + ' miles.'

print3 = games_and_coordinates[games_and_coordinates['WDistance'] < games_and_coordinates['LDistance']].shape[0] - games_and_coordinates[games_and_coordinates['WDistance'] > games_and_coordinates['LDistance']].shape[0]
# reset "all_tourney_results" dataframe

all_tourney_results = pd.read_csv('/kaggle/input/march-madness-analytics-2020/MDataFiles_Stage2/MNCAATourneyDetailedResults.csv')



# get data to check correlations

correlation_data = pd.merge(games_and_coordinates,all_tourney_results[all_tourney_results['Season'] > 2009].drop(['WLoc','NumOT'], axis = 1), on = ['Season','DayNum','WTeamID','LTeamID'])

correlation_data.drop(['WTeamID','GameCityID', 'LTeamID', 'Season','GameCityLat','GameCityLng','WTeamLat','WTeamLng','LTeamLat','LTeamLng'],axis = 1, inplace = True)



plt.figure(figsize=(3,10))

sns.heatmap(correlation_data.corr()[['WDistance','LDistance']], center = 0);
no_high_seed_games = games_and_coordinates[(games_and_coordinates['WSeed'] > 4) & (games_and_coordinates['LSeed'] > 4)]



plt.figure(figsize = (17,5))

plt.title('Relative frequency distribution of games (no 1st, 2nd, 3rd and 4th seed games)', fontsize = 14)

sns.distplot(no_high_seed_games['WDistance'], bins = 24, hist = False, label="Distance traveled by winning team");



sns.distplot(no_high_seed_games['LDistance'], axlabel = "Distance traveled", label="Distance traveled by losing team",bins = 24, hist = False);

plt.legend()



# PRINTS

print1 = 'Average distance in loses: ' + str(int(no_high_seed_games['LDistance'].mean())) + ' miles.'

print2 = 'Average distance in wins: ' + str(int(no_high_seed_games['WDistance'].mean())) + ' miles.'

print3 = no_high_seed_games[no_high_seed_games['WDistance'] < no_high_seed_games['LDistance']].shape[0] - no_high_seed_games[no_high_seed_games['WDistance'] > no_high_seed_games['LDistance']].shape[0]
high_seed_games = games_and_coordinates[(games_and_coordinates['WSeed']< 5) | (games_and_coordinates['LSeed'] < 5)]



plt.figure(figsize = (17,5))

plt.title('Relative frequency distribution of games(1st, 2nd, 3rd or 4th seeds are participated)', fontsize = 14)

sns.distplot(high_seed_games['WDistance'], bins = 24, hist = False,label="Distance traveled by winning team");



sns.distplot(high_seed_games['LDistance'], axlabel = "Distance traveled", label="Distance traveled by losing team",bins = 24, hist = False);

plt.legend()



# PRINTS

print1 = 'Average distance in loses: ' + str(int(high_seed_games['LDistance'].mean())) + ' miles.'

print2 = 'Average distance in wins: ' + str(int(high_seed_games['WDistance'].mean())) + ' miles.'

print3 = high_seed_games[high_seed_games['WDistance'] < high_seed_games['LDistance']].shape[0] - high_seed_games[high_seed_games['WDistance'] > high_seed_games['LDistance']].shape[0]
plt.figure(figsize = (17,5))

plt.title('Wins distribution (1st, 2nd, 3rd or 4th seeds are participated)', fontsize = 14)

sns.distplot(games_and_coordinates['Time_advantage'], bins = 24, axlabel = "Time advantage", hist = False);



# PRINTS

print1 = 'Average recovery time of winning teams: ' + str(round(games_and_coordinates['WRecoveryTime'].mean(), 3)) + ' days.'

print2 = 'Average recovery time of losing teams: ' + str(round(games_and_coordinates['LRecoveryTime'].mean(), 3)) + ' days.'

print3 = games_and_coordinates[games_and_coordinates['WRecoveryTime'] > games_and_coordinates['LRecoveryTime']].shape[0] - games_and_coordinates[games_and_coordinates['WRecoveryTime'] < games_and_coordinates['LRecoveryTime']].shape[0]
seed_time_advantages = pd.concat((games_and_coordinates.groupby('WSeed').Time_advantage.sum(),-games_and_coordinates.groupby('LSeed').Time_advantage.sum()), axis = 1)

seed_time_advantages = pd.concat((seed_time_advantages,pd.concat((games_and_coordinates.groupby('WSeed').WRecoveryTime.sum(),games_and_coordinates.groupby('LSeed').LRecoveryTime.sum()), axis = 1)), axis = 1)

seed_time_advantages.columns = ['Time advantage in wins', 'Time advantage in loses','Recovery time in wins', 'Recovery time in loses']

seed_time_advantages = pd.concat((seed_time_advantages,pd.concat((games_and_coordinates.groupby('WSeed').WRecoveryTime.count(),games_and_coordinates.groupby('LSeed').LRecoveryTime.count()), axis = 1)),axis = 1)

seed_time_advantages.rename(columns = {'WRecoveryTime':'Wins','LRecoveryTime':'Loses'}, inplace = True)

seed_time_advantages['Average time advantage in wins'] = seed_time_advantages['Time advantage in wins']/seed_time_advantages['Wins']

seed_time_advantages['Average time advantage in loses'] = seed_time_advantages['Time advantage in loses']/seed_time_advantages['Loses']

seed_time_advantages['Average time advantage'] = (seed_time_advantages['Time advantage in wins']+seed_time_advantages['Time advantage in loses'])/(seed_time_advantages['Loses']+seed_time_advantages['Wins'])

seed_time_advantages['Average recovery time in wins'] = seed_time_advantages['Recovery time in wins']/seed_time_advantages['Wins']

seed_time_advantages['Average recovery time in loses'] = seed_time_advantages['Recovery time in loses']/seed_time_advantages['Loses']

seed_time_advantages['Average recovery time'] = (seed_time_advantages['Average recovery time in wins'] + seed_time_advantages['Average recovery time in loses'])/2



from IPython.display import display, HTML

display(HTML(seed_time_advantages[['Average time advantage','Average time advantage in wins','Average time advantage in loses', 'Average recovery time in wins', 'Average recovery time in loses','Average recovery time']].reset_index().rename(columns={'index':'Seed'}).to_html(index=False)))
teams_time_advantages = pd.merge(games_and_coordinates.groupby('WTeamID').Time_advantage.sum().reset_index(),games_and_coordinates.groupby('LTeamID').Time_advantage.sum().reset_index(), left_on = 'WTeamID', right_on = 'LTeamID').dropna()

teams_time_advantages = teams_time_advantages.assign(Time_advantage_y = -teams_time_advantages['Time_advantage_y'])

teams_time_advantages.drop('LTeamID', axis = 1, inplace = True)

teams_time_advantages.columns = ['TeamID','Time advantage in wins', 'Time advantage in loses']

teams_time_advantages = pd.merge(teams_time_advantages,pd.concat((games_and_coordinates.groupby('WTeamID').WRecoveryTime.count(),games_and_coordinates.groupby('LTeamID').LRecoveryTime.count()), axis = 1).dropna().reset_index().rename(columns = {'index':'TeamID'}))

teams_time_advantages.rename(columns = {'WRecoveryTime':'Wins','LRecoveryTime':'Loses'}, inplace = True)

teams_time_advantages['Average time advantage in wins'] = teams_time_advantages['Time advantage in wins']/teams_time_advantages['Wins']

teams_time_advantages['Average time advantage in loses'] = teams_time_advantages['Time advantage in loses']/teams_time_advantages['Loses']

teams_time_advantages['Average time advantage'] = (teams_time_advantages['Time advantage in wins']+teams_time_advantages['Time advantage in loses'])/(teams_time_advantages['Loses']+teams_time_advantages['Wins'])

teams_time_advantages = teams_time_advantages[teams_time_advantages['Wins'] + teams_time_advantages['Loses'] > 5][['TeamID','Average time advantage','Average time advantage in wins','Average time advantage in loses']]



map8_data = pd.merge(teams_time_advantages, location_data)

map8_data = pd.merge(map8_data, team_names)

map8_data.drop_duplicates(subset='TeamName', inplace = True)

map8_data['text'] = map8_data['TeamName'] + '<br><br>Average time advantage: ' + map8_data['Average time advantage'].round(2).astype(str)



fig3 = px.scatter_geo(map8_data, title = 'Map of teams by average time advantage<br>(more than 5 games after travel)',

                     lon = map8_data.lng,

                     lat = map8_data.lat,

                     color= 'Average time advantage',

                     hover_name="TeamName",

                     projection="albers usa",

                     color_continuous_scale = 'bluered')



fig3.update_traces(hovertemplate='%{text}',text = map8_data.text, hoverinfo='skip')

        

fig3.show()