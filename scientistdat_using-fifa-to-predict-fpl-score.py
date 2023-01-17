import numpy as np 

import pandas as pd 

import json





fifa20 = pd.read_csv('../input/fifa-player-stats-database/FIFA20_official_data.csv')

fifa21 = pd.read_csv('../input/fifa-2021-complete-player-data/FIFA-21 Complete.csv',sep = ';')



#The FPL datasets are JSONs and we need to read them from line and use the JSON library to convert it to a Pandas DataFrame. 

fpl2020_file = open('../input/fantasy-epl-new-season-research-2020-2021/FPL_2019_20_season_stats.jscsrc')

fpl2020 = fpl2020_file.read()

fpl2020 = json.loads(fpl2020)

fpl2021_file = open('../input/fantasy-epl-new-season-research-2020-2021/FPL_2020_21_player_list.jscsrc')

fpl2021 = fpl2021_file.read()

fpl2021 = json.loads(fpl2021)

teams2020 = pd.DataFrame(fpl2020['teams'])

players2020 = pd.DataFrame(fpl2020['elements'])
fifa20.head()
fifa21.head()
fifa20_abbr = fifa20[['ID','Name','Nationality','Best Position','Overall','Age','Potential','Club']].copy()

# I have selected Best Position over Position since the Position column had values like SUB which aren't ideal. 

fifa21_abbr = fifa21.drop(['hits'],axis = 1)

fifa20_abbr.columns = fifa21_abbr.columns

fifa21_abbr['position'] = fifa21_abbr['position'].apply(lambda x: x.split('|')[0])

# Another eyecheck

fifa20_abbr.head()
#Merge FPL players data with FPL teams data 

players_wteams =  players2020.merge(teams2020, how = 'left', left_on = 'team_code',right_on = 'code',suffixes = ['','_team'])

#Drop players with 0 points 

players_wteams = players_wteams[players_wteams['total_points']!=0]

#Drop players with ppg less than 1. Since we are not interested in substitute appearances which do not produce any points

players_wteams = players_wteams[players_wteams['points_per_game'].astype('float')>1]
team_overall = fifa21_abbr.groupby(by='team',axis = 0,)['player_id'].count()

team_overall.describe()
fifa20_teams = team_overall[team_overall>22]



fpl20_teams = players_wteams.name.unique()



fpl20_teamnames = ['Arsenal', 'Aston Villa', 'Bournemouth', 'Brighton & Hove Albion', 'Burnley',

       'Chelsea', 'Crystal Palace', 'Everton', 'Leicester City', 'Liverpool',

       'Manchester City', 'Manchester United', 'Newcastle United', 'Norwich City', 'Sheffield United',

       'Southampton', 'Tottenham Hotspurs', 'Watford', 'West Ham United', 'Wolverhampton Wanderers']

fullnames_df = pd.DataFrame(data = [fpl20_teams,fpl20_teamnames])

fullnames_df = fullnames_df.transpose()

fullnames_df.columns = ['name','full_club_name']



players_wteamnames = players_wteams.merge(fullnames_df,on = ['name'])
from fuzzywuzzy import fuzz

from fuzzywuzzy import process

#Create matching fpl team column for each FIFA club. 

fifa21_abbr['matching_fpl_team'] = fifa21_abbr['team'].apply(lambda x: process.extractOne(str(x),fpl20_teamnames))



# Extract the name of the club and score for the matched club. 

fifa21_abbr['matching_fpl_team_name'] = fifa21_abbr['matching_fpl_team'].apply(lambda x: x[0])

fifa21_abbr['matching_fpl_team_score'] = fifa21_abbr['matching_fpl_team'].apply(lambda x: x[1])



#Sort the fifa clubs by the match score and retain the highest match as the correct result. 

fifa21_abbr_epl_teams = fifa21_abbr.sort_values('matching_fpl_team_score',ascending = False).groupby(['matching_fpl_team_name']).apply(lambda df: df.iloc[0])
#Select only matched teams from FIFA 

fifa21_abbr_epl = fifa21_abbr[fifa21_abbr['team'].isin(fifa21_abbr_epl_teams['team'])]



players_wteamnames['fullname'] = players_wteamnames['first_name'] + ' ' + players_wteamnames['second_name']

#Fuzzywuzzy matching similar to club name

players_wteamnames['name_matching'] = players_wteamnames.apply(lambda row : process.extractOne(str(row['web_name']),fifa21_abbr_epl[fifa21_abbr_epl['matching_fpl_team_name']==row['full_club_name']]['name'].tolist()),axis = 1 )

players_wteamnames['name_match_score'] = players_wteamnames['name_matching'].apply(lambda x: x[1])

players_wteamnames['name_match_name'] = players_wteamnames['name_matching'].apply(lambda x: x[0])

#Merge on player name

players_merged = players_wteamnames.merge(fifa21_abbr_epl,left_on = 'name_match_name',right_on = 'name',suffixes = ('','_fifa'))

#Pick the highest match in case a player matches with multiple players

players_merged = players_merged.groupby('player_id',as_index = False).apply(lambda df: df.sort_values('name_match_score',ascending = False).iloc[0])
players = players_merged[['code','dreamteam_count', 'element_type','first_name','id', 'now_cost',

                          'second_name', 'points_per_game', 'special', 'squad_number',

    'status', 'team', 'team_code', 'total_points', 'transfers_in', 'transfers_out', 

    'web_name', 'minutes', 'goals_scored',

    'assists', 'clean_sheets', 'goals_conceded', 'own_goals',

    'penalties_saved', 'penalties_missed', 'yellow_cards', 'red_cards',

    'saves', 'bonus', 'bps', 'influence', 'creativity', 'threat',

    'ict_index', 'name',  'points', 'position', 'win','played','draw', 'loss',

    'short_name', 'strength', 'full_club_name','player_id', 'name_fifa', 

    'nationality', 'position_fifa', 'overall',

    'age', 'potential', 'team_fifa','cost_change_start']].copy()

#Get the starting cost of the player

players['starting_cost'] = players['now_cost'] - players['cost_change_start']
players['points_per_game'] = players['points_per_game'].astype('float')

players.sort_values('overall')



import matplotlib.pyplot as plt

import seaborn as sns



plt.figure(figsize=(15,10))



sns.regplot(x = players['overall'], y = players['points_per_game'])



plt.title('Points per game vs FIFA Overall Rating')

plt.xlabel('FIFA Overall Rating')

plt.ylabel('Points per game')



plt.show()
plt.figure(figsize=(15,10))

plt.title("Clubs by Average Player Rating in FIFA")

club_overall = players.groupby('short_name')['overall'].mean().sort_values(ascending = False)

import seaborn as sns

plt.ylabel("Team FIFA Rating")

plt.xlabel("FPL Club")

sns.barplot(x=club_overall.index, y=club_overall).set(ylim = (65,85))

plt.show()
club_pos_overall = players.groupby(by = ['short_name','element_type'],as_index = False)['overall'].mean()

#club_pos_overall['pos'] = club_pos_overall['element_type'].replace({1:'GK',2:'DEF',3:'MID',4:'FWD'})

club_pos_overall

club_pos_overall = club_pos_overall.pivot_table(values = 'overall',index = ['element_type'],columns = 'short_name')

club_pos_overall.sort_index(ascending = False)

club_pos_overall.index = ['GK','DEF','MID','FWD']

plt.figure(figsize=(20,5))

plt.title("Average Player Rating")



sns.heatmap(data=club_pos_overall, annot=True, cmap = 'coolwarm')



# Add label for horizontal axis

plt.xlabel("FPL Club")

plt.ylabel("FPL Position 2019-20")

plt.show()
club_pos_overall = players.groupby(by = ['short_name','element_type'],as_index = False)['overall'].max()

#club_pos_overall['pos'] = club_pos_overall['element_type'].replace({1:'GK',2:'DEF',3:'MID',4:'FWD'})

club_pos_overall

club_pos_overall = club_pos_overall.pivot_table(values = 'overall',index = ['element_type'],columns = 'short_name')

club_pos_overall.sort_index(ascending = False)

club_pos_overall.index = ['GK','DEF','MID','FWD']

plt.figure(figsize=(20,5))

plt.title("Best Player Rating")



sns.heatmap(data=club_pos_overall, annot=True, cmap = 'coolwarm')



# Add label for horizontal axis

plt.xlabel("FPL Club")

plt.show()
model_data = players[['code', 'element_type','first_name','id','starting_cost','second_name', 'points_per_game',

                      'status', 'team', 'team_code', 'web_name','bps', 'name', 'short_name', 'full_club_name',

                      'player_id', 'name_fifa', 'nationality', 'position_fifa', 'overall','age', 'potential', 

                      'team_fifa']].copy()



from sklearn.model_selection import train_test_split

y = model_data['points_per_game'].astype('float')

X = model_data.drop(['points_per_game','bps'],axis = 1)

X_train, X_test, y_train, y_test = train_test_split(X,y,train_size = 0.8, random_state = 0)
model_1_columns = ['starting_cost','overall','age','potential']

X_train_model_1 = X_train[model_1_columns]

from xgboost import XGBRegressor

model_1 = XGBRegressor(random_state = 0,learning_rate = 0.05)

model_1.fit(X_train_model_1,y_train)

preds = model_1.predict(X_test[model_1_columns])

from sklearn.metrics import mean_absolute_error , mean_squared_error

error = mean_absolute_error(y_test,preds)

#sum([ abs(x - y) for x,y in zip(y_test.values,preds)])
model_2_columns = ['starting_cost','overall','age','potential','element_type']

X_train_model_2 = X_train[model_2_columns].reset_index()

X_test_model_2 = X_test[model_2_columns].reset_index()



from sklearn.preprocessing import OneHotEncoder

OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)

OH_position = OH_encoder.fit_transform(X_train_model_2['element_type'].values.reshape(-1,1))

position_df_xtrain = pd.DataFrame(OH_position,columns = ['GK','DEF','MID','FWD'])

X_train_model_2_encoded = pd.concat([X_train_model_2,position_df_xtrain],axis = 1,ignore_index=False)

X_train_model_2_encoded.drop(['index','element_type'],axis = 1)



model_2 = XGBRegressor(random_state = 0,learning_rate = 0.05)

model_2.fit(X_train_model_2_encoded,y_train)



position_df_xtest = OH_encoder.transform(X_test_model_2['element_type'].values.reshape(-1,1))

position_df_xtest = pd.DataFrame(position_df_xtest,columns = ['GK','DEF','MID','FWD'])

X_test_model_2_encoded = pd.concat([X_test_model_2,position_df_xtest],axis = 1,ignore_index=False)

X_test_model_2_encoded.drop(['index','element_type'],axis = 1)

preds = model_2.predict(X_test_model_2_encoded)

error = mean_absolute_error(y_test,preds)

#sum([ abs(x - y) for x,y in zip(y_test.values,preds)])

error
model_3_columns = ['starting_cost','overall','age','potential','element_type','short_name']

X_train_model_3 = X_train[model_3_columns].reset_index()

X_test_model_3 = X_test[model_3_columns].reset_index()

from sklearn.preprocessing import OneHotEncoder

OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)



OH_pos_club = OH_encoder.fit_transform(X_train_model_3[['element_type','short_name']])

club_pos_df_xtrain = pd.DataFrame(OH_pos_club)



X_train_model_3_encoded = pd.concat([X_train_model_3,club_pos_df_xtrain],axis = 1,ignore_index=False)

X_train_model_3_encoded = X_train_model_3_encoded.drop(['index','element_type','short_name'],axis = 1)



model_3 = XGBRegressor(random_state = 0,learning_rate = 0.05)

model_3.fit(X_train_model_3_encoded,y_train)



club_pos_df_xtest = OH_encoder.transform(X_test_model_3[['element_type','short_name']])



club_pos_df_xtest = pd.DataFrame(club_pos_df_xtest)



X_test_model_3_encoded = pd.concat([X_test_model_3,club_pos_df_xtest],axis = 1,ignore_index=False)

X_test_model_3_encoded = X_test_model_3_encoded.drop(['index','element_type','short_name'],axis = 1)



preds3 = model_3.predict(X_test_model_3_encoded)

error = mean_absolute_error(y_test,preds3)

errors_df_model_3 = pd.DataFrame(y_test.values,preds3).reset_index()

errors_df_model_3.columns = ['actual','predicted']

higher_ppg_errors = errors_df_model_3[errors_df_model_3['predicted']>2]

mae_value = mean_absolute_error(higher_ppg_errors.actual,higher_ppg_errors.predicted)
fifa_row = ['Ziyech' in name for name in fifa21_abbr['name']].index(True)

fifa_details = fifa21_abbr.loc[fifa_row]

fifa_details['overall']

player_df = pd.DataFrame(np.array([fifa_details['player_id'],80,fifa_details['overall'],fifa_details['age'],fifa_details['potential'],3,'CHE']).reshape(1,-1),columns = X_test_model_3.columns).apply(pd.to_numeric,errors = 'ignore')

player_transformed = OH_encoder.transform(player_df[['element_type','short_name']])

club_pos_player = pd.DataFrame(player_transformed)

player_encoded = pd.concat([player_df,club_pos_player],axis = 1,ignore_index=False)

player_encoded = player_encoded.drop(['index','element_type','short_name'],axis = 1)

player_pred = model_3.predict(player_encoded)
