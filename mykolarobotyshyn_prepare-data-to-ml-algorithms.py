import sqlite3

import pandas as pd





def to_csv():

    db = sqlite3.connect('../input/database.sqlite')

    cursor = db.cursor()

    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")

    tables = cursor.fetchall()

    for table_name in tables:

        table_name = table_name[0]

        table = pd.read_sql_query("SELECT * from %s" % table_name, db)

        table.to_csv(table_name + '.csv', index_label='index')

to_csv()
match=pd.read_csv('Match.csv')

epl_match=match[match['league_id']==1729]

print(epl_match.shape)

print(epl_match['season'].value_counts())
print(match.columns.tolist())
columns=epl_match.columns.tolist()

# delete unuseful features

# coordinates, bet, stats(bcs we don't know this stats before game, we want to predict)

coord_index_start=columns.index('home_player_X1')

coord_index_end=columns.index('away_player_Y11')

goal_index=columns.index('goal')

columns=columns[:coord_index_start]+columns[coord_index_end+1:goal_index]

columns.append('B365H')

columns.append('B365D')

columns.append('B365A')

epl_match=epl_match[columns]

epl_match.drop(['country_id','league_id','stage','index','id'],axis=1,inplace=True)

epl_match['date']=pd.to_datetime(epl_match['date'])

print(epl_match.columns.tolist())
team=pd.read_csv('Team.csv',encoding='latin1')

team=team.drop(['team_fifa_api_id','team_short_name','index','id'],axis=1)

# add two columns with teams(home and away names) and sort data by date

epl_match=epl_match.merge(team,left_on='home_team_api_id',right_on='team_api_id')

epl_match=epl_match.merge(team,left_on='away_team_api_id',right_on='team_api_id')

epl_match.drop(['team_api_id_x','team_api_id_y'],inplace=True,axis=1)

epl_match.rename(columns={'team_long_name_x':'home_team_name','team_long_name_y':'away_team_name'},inplace=True)

epl_match.sort_values(by=['date'],inplace=True)

epl_match.reset_index(drop=True,inplace=True)

print(epl_match.columns.tolist())
def get_match_result(raw):

    # 1 -home win, 0 - draw, 2 - away win

    home_goals=raw['home_team_goal']

    away_goals=raw['away_team_goal']

    if home_goals>away_goals:

        return 1

    elif home_goals==away_goals:

        return 0

    else:

        return 2

# 1. create result column

epl_match['result']=epl_match.apply(get_match_result,axis=1)

print(epl_match['result'])
player_attributes=pd.read_csv('Player_Attributes.csv')

features=['player_api_id','date','overall_rating','potential']

player_attributes=player_attributes[features]

player_attributes['date']=pd.to_datetime(player_attributes['date'])

player_attributes.dropna(inplace=True)

print(player_attributes.columns.tolist())
def get_closest_date_to_match(match_date,other_dates):

    # return date from other_dates,that is the closest to given (match_date)

    if len(other_dates)==0:

        return 0

    elif len(other_dates)==1:

        return other_dates

    else:

        diff=[abs(match_date-i) for i in other_dates]

        closest_date=min(diff)

        return other_dates[diff.index(closest_date)]





def get_teams_rating(raw):

    '''

    :param raw: raw from dataframe 

    :return:  new raw in dataframe with new column

    1. home_top_player

    2. away_top_player

    3. home_team_mean_rating

    4. away_team_mean_rating

    '''

    '''

    Algorithm:

    1. Extract players id and divide it into home id's and away id's

    2. for each player find the closest data to extract current ratings

    3. based on each player ratings find team average and number of top players.

    '''



    match_date=raw['date']

    home_player_1_index = raw.keys().tolist().index('home_player_1')

    away_player_1_index = raw.keys().tolist().index('away_player_1')

    home_players_api_id = raw[home_player_1_index:home_player_1_index + 11]

    away_players_api_id = raw[away_player_1_index:away_player_1_index + 11]

    home_players_ratings = []

    away_players_ratings = []



    for id in home_players_api_id:

        if np.isnan(id):

            home_players_ratings.append(0)

            continue

        id_df=player_attributes[player_attributes['player_api_id']==id]

        if id_df.empty:

            home_players_ratings.append(0)

            continue

        dates=id_df['date']

        closest_date=get_closest_date_to_match(match_date,list(dates))

        if closest_date==0:

            home_players_ratings.append(0)

            continue

        id_df=id_df[id_df['date']==closest_date]

        ratings=id_df[['overall_rating','potential']]

        mean_rating=float((ratings['overall_rating']+ratings['potential'])/2)

        home_players_ratings.append(mean_rating)



    for id in away_players_api_id:

        if np.isnan(id):

            home_players_ratings.append(0)

            continue

        id_df = player_attributes[player_attributes['player_api_id'] == id]

        if id_df.empty:

            home_players_ratings.append(0)

            continue

        dates = id_df['date']

        closest_date = get_closest_date_to_match(match_date, list(dates))

        if closest_date==0:

            away_players_ratings.append(0)

            continue

        ratings = id_df[id_df['date'] == closest_date][['overall_rating','potential']]

        mean_rating = float((ratings['overall_rating'] + ratings['potential']) / 2)

        away_players_ratings.append(mean_rating)



    top_rating_level = 83.5



    raw['home_top_players'] = len([x for x in home_players_ratings if x > top_rating_level])

    raw['away_top_players'] = len([x for x in away_players_ratings if x > top_rating_level])

    raw['home_team_mean_rating'] = np.average([x if x>0 else np.average([i for i in home_players_ratings if i>0]) for x in home_players_ratings])

    raw['away_team_mean_rating'] = np.average([x if x>0 else np.average([i for i in away_players_ratings if i>0]) for x in away_players_ratings])

    return raw

def get_team_points_per_match(raw,args):

    '''

    :param raw: 

    :param args: team name 

    :return: result(0,1,2) -> points

    '''

    team=args

    if raw['result']==0:

        return 1

    else:

        if raw['home_team_name']==team:

            if raw['result']==1:

                return 3

            else:

                return 0

        else:

            if raw['result']==1:

                return 0

            else:

                return 3



    
def transform_to_sum_5(points):

    '''

    :param points: array, where el=how many points team earned in match 

    :return: each el=points in last 5 match (ex: i=10, array[10]=sum(array[9]+array[8]+...array[5]))

    '''

    last_5_points = [0]

    for i in range(1, len(points)):

        if i < 5:

            last_5_points.append(sum(points[:i]))

        else:

            last_5_points.append(sum(points[i - 5:i]))

    return last_5_points

def assign_last_5_points_to_team(raw,args1,args2):

    '''

    

    :param raw: raw of datadrame

    :param args1: team name

    :param args2: dictionary, where key=index of raw, value = last_5_points

    :return: updated raw

    '''

    team=args1

    d=args2

    index=raw.name

    value=d[index]



    if raw['home_team_name']==team:

        raw['home_team_points_last_5']=value

    else:

        raw['away_team_points_last_5'] = value

    return raw

def assign_side_last_5_points_to_team(raw,args1,args2,args3):

    '''

    

    :param raw: dataframe's raw 

    :param args1: team nane

    :param args2: dictionary where where key=index of raw, value = last_5_points

    :param args3: 1(home) or not 1(away)

    :return: updated row of dataframe

    '''

    team = args1

    d = args2

    index = raw.name

    value = d[index]



    if args3==1:

        column='home_team_points_home_last_5'

    else:

        column = 'away_team_points_away_last_5'



    raw[column]=value

    return raw

def get_team_last_5(team,team_dataframe):

    # return array: each el=points in last 5 match (ex: i=10, array[10]=sum(array[9]+array[8]+...array[5]))

    team_df=team_dataframe.copy()

    results=team_df.apply(get_team_points_per_match,args=(team,),axis=1)

    points=results.values

    # points is array where each el describe how many points team earned per game

    last_5_points=transform_to_sum_5(points)

    return last_5_points



def get_season_last_5(season,dataframe):

    season_epl=dataframe[dataframe['season']==season]

    teams=set(season_epl['home_team_name'].values)



    for team in teams:

        team_df = season_epl[(season_epl['home_team_name'] == team) | (season_epl['away_team_name'] == team)]

        # 1. last 5 matches (in general)

        indexes=team_df.index.values

        points=get_team_last_5(team,team_df)

        # create dict, where key=index of raw, value = last_5_points

        d={indexes[i]:points[i] for i in range(len(points))}

        team_df=team_df.apply(assign_last_5_points_to_team,args=(team,d),axis=1)

        season_epl.update(team_df)

        # 2. last 5 matches home

        team_df=team_df[team_df['home_team_name']==team]

        indexes=team_df.index.values

        points=get_team_last_5(team,team_df)

        d = {indexes[i]: points[i] for i in range(len(points))}

        team_df = team_df.apply(assign_side_last_5_points_to_team, args=(team, d,1), axis=1)

        season_epl.update(team_df)



        # 3. last 5 matches away

        team_df = season_epl[season_epl['away_team_name'] == team]

        indexes = team_df.index.values

        points = get_team_last_5(team, team_df)

        d = {indexes[i]: points[i] for i in range(len(points))}

        team_df = team_df.apply(assign_side_last_5_points_to_team, args=(team, d, 2), axis=1)

        season_epl.update(team_df)

        #check=season_epl[(season_epl['home_team_name'] == team) | (season_epl['away_team_name'] == team)]

        #print('{} in {} updated'.format(team,season))



    return season_epl



def get_last_5_matches(dataframe):

    epl_updated=dataframe.copy()

    epl_updated.loc[:,'home_team_points_last_5']=['not updated']*epl_updated.shape[0]

    epl_updated.loc[:, 'away_team_points_last_5'] = ['not updated'] * epl_updated.shape[0]

    epl_updated.loc[:, 'home_team_points_home_last_5'] = ['not updated'] * epl_updated.shape[0]

    epl_updated.loc[:, 'away_team_points_away_last_5'] = ['not updated'] * epl_updated.shape[0]

    seasons=set(epl_updated['season'].values)

    for season in seasons:

        df_season=get_season_last_5(season,epl_updated)

        epl_updated.update(df_season)

        print('season {} updated'.format(season))

    return epl_updated

### get teams rating from Team_Attributes



team_attributes=pd.read_csv('Team_Attributes.csv')

team_features=['index','team_api_id','date','buildUpPlaySpeed','buildUpPlayPassing',

               'chanceCreationPassing', 'chanceCreationCrossing','chanceCreationShooting',

               'defencePressure','defenceAggression','defenceTeamWidth']

# buildUpPlayDribling column has many missed value, that's why i remove it from team_features

# team_features - numeric columns

team_attributes=team_attributes[team_features]

team_attributes['date']=pd.to_datetime(team_attributes['date'])

print(team_attributes.isnull().sum())

print(team_attributes.dtypes)

def get_team_attributes(raw,args1):

    '''

    

    :param raw: raw of dataframe 

    :param args1: 1 or 2 : 1 -mean difference stats, 2: stats for both side

    :return: 

    '''

    match_date=raw['date']

    home_team_id=raw['home_team_api_id']

    away_team_id=raw['away_team_api_id']

    home_df=team_attributes[team_attributes['team_api_id']==home_team_id]

    home_dates=home_df['date']

    home_closest_date=get_closest_date_to_match(match_date,list(home_dates))

    away_df=team_attributes[team_attributes['team_api_id']==away_team_id]

    away_dates=away_df['date']

    away_closest_date=get_closest_date_to_match(match_date,list(away_dates))



    home_df=home_df[home_df['date']==home_closest_date]

    away_df=away_df[away_df['date']==away_closest_date]

    stats_features = ['buildUpPlaySpeed', 'buildUpPlayPassing',

                     'chanceCreationPassing', 'chanceCreationCrossing', 'chanceCreationShooting',

                     'defencePressure', 'defenceAggression', 'defenceTeamWidth']



    action=args1

    if action==1:

        for feature in stats_features:

            home_feature_value=home_df[feature].values[0]

            away_feature_value = away_df[feature].values[0]

            raw[feature+'D']=home_feature_value-away_feature_value

            #print('raw successfully updated')

    elif action==2:

        for feature in stats_features:

            raw['home_'+feature]=home_df[feature].values[0]

            raw['away_'+feature]=away_df[feature].values[0]

            #print('raw successfully updated')

    else:

        return Exception

    return raw

# 2. get teams rating(based on players)

epl_match=epl_match.apply(get_teams_rating,axis=1)

print(epl_match.columns.tolist())

# 3. get points from last 5 matches

epl_match=get_last_5_matches(epl_match)

print(epl_match.columns.tolist())
# 4. get team_attributes as difference

epl_match=epl_match.apply(get_team_attributes,args=(1,),axis=1)

print(epl_match.columns.tolist())

# remove unuseful features for machine learning algorithms( players id, teams id)

num_columns=epl_match.columns.tolist()

num_columns.remove('home_team_api_id')

num_columns.remove('away_team_api_id')

print(num_columns)

player_index=num_columns.index('home_player_1')

num_columns=num_columns[:player_index]+num_columns[player_index+22:] # 22 bcs 22 players

# assign new dataframe

ml_epl_match=epl_match[num_columns]

print(ml_epl_match.isnull().sum())

print(ml_epl_match.columns.tolist())



# write dataframe to csv

ml_epl_match.to_csv('my_data.csv')

print('Everything OK =)')