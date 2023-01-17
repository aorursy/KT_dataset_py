
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


import os
print(os.listdir("../input"))
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
import dask.dataframe as dd
from pandas_summary import DataFrameSummary
from IPython.display import display
import statsmodels.api as sm
from sklearn import preprocessing
# Read play data
ngs = pd.read_csv('../input/NGS-2016-pre.csv')
ngs.columns = [col.lower() for col in ngs.columns]

pprd = pd.read_csv('../input/play_player_role_data.csv')
pprd.columns = [col.lower() for col in pprd.columns]

vr = pd.read_csv('../input/video_review.csv')
vr.columns = [col.lower() for col in vr.columns]

ngs2017pre = pd.read_csv('../input/NGS-2017-pre.csv')
ngs2017pre.columns = [col.lower() for col in ngs2017pre.columns]

video_injury = pd.read_csv('../input/video_footage-injury.csv')
video_injury.columns = [col.lower() for col in video_injury.columns]

play_player_role_data = pd.read_csv('../input/play_player_role_data.csv', low_memory=False)
play_player_role_data.columns = [col.lower() for col in play_player_role_data.columns]

play_info = pd.read_csv('../input/play_information.csv', low_memory=False)
play_info.columns = [col.lower() for col in play_info.columns]
import re
def get_return_yards(s):
    m = re.search('for ([0-9]+) yards', s)
    if m:
        return int(m.group(1))
    elif re.search('for no gain', s):
        return 0
    else:
        return np.nan

play_info['return_yards'] = play_info['playdescription'].map(
        lambda x: get_return_yards(x))
ngs_2016reg_part1 = pd.read_csv('../input/NGS-2016-reg-wk1-6.csv', low_memory=False)
ngs_2016reg_part1.columns = [col.lower() for col in ngs_2016reg_part1.columns]

ngs_2016reg_part2 = pd.read_csv('../input/NGS-2016-reg-wk7-12.csv', low_memory=False)
ngs_2016reg_part2.columns = [col.lower() for col in ngs_2016reg_part2.columns]

ngs_2016reg_part3 = pd.read_csv('../input/NGS-2016-reg-wk13-17.csv', low_memory=False)
ngs_2016reg_part3.columns = [col.lower() for col in ngs_2016reg_part3.columns]

ngs_2017reg_part1 = pd.read_csv('../input/NGS-2017-reg-wk1-6.csv', low_memory=False)
ngs_2017reg_part1.columns = [col.lower() for col in ngs_2017reg_part1.columns]

ngs_2017reg_part2 = pd.read_csv('../input/NGS-2017-reg-wk7-12.csv', low_memory=False)
ngs_2017reg_part2.columns = [col.lower() for col in ngs_2017reg_part2.columns]

ngs_2017reg_part3 = pd.read_csv('../input/NGS-2017-reg-wk13-17.csv', low_memory=False)
ngs_2017reg_part3.columns = [col.lower() for col in ngs_2017reg_part3.columns]
formation = pd.DataFrame()
result_of_punt_df = pd.DataFrame()
top_velocity = pd.DataFrame()
for i in range(len(vr.index)):
    
    game_details = vr[['season_year','gamekey','playid']].sort_values(['season_year','gamekey','playid']).loc[i]
    season_year = game_details.season_year
    gamekey = game_details.gamekey
    playid = game_details.playid
    
    season_dfs = [ngs2017pre, ngs, ngs_2016reg_part1, ngs_2016reg_part2, ngs_2016reg_part3, ngs_2017reg_part1,
                 ngs_2017reg_part2, ngs_2017reg_part3]

    for i in season_dfs:

        play = i[(i['season_year'] == season_year) &
               (i['gamekey'] == gamekey) &
               (i['playid'] == playid)]
        


        if len(play.index) != 0:
            ngs_with_roles = pd.merge(play, pprd)
            features_df = ngs_with_roles.drop(['season_year'], axis=1).pivot(index='time',
                                                                columns='role',
                                                                values=['x', 'y', 'dis', 'event', 'o', 'dir',  'gamekey', 'playid'])
            
            # Collect Roles and append them to the formation list
            formation = formation.append(pd.Series(features_df.x.columns))
                                    
            result_of_punt_df = result_of_punt_df.append(
                pd.DataFrame(features_df.event.GL.loc[(features_df.event['PR']=='punt_received') |
                                                    (features_df.event['PR']=='punt_downed') |
                                                    (features_df.event['PR']=='fair_catch') |
                                                    (features_df.event['PR']=='fumble_offense_recovered')]))
            
            for i in features_df.x.columns:
                features_df.loc[:,('velocity',i)] = features_df.dis[i] / 1.094 / 0.1 * 2.237 #mph
                
                
            max_velocity = pd.concat([features_df.velocity, features_df.gamekey.GL.rename('gamekey'), features_df.playid.GL.rename('playid')],axis=1, sort=False)

            max_velocity['game_play_id'] = max_velocity.gamekey.astype(str) + '_' + max_velocity.playid.astype(str)
            max_velocity = max_velocity.reset_index().drop(['time', 'playid', 'gamekey'],axis=1)
            max_velocity = max_velocity.groupby('game_play_id').max()
            
            top_velocity = top_velocity.append(max_velocity, sort=True)


play_player_role_data['game_play_id'] = play_player_role_data['gamekey'].astype(str) + '_' + play_player_role_data['playid'].astype(str)
play_player_role_data = play_player_role_data.sort_values('role')

play_info['game_play_id'] = play_info.gamekey.astype(str) + '_' + play_info.playid.astype(str)
vr['game_play_id'] = vr.gamekey.astype(str) + '_' + vr.playid.astype(str)
video_injury['game_play_id'] = video_injury.gamekey.astype(str) + '_' + video_injury.playid.astype(str)

play_formations = pd.DataFrame({'formation' :play_player_role_data.groupby('game_play_id').apply(lambda x: '_'.join(x['role']))})
play_formations = play_formations.reset_index()
injured_player_role = pd.merge(play_player_role_data, vr, left_on=['gsisid', 'gamekey', 'season_year', 'playid', 'game_play_id'], 
                               right_on=['gsisid', 'gamekey', 'season_year', 'playid', 'game_play_id'], how='inner')
injured_player_role[['primary_partner_gsisid']] = injured_player_role[['primary_partner_gsisid']].fillna(0).replace('Unclear', 1).astype(int)
primary_partner = play_player_role_data.drop(['season_year', 'gamekey', 'playid'], axis=1)
injured_player_role = pd.merge(injured_player_role, primary_partner,
                               left_on=['primary_partner_gsisid', 'game_play_id'],
                               right_on=['gsisid', 'game_play_id'], how='left')

injured_player_role =injured_player_role.drop(['gsisid_y'], axis=1)
injured_player_role = injured_player_role.rename(columns={'role_x': 'injury_role', 'role_y': 'primary_partner_role', 'gsisid_x': 'gsisid'})

injured_player_role.sort_values('gamekey', ascending=True);
all_punt_data = pd.merge(play_info, play_formations, how='outer', on='game_play_id')
all_punt_data = pd.merge(all_punt_data, top_velocity, how='inner', on='game_play_id')
all_punt_data = pd.merge(all_punt_data, vr, how='inner', on=['game_play_id', 'season_year', 'gamekey', 'playid'])
all_punt_data = pd.merge(all_punt_data, video_injury, how='left', on=['game_play_id', 'gamekey', 'playid', 'week', 'playdescription']).drop('quarter', axis=1)
all_punt_data[['primary_partner_gsisid']] = all_punt_data[['primary_partner_gsisid']].fillna(0).replace('Unclear', 1).astype(int)
all_punt_data = pd.merge(all_punt_data, injured_player_role, how='inner', 
                         left_on=['season_year', 'gamekey', 'playid', 'game_play_id', 'gsisid', 'player_activity_derived', 'turnover_related', 
                                  'primary_impact_type', 'primary_partner_gsisid', 'primary_partner_activity_derived', 'friendly_fire'],
                         right_on=['season_year', 'gamekey', 'playid', 'game_play_id', 'gsisid', 'player_activity_derived', 'turnover_related', 
                                  'primary_impact_type', 'primary_partner_gsisid', 'primary_partner_activity_derived', 'friendly_fire'])
result_of_punt_df.GL.value_counts().plot(kind='bar', title='Punt Results')
injured_player_role.injury_role.value_counts().plot(kind='bar', title='Role of Concussed')
injured_player_role.primary_partner_role.value_counts().plot(kind='bar', title='Role of Primary Partner')
receiving_position = ['PR', 'PFB', 'VR', 'PDR1', 'PDL2']

recieving_team = all_punt_data[all_punt_data.injury_role.isin(receiving_position)]
pd.value_counts(recieving_team['injury_role']).plot(kind='bar', title='Concussed data by receiving team position')
kicking_team = all_punt_data[~ all_punt_data.injury_role.isin(receiving_position)]
pd.value_counts(kicking_team['injury_role']).plot(kind='bar', title='Concussed data by punting team position')
injured_player_role.player_activity_derived.value_counts().plot(kind='bar', title='Activity of Player Before Injury')
injured_player_role.primary_partner_activity_derived.value_counts().plot(kind='bar', title='Activity of Primary Partner Before Injury')
impact_type =  injured_player_role.primary_impact_type.value_counts().plot(kind='bar', title='Primary Impact')
def get_hang_time(ngs_df, start_event='punt', *stop_events):
    punt_event = ngs_df.loc[ngs_df.event==start_event] \
        .groupby(['season_year', 'gamekey','playid'], as_index = False)['time'].min()
    punt_event.rename(columns = {'time':'punt_time'}, inplace=True)
    punt_event['punt_time'] = pd.to_datetime(punt_event['punt_time'],\
                                             format='%Y-%m-%d %H:%M:%S.%f')
    
    receiving_event = ngs_df.loc[ngs_df.event.isin(stop_events)] \
        .groupby(['season_year', 'gamekey','playid'], as_index = False)['time'].min()
    receiving_event.rename(columns = {'time':'receiving_time'}, inplace=True)
    receiving_event['receiving_time'] = pd.to_datetime(receiving_event['receiving_time'],\
                                             format='%Y-%m-%d %H:%M:%S.%f')
    
    punt_df = punt_event.merge(receiving_event, how='inner', on = ['season_year', 'gamekey','playid']) \
                .reset_index(drop=True)
    
    punt_df['hang_time'] = (punt_df['receiving_time'] - punt_df['punt_time']).dt.total_seconds()
    
    return punt_df
punt_df = get_hang_time(ngs_2016reg_part1, 'punt', 'punt_received', 'fair_catch')
print('The average hang time is {} seconds' .format(round(punt_df['hang_time'].mean(), 1)))
punt_df.hang_time.hist()
return_yards = play_info.return_yards.value_counts(ascending=True, bins=10)
return_yards.plot('bar', title='Yards Gained on Punts')