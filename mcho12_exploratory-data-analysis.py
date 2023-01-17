# Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from nltk import word_tokenize
from nltk.util import skipgrams

# Load relevant data for Exploratory Analysis

# Play Information
punt_play_info = pd.read_csv('../input/play_information.csv')

# Injury Plays
injury_plays = pd.read_csv('../input/video_review.csv')

# Player Role Data
role_player_data = pd.read_csv('../input/play_player_role_data.csv')
def play_type_col(s, key_words):
    token = [x.lower() for x in word_tokenize(s)]
    if all(x in token for x in key_words):
        return 'Y'
    else:
        return 'N'


def punt_return(s):
    token_punt = [x.lower() for x in word_tokenize(s)]
    for triple in list(skipgrams(token_punt, 3, 0)):
        if triple[0] == 'for':
            try:
                yards = int(triple[1])
                if triple[2] == 'yards':
                    return 'Y'
            except:
                continue
    return 'N'


def metrics_by_role(df, col, output_cols=None):
    tot = len(df)
    output_list = []
    for i in df[col].unique():
        num = len(df[df[col]==i])
        pct_tot_injury = 100*num/tot
        output_list.append([i, num, pct_tot_injury])
    
    if output_cols is None:
        output_df = pd.DataFrame(output_list, columns=['Role', 'Number of Injuries',
                                                      'Percent Total Injuries'])
        output_df = output_df[output_df['Number of Injuries'] != 0]
        return output_df.sort_values(by=['Percent Total Injuries'], ascending=False)


def metrics_by_play_type(df, cols):
    output_list = []
    for col in cols:
        name_of_play = ' '.join(col.split('_'))
        col_df = df[df[col]=='Y']
        col_len = len(col_df)
        pct_total = 100*col_len/len(df)
        output_list.append([name_of_play, col_len, pct_total])
        
    output_df = pd.DataFrame(output_list, columns=['play type', 'number of plays',
                                                  'percent of total plays'])
    
    output_df = output_df[output_df['number of plays'] != 0]
    
    return output_df.sort_values(by=['percent of total plays'], ascending=False)
    
    
def create_plot_from_table(df, x_col, y_col, y_label, title, axis_angle=0):
    x_cats = list(df[x_col])
    y_pos = np.arange(len(x_cats))
    y_vals = np.array(df[y_col])
    plt.bar(x_cats, y_vals, align='center', alpha=1)
    plt.xticks(y_pos, x_cats, rotation=axis_angle)
    plt.ylabel(y_label)
    plt.title(title)
    plt.show()
punt_play_info['fair_catch'] = punt_play_info.apply(
    lambda x: play_type_col(x['PlayDescription'], ['fair', 'catch']), axis=1)

punt_play_info['out_of_bounds'] = punt_play_info.apply(
    lambda x: play_type_col(x['PlayDescription'], ['out', 'of', 'bounds']), axis=1)

punt_play_info['downed'] = punt_play_info.apply(
    lambda x: play_type_col(x['PlayDescription'], ['downed', 'by']), axis=1)

punt_play_info['touchback'] = punt_play_info.apply(
    lambda x: play_type_col(x['PlayDescription'], ['touchback']), axis=1)

punt_play_info['no_play'] = punt_play_info.apply(
    lambda x: play_type_col(x['PlayDescription'], ['-', 'no', 'play']), axis=1)

punt_play_info['muff'] = punt_play_info.apply(
    lambda x: play_type_col(x['PlayDescription'], ['muffs']), axis=1)

punt_play_info['fumble'] = punt_play_info.apply(
    lambda x: play_type_col(x['PlayDescription'], ['fumbles']), axis=1)

punt_play_info['touchdown'] = punt_play_info.apply(
    lambda x: play_type_col(x['PlayDescription'], ['touchdown']), axis=1)

punt_play_info['block'] = punt_play_info.apply(
    lambda x: play_type_col(x['PlayDescription'], ['blocked']), axis=1)

punt_play_info['penalty'] = punt_play_info.apply(
    lambda x: play_type_col(x['PlayDescription'], ['penalty']), axis=1)

punt_play_info['fake_punt'] = punt_play_info.apply(
    lambda x: play_type_col(x['PlayDescription'], ['fake', 'punt']), axis=1)

punt_play_info['return'] = punt_play_info['PlayDescription'].apply(punt_return)
punts_join_injuries = punt_play_info.merge(
    injury_plays, left_on=['PlayID', 'Season_Year', 'GameKey'],
    right_on=['PlayID', 'Season_Year', 'GameKey'], how='left')

injury_role_player_join = injury_plays.merge(
    role_player_data, left_on=['Season_Year', 'GameKey', 'PlayID', 'GSISID'],
    right_on=['Season_Year', 'GameKey', 'PlayID', 'GSISID'], how='left')

punts_join_injuries_only = punt_play_info.merge(
    injury_plays, left_on=['PlayID', 'Season_Year', 'GameKey'],
    right_on=['PlayID', 'Season_Year', 'GameKey'], how='right')
metrics_by_play_type(punt_play_info, ['fair_catch', 'out_of_bounds', 'downed', 
        'touchback', 'no_play', 'muff', 'fumble', 'block', 'fake_punt', 'return'])
table = metrics_by_play_type(punt_play_info, ['fair_catch', 'downed', 'muff', 'fumble', 'return'])
create_plot_from_table(table, 'play type', 'number of plays', 'Count of Injuries', 'Number of Injuries by Play Type')
table = metrics_by_play_type(punts_join_injuries_only, ['fair_catch', 'out_of_bounds', 'downed', 
        'touchback', 'muff', 'fumble', 'block', 'fake_punt', 'return'])
create_plot_from_table(table, 'play type', 'number of plays', 'Count of Injuries', 'Number of Injuries by Play Type')
print('Percent of Return Plays that result in injury: {} percent, ({}/{} plays)'.format(100*29/2510, 29, 2510))
print('Percent of Fair Catch Plays that result in injury: {} percent, ({}/{} plays)'.format(100*2/1663, 2, 1663))
injuries_by_role = metrics_by_role(injury_role_player_join, 'Role')
injuries_by_role
num_kicking_team_injuries = 0
num_return_team_injuries = 0
for row in range(len(injuries_by_role)):
    role = injuries_by_role['Role'].iloc[row]
    if role in ['PLW', 'PRG', 'GL', 'PLG', 'PRW', 'PRT', 'PLT', 'PLS', 'GR', 'P', 'PPR']:
        num_kicking_team_injuries += injuries_by_role['Number of Injuries'].iloc[row]
    elif role in ['PR', 'VR', 'PFB', 'PDR1', 'PDL2', 'PLL']:
        num_return_team_injuries += injuries_by_role['Number of Injuries'].iloc[row]

table = pd.DataFrame([['Kicking Team', num_kicking_team_injuries, num_kicking_team_injuries*100/37], 
         ['Returning Team', num_return_team_injuries, num_return_team_injuries*100/37]], 
                     columns=['Team', 'Number of Injuries', 'Percent of Total Injuries'])

create_plot_from_table(table, 'Team', 'Number of Injuries', 'Count of Injuries', 'Number of Injuries by Team')
table = metrics_by_role(injury_role_player_join, 'Primary_Impact_Type')
create_plot_from_table(table, 'Role', 'Number of Injuries', 'Count of Injuries', 
                       'Number of Injuries by Contact', axis_angle=45)
table = metrics_by_role(injury_role_player_join, 'Primary_Partner_Activity_Derived')
create_plot_from_table(table, 'Role', 'Number of Injuries', 'Count of Injuries', 
                       'Number of Injuries by Activity', axis_angle=45)
def convert_to_mph(dis_vector, converter):
    mph_vector = dis_vector * converter
    return mph_vector

def get_speed(ng_data, playId, gameKey, player, partner,use_loaded_table = False):
    
    if use_loaded_table==False:
        ng_data = pd.read_csv(ng_data)
    else:
        #ng_data is the table not the file location
        pass
    ng_data['mph'] = convert_to_mph(ng_data['dis'], 20.455)
    player_data = ng_data.loc[(ng_data.GameKey == gameKey) & (ng_data.PlayID == playId) 
                               & (ng_data.GSISID == player)].sort_values('Time')
    partner_data = ng_data.loc[(ng_data.GameKey == gameKey) & (ng_data.PlayID == playId) 
                              & (ng_data.GSISID == partner)].sort_values('Time')
    player_grouped = player_data.groupby(['GameKey','PlayID','GSISID'], 
                               as_index = False)['mph'].agg({'max_mph': max,
                                                             'avg_mph': np.mean
                                                            })
    player_grouped['involvement'] = 'player_injured'
    partner_grouped = partner_data.groupby(['GameKey','PlayID','GSISID'], 
                               as_index = False)['mph'].agg({'max_mph': max,
                                                             'avg_mph': np.mean
                                                            })
    partner_grouped['involvement'] = 'primary_partner'
    return pd.concat([player_grouped, partner_grouped], axis = 0)[['involvement',
                                                                   'max_mph',
                                                                   'avg_mph']].reset_index(drop=True)

#Run an example
get_speed('../input/NGS-2016-pre.csv', 3129, 5, 31057, 32482)
injury_plays.columns = [col.lower() for col in injury_plays.columns]
for _file in ['../input/NGS-2016-pre.csv',
              '../input/NGS-2016-reg-wk1-6.csv',
              '../input/NGS-2016-reg-wk7-12.csv',
              '../input/NGS-2016-reg-wk13-17.csv',
              '../input/NGS-2016-post.csv',
              '../input/NGS-2017-pre.csv',
              '../input/NGS-2017-reg-wk1-6.csv',
              '../input/NGS-2017-reg-wk7-12.csv',
              '../input/NGS-2017-reg-wk13-17.csv',
              '../input/NGS-2017-post.csv']:
    ng_data = pd.read_csv(_file, low_memory=False)
    for idx, row in injury_plays.iterrows():
        try:
            a = get_speed(ng_data, row.playid, row.gamekey, row.gsisid, int(row.primary_partner_gsisid),use_loaded_table=True)
            speeds = a.values.flatten().flatten()[[1,2,4,5]]
            injury_plays.at[idx,'player_max_mph'] = speeds[0]
            injury_plays.at[idx,'player_avg_mph'] = speeds[1]
            injury_plays.at[idx,'primary_partner_max_mph'] = speeds[2]
            injury_plays.at[idx,'primary_partner_avg_mph'] = speeds[3]
        except:
            continue
speed_results = \
injury_plays.groupby(['player_activity_derived',
            'primary_partner_activity_derived',
            'friendly_fire']).\
                agg({'playid':'count',
                     'player_max_mph':'mean',
                     'primary_partner_max_mph':'mean',
                     'player_avg_mph':'mean',
                     'primary_partner_avg_mph':'mean'}).round(1)
speed_results
