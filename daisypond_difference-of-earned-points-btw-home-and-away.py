import matplotlib.pyplot as plt

import numpy as np

import os

import pandas as pd

import zipfile

import glob

import json
season_stats1718 = json.load(open('../input/datafilev2/datafile/season17-18/season_stats.json','r'))

print("{}".format(json.dumps(season_stats1718,indent=2))[:3000])
match_stats1718 = json.load(open('../input/datafilev2/datafile/season17-18/season_match_stats.json','r'))

print("{}".format(json.dumps(match_stats1718,indent=2))[:3000])
match_df = pd.DataFrame.from_dict(match_stats1718, orient='index')



print(match_df.shape)

match_df[:5]


def divideScore(df, scoreName, name1, name2):

    df2 = df[scoreName].str.split(':',expand=True)

    df2.rename(columns={0: name1, 1: name2}, inplace=True)

    

    df2 = pd.concat([df, df2], axis=1)



    return df2



match_df = divideScore(match_df,'half_time_score','half_time_score_home', 'half_time_score_away')

match_df = divideScore(match_df,'full_time_score','full_time_score_home', 'full_time_score_away')
match_df[:5]
match_df = match_df.astype({'half_time_score_home': 'int8', 'half_time_score_away': 'int8', 'full_time_score_home': 'int8', 'full_time_score_away': 'int8'})



match_df.loc[match_df['full_time_score_home'] == match_df['full_time_score_away'], 'result_home'] = 'D'

match_df.loc[match_df['full_time_score_home'] > match_df['full_time_score_away'], 'result_home'] = 'W'

match_df.loc[match_df['full_time_score_home'] < match_df['full_time_score_away'], 'result_home'] = 'L'
match_df[:10]
clubs = match_df['home_team_name'].unique()

clubs
club_df = pd.DataFrame()



grp_home = match_df.groupby('home_team_name')

grp_away  = match_df.groupby('away_team_name')



def resultCount(df, str):

    if (str in df.keys()):

        return df[str]

    else:

        return 0

    

for c in clubs:

        home_matches = grp_home.get_group(c)

        away_matches = grp_away.get_group(c)

        

        result_home = home_matches['result_home'].value_counts(normalize=False)

        result_away =  away_matches['result_home'].value_counts(normalize=False)

        

        home_win   = resultCount(result_home,'W')

        home_draw = resultCount(result_home,'D')

        home_lose  = resultCount(result_home,'L')

        

        away_win   = resultCount(result_away,'L')

        away_draw = resultCount(result_away,'D')

        away_lose  = resultCount(result_away,'W')

        

        home_point = home_win *3 + home_draw

        away_point = away_win *3 + away_draw

        

        new_data = [(c, home_win, home_draw, home_lose, away_win, away_draw, away_lose, home_point, away_point)]

        

        club_df = club_df.append(new_data)



club_df.columns=['Club','Home_win','Home_draw','Home_lose','Away_win','Away_draw','Away_lose','Home_point','Away_point']



club_df
plt.figure(figsize=(15, 15), dpi=50)

plt.rcParams["font.size"] = 18



plt.xlabel("Points at home")

plt.ylabel("Points at away")



plt.xlim([0,60])

plt.ylim([0,60])

plt.plot([0,60],[0,60])

plt.scatter(club_df['Home_point'],club_df['Away_point'])



for (i, j, k) in zip(club_df['Club'], club_df['Home_point'],club_df['Away_point']):

             plt.annotate(i, xy=(j, k))



plt.show()
match_key = season_stats1718.keys()

match_df = pd.DataFrame()



for key in match_key:

    

    for team_key in season_stats1718[key]:

        

        x = season_stats1718[key][team_key]

        # a_df = season_stats1718[key][team_key]['team_details']

        a_df = pd.io.json.json_normalize(season_stats1718[key][team_key]['team_details'])

        b_df = pd.io.json.json_normalize(season_stats1718[key][team_key]['aggregate_stats'])

        c_df = pd.concat([a_df,b_df], axis =1)

        match_df = pd.concat([match_df, c_df], sort=False)



match_df = match_df.fillna(0)

match_df.head(5)
match_df.dtypes

int_val = ['accurate_pass',  'aerial_lost', 'aerial_won', 'att_miss_left', 

          'att_miss_right','att_sv_low_centre', 'blocked_scoring_att', 'fk_foul_lost',

       'ontarget_scoring_att', 'shot_off_target',

       'total_offside', 'total_pass', 'total_scoring_att', 'total_tackle',

       'total_throws', 'won_contest', 'won_corners', 'att_goal_low_centre',

       'att_goal_low_left', 'att_goal_low_right', 'att_pen_goal',

       'att_sv_high_centre', 'att_sv_low_left', 'goals', 'att_miss_high',

       'att_miss_high_right', 'att_post_left', 'att_sv_high_right',

       'att_sv_low_right', 'post_scoring_att', 'att_sv_high_left',

       'att_miss_high_left', 'att_goal_high_left', 'att_goal_high_right',

       'att_post_right', 'att_goal_high_centre', 'penalty_save',

       'att_post_high']



for i in int_val:

    match_df[i] = match_df[i].astype(np.int64)



match_df['possession_percentage'] = match_df['possession_percentage'].astype(np.float64)

match_df.dtypes
grp = match_df.groupby('team_name' , as_index=False)

grp_mean = grp.mean()

grp_mean
import seaborn as sns



plt.rcParams["font.size"] = 9



grp_mean.plot.scatter(x='possession_percentage', y='goals')



for (i, j, k) in zip(grp_mean['team_name'], grp_mean['possession_percentage'],grp_mean['goals']):

             plt.annotate(i, xy=(j, k))



        

sns.regplot(x="possession_percentage", y="goals", data=grp_mean);



plt.show()
att = ['possession_percentage','total_pass', 'goals']

pd.plotting.scatter_matrix(match_df[att])