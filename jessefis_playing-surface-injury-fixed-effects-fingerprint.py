import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from datetime import datetime

import datetime

from sklearn.preprocessing import StandardScaler

from datetime import timedelta

import re

from scipy import sparse

from sklearn.linear_model import Ridge

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import GridSearchCV

import math



import pandas as pd

import numpy as np

from sklearn import linear_model

from sklearn import model_selection

from sklearn import metrics

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report

import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression

from sklearn.linear_model import SGDClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import learning_curve

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import Ridge

from sklearn import preprocessing

from datetime import datetime, timedelta

import datetime

from xgboost.sklearn import XGBRegressor

from xgboost.sklearn import XGBClassifier

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import cross_val_predict

from sklearn.model_selection import GroupKFold

from sklearn.metrics import r2_score

import seaborn as sns



import statsmodels.api as sm



from sklearn.linear_model import Ridge

from sklearn.linear_model import RidgeCV

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import LogisticRegression

from scipy.stats.mstats import winsorize



from sklearn.cluster import MiniBatchKMeans 

from scipy.spatial.distance import cdist

import matplotlib.style as style



play_list = pd.read_csv('../input/nfl-playing-surface-analytics/PlayList.csv', delimiter=',')

# play_list.info()

# play_list.head().T



injuries = pd.read_csv('../input/nfl-playing-surface-analytics/InjuryRecord.csv', delimiter=',')

# injuries.info()

# injuries.head()



# Pick first body part for duplicate injuries

play_list['BodyPart'] = play_list[['PlayKey']].merge(injuries.drop_duplicates(subset=['PlayKey']), how='left',left_on=['PlayKey'],right_on=['PlayKey'],validate='many_to_one')['BodyPart'].values

play_list['Surface'] = play_list[['PlayKey']].merge(injuries.drop_duplicates(subset=['PlayKey']), how='left',left_on=['PlayKey'],right_on=['PlayKey'],validate='many_to_one')['Surface'].values

play_list['has_injury'] = np.where(play_list['BodyPart'].isnull(), 0, 1)



play_list['PlayType'] = play_list.PlayType.str.replace(' Not Returned| Returned','')



play_list['workload_num_plays'] = play_list.PlayKey.str.replace('.*-','').astype(int)



prior_game = play_list.copy().sort_values(['PlayerKey','PlayerDay'])[['PlayerKey','PlayerDay','FieldType']].drop_duplicates()

prior_game['prior_PlayerDay'] = prior_game['PlayerDay'].shift()

prior_game['prior_PlayerKey'] = prior_game['PlayerKey'].shift()

prior_game['prior_FieldType'] = prior_game['FieldType'].shift()



prior_game['days_rest'] = np.where(prior_game['prior_PlayerKey']==prior_game['PlayerKey'],

                                  prior_game['PlayerDay']-prior_game['prior_PlayerDay'], np.nan)

prior_game['prior_field_type'] = np.where(prior_game['prior_PlayerKey']==prior_game['PlayerKey'], prior_game['prior_FieldType'], 'Unknown')



play_list['workload_days_rest'] = play_list[['PlayerKey','PlayerDay']].merge(prior_game, how='left',left_on=['PlayerKey','PlayerDay'],right_on=['PlayerKey','PlayerDay'],validate='many_to_one')['days_rest'].values

# 4, 6, 7, 8+

play_list['workload_days_rest'] = np.where(play_list['workload_days_rest']<=7, play_list['workload_days_rest'].fillna('-1').astype(int).astype(str), '8+')



play_list['stadium_type'] = np.where(play_list['StadiumType'].str.contains('Out|Ourdoor|Oudoor|Open|Cloudy|open|Heinz Field|Bowl'),

                                     'Open air', 'Indoor')



play_list['weather'] = np.where(play_list['stadium_type']=="Indoor", 'Indoor',

                            np.where(play_list['Weather'].str.contains('Rain|Showers|rain'), 'Rain', 

                            np.where(play_list['Weather'].str.contains('Snow|snow|Cold'), 'Cold/Snow', 

                            np.where(play_list['Weather'].str.contains('Cloudy|cloudy|Clouidy|Coudy|Overcast|Hazy'), 'Cloudy', 

                            np.where(play_list['Weather'].str.contains('Clear|clear|Fair'), 'Clear/Fair', 

                            np.where(play_list['Weather'].str.contains('Sunny|sunny|Sun|Heat'), 'Sunny', 'Unknown'))))))



play_list['injury'] = play_list.BodyPart.map({'Ankle':'Ankle/Foot','Foot':'Ankle/Foot','Knee':'Knee'}).fillna('No Injury')



play_list['role'] = play_list['PlayType'] + '_' + play_list['PositionGroup']



play_list['field_type_previous_game'] = play_list[['PlayerKey','PlayerDay']].merge(prior_game, how='left',left_on=['PlayerKey','PlayerDay'],right_on=['PlayerKey','PlayerDay'],validate='many_to_one')['prior_field_type'].values

play_list['field_type_previous_game'] = play_list['field_type_previous_game'] + ' -> ' + play_list['FieldType']

play_list['field_type_previous_game_Natural_to_Synthetic'] = np.where(play_list['field_type_previous_game'] == 'Natural -> Synthetic', 1, 0)

play_list['field_type_previous_game_Synthetic_to_Natural'] = np.where(play_list['field_type_previous_game'] == 'Synthetic -> Natural', 1, 0)

play_list['field_type_previous_game_Same'] = np.where(play_list['field_type_previous_game'] == play_list['FieldType'], 1, 0)



most_freq_field_type = play_list.groupby(['PlayerKey','FieldType'])['PlayKey'].count().reset_index().sort_values('PlayKey',ascending=False).drop_duplicates(subset=['PlayerKey'])[['PlayerKey','FieldType']]

play_list['field_type_most_frequent'] = play_list[['PlayerKey']].merge(most_freq_field_type, how='left',left_on=['PlayerKey'],right_on=['PlayerKey'],validate='many_to_one')['FieldType'].values

play_list['field_type_most_frequent'] = play_list['field_type_most_frequent'] + ' -> ' + play_list['FieldType']

play_list['field_type_most_frequent_Natural_to_Synthetic'] = np.where(play_list['field_type_most_frequent'] == 'Natural -> Synthetic', 1, 0)

play_list['field_type_most_frequent_Synthetic_to_Natural'] = np.where(play_list['field_type_most_frequent'] == 'Synthetic -> Natural', 1, 0)

play_list['field_type_most_frequent_Same'] = np.where(play_list['field_type_most_frequent'] == play_list['FieldType'], 1, 0)



tracking_data = pd.read_csv('../input/nfl-playing-surface-analytics/PlayerTrackData.csv', delimiter=',',

           dtype={'x': np.float32,'y': np.float32,'dir': np.float32,'dis': np.float32,'time': np.float32},

           usecols=['PlayKey','time','event','x','y','dir','dis']).sort_values(['PlayKey','time'])

# tracking_data.info()

# tracking_data.head()



tracking_data['time_prev'] = np.where(tracking_data.PlayKey != tracking_data.PlayKey.shift(), np.nan, tracking_data.time.shift()).astype(np.float32)

tracking_data['speed'] = round(tracking_data['dis']/(tracking_data['time']-tracking_data['time_prev']),2)

tracking_data['speed_prev'] = np.where(tracking_data.PlayKey != tracking_data.PlayKey.shift(), np.nan, tracking_data.speed.shift()).astype(np.float32)

tracking_data['dir_prev'] = np.where(tracking_data.PlayKey != tracking_data.PlayKey.shift(), np.nan, tracking_data.dir.shift()).astype(np.float32)



play_start = tracking_data[tracking_data.event.isin(['ball_snap','kickoff','snap_direct','onside_kick'])].drop_duplicates(subset=['PlayKey'])

# print(len(tracking_data.drop_duplicates(subset=['PlayKey'])))

# print(len(play_start))



play_end = tracking_data[tracking_data.event.isin(['pass_outcome_incomplete','tackle',

        'out_of_bounds','qb_sack','touchback','touchdown','field_goal','extra_point','fair_catch','pass_outcome_touchdown',

        'qb_kneel','punt_downed','field_goal_missed','qb_spike','extra_point_missed','safety','two_point_conversion',

        'kick_recovered','field_goal_blocked','extra_point_blocked','qb_strip_sack'])].drop_duplicates(subset=['PlayKey'])

# print(len(tracking_data.drop_duplicates(subset=['PlayKey'])))

# print(len(play_end))



# print(len(tracking_data))



tracking_data['time_start'] = tracking_data[['PlayKey']].merge(play_start, how='left',left_on=['PlayKey'],right_on=['PlayKey'],validate='many_to_one')['time'].values

tracking_data = tracking_data[tracking_data['time'] >= tracking_data['time_start']]

# print(len(tracking_data))



tracking_data['time_end'] = tracking_data[['PlayKey']].merge(play_end, how='left',left_on=['PlayKey'],right_on=['PlayKey'],validate='many_to_one')['time'].values

tracking_data = tracking_data[tracking_data['time'] <= tracking_data['time_end']]

# print(len(tracking_data))



tracking_data['x_start'] = tracking_data[['PlayKey']].merge(play_start, how='left',left_on=['PlayKey'],right_on=['PlayKey'],validate='many_to_one')['x'].values

tracking_data['y_start'] = tracking_data[['PlayKey']].merge(play_start, how='left',left_on=['PlayKey'],right_on=['PlayKey'],validate='many_to_one')['y'].values

tracking_data['dir_start'] = tracking_data[['PlayKey']].merge(play_start, how='left',left_on=['PlayKey'],right_on=['PlayKey'],validate='many_to_one')['dir'].values



tracking_data['x_end'] = tracking_data[['PlayKey']].merge(play_end, how='left',left_on=['PlayKey'],right_on=['PlayKey'],validate='many_to_one')['x'].values

tracking_data['y_end'] = tracking_data[['PlayKey']].merge(play_end, how='left',left_on=['PlayKey'],right_on=['PlayKey'],validate='many_to_one')['y'].values



tracking_data['x_prev'] = np.where(tracking_data.PlayKey != tracking_data.PlayKey.shift(), np.nan, tracking_data.x.shift()).astype(np.float32)

tracking_data['y_prev'] = np.where(tracking_data.PlayKey != tracking_data.PlayKey.shift(), np.nan, tracking_data.y.shift()).astype(np.float32)



tracking_data['time_norm'] = round(tracking_data['time'] - tracking_data['time_start'],1)



tracking_data['acceleration'] = (tracking_data['speed'] - tracking_data['speed_prev'])/(tracking_data['time']-tracking_data['time_prev']).astype(np.float32)



# This normalizes to 0 -> 90 where 0 is fully horizontal (i.e. facing the sideline) and 90 is fully vertical (i.e. facing an end zone)

tracking_data['dir_norm'] = np.where(tracking_data['dir']<=90, tracking_data['dir'],

                                np.where(tracking_data['dir']<=180, 180-tracking_data['dir'],

                                    np.where(tracking_data['dir']<=270, np.abs(180-tracking_data['dir']), 360-tracking_data['dir'])))

# normalize to 100

tracking_data['dir_norm'] = tracking_data['dir_norm'] / 90 * 100



max_speed_time = tracking_data.sort_values(['PlayKey','speed'], ascending=False).drop_duplicates(

    subset=['PlayKey'])[['PlayKey','time_norm','x','y','dir']]



tracking_data['max_speed_time'] = tracking_data[['PlayKey']].merge(max_speed_time, how='left',left_on=['PlayKey'],right_on=['PlayKey'],validate='many_to_one')['time_norm'].values

tracking_data['max_speed_x'] = tracking_data[['PlayKey']].merge(max_speed_time, how='left',left_on=['PlayKey'],right_on=['PlayKey'],validate='many_to_one')['x'].values

tracking_data['max_speed_y'] = tracking_data[['PlayKey']].merge(max_speed_time, how='left',left_on=['PlayKey'],right_on=['PlayKey'],validate='many_to_one')['y'].values

tracking_data['max_speed_dir'] = tracking_data[['PlayKey']].merge(max_speed_time, how='left',left_on=['PlayKey'],right_on=['PlayKey'],validate='many_to_one')['dir'].values



max_speed_times = tracking_data.query('time_norm >= max_speed_time-1 and time_norm <= max_speed_time+1').copy()

max_speed_times['time_norm_max_speed'] = round((max_speed_times['time_norm'] - max_speed_times['max_speed_time'])*10).astype(int)



def rotate(ox, oy, px, py, degrees):

    radians = np.deg2rad(degrees)

    x, y = px-ox, py-oy

    xx = x * np.cos(radians) + y * -np.sin(radians)

    yy = x * np.sin(radians) + y * np.cos(radians)

    return xx, yy

    

max_speed_times['x_norm_max_speed'], max_speed_times['y_norm_max_speed'] = rotate(

                                        max_speed_times['max_speed_x'], max_speed_times['max_speed_y'],

                                        max_speed_times['x'], max_speed_times['y'],

                                        max_speed_times['max_speed_dir'])



tracking_data['x_norm_prev'], tracking_data['y_norm_prev'] = rotate(

                                        tracking_data['x_prev'], tracking_data['y_prev'],

                                        tracking_data['x'], tracking_data['y'],

                                        tracking_data['dir_prev'])



# max horizontal change in direction/movement (relateive to prior movement direction)

# max vertical change in direction/movement

tracking_data['lateral_speed'] = (tracking_data['x_norm_prev']/(tracking_data['time']-tracking_data['time_prev'])).abs()

tracking_data['vertical_speed'] = tracking_data['y_norm_prev']/(tracking_data['time']-tracking_data['time_prev'])



# Speed (yards/sec)

speed_max = tracking_data.groupby(['PlayKey'])['speed'].max().reset_index().rename(columns={'speed':'speed_max'})

speed_avg = tracking_data.groupby(['PlayKey'])['speed'].mean().reset_index().rename(columns={'speed':'speed_avg'})



# Directional changes



#   Horizontal vs. Vertical distance ratio (based on start/end position)

#   0 = all distance traveled horizontal i.e. toward sideline

#   100 = all distance traveled vertical i.e. toward end zone

hv_distance = tracking_data.drop_duplicates(['PlayKey']).copy()

hv_distance['hv_distance'] = np.abs(hv_distance['x_end'] - hv_distance['x_start']) + np.abs(hv_distance['y_end'] - hv_distance['y_start'])

hv_distance['hv_distance'] = np.where(hv_distance['hv_distance']>0, 

                                      (np.abs(hv_distance['y_end'] - hv_distance['y_start']) / hv_distance['hv_distance'])*100, 50)

hv_distance = hv_distance[['PlayKey','hv_distance']]



#   Horizontal vs. Vertical direction

#   0 = all movement horizontal i.e. toward sideline

#   100 = all movement vertical i.e. toward end zone

hv_direction = tracking_data.groupby(['PlayKey'])['dir_norm'].mean().reset_index().rename(columns={'dir_norm':'hv_direction'})



# total dir change

dir_change = tracking_data.groupby(['PlayKey'])['dir'].agg({'max','min'}).reset_index()

dir_change['total_dir_change'] = dir_change['max']-dir_change['min']

dir_change = dir_change[['PlayKey','total_dir_change']]



max_lateral_speed = tracking_data.groupby(['PlayKey'])['lateral_speed'].max().reset_index().rename(columns={'lateral_speed':'max_lateral_speed'})

max_vertical_speed = tracking_data.groupby(['PlayKey'])['vertical_speed'].max().reset_index().rename(columns={'vertical_speed':'max_vertical_speed'})

min_vertical_speed = tracking_data.groupby(['PlayKey'])['vertical_speed'].min().reset_index().rename(columns={'vertical_speed':'min_vertical_speed'})



# Acceleration

acceleration_max = tracking_data.groupby(['PlayKey'])['acceleration'].max().reset_index().rename(columns={'acceleration':'acceleration_max'})

seconds_until_max_speed = tracking_data.sort_values('speed', ascending=False).drop_duplicates('PlayKey')[['PlayKey','time_norm']].rename(columns={'time_norm':'seconds_until_max_speed'})



# Deceleration

deceleration_max = tracking_data.groupby(['PlayKey'])['acceleration'].min().reset_index().rename(columns={'acceleration':'deceleration_max'})



# Distance

total_distance = tracking_data.groupby(['PlayKey'])['dis'].sum().reset_index().rename(columns={'dis':'total_distance'})



# Play length

play_length = tracking_data.groupby(['PlayKey'])['time_norm'].max().reset_index().rename(columns={'time_norm':'play_length'})



cluster_features = pd.concat([max_speed_times.pivot('PlayKey','time_norm_max_speed','x_norm_max_speed').add_prefix('x_'),

                 max_speed_times.pivot('PlayKey','time_norm_max_speed','y_norm_max_speed').add_prefix('y_')], axis=1, sort=False)

for prefix in ['x','y']:

    for mult in [1,-1]:

        for n in range(1,11):

            prev = mult*(n-1)

            cur = round(mult*n,1)

            col_prev = '%s_%s'%(prefix, prev)

            col = '%s_%s'%(prefix, cur)

            cluster_features[col] = winsorize(cluster_features[col].fillna(cluster_features[col_prev]), limits=[0.01, 0.01])

            

# transpose

for n in [10,9,8,7,6,5,4,3,2,1]:

    cluster_features['x_%s'%n] = np.where(cluster_features['x_-1'] < 0, -cluster_features['x_%s'%n], cluster_features['x_%s'%n])

    cluster_features['x_-%s'%n] = np.where(cluster_features['x_-1'] < 0, -cluster_features['x_-%s'%n], cluster_features['x_-%s'%n])

    

X = StandardScaler().fit_transform(cluster_features[[

        'x_-10', 'x_-9', 'x_-8', 'x_-7', 'x_-6', 'x_-5', 'x_-4', 'x_-3', 'x_-2', 'x_-1',

    'y_-10', 'y_-9', 'y_-8', 'y_-7', 'y_-6', 'y_-5', 'y_-4', 'y_-3', 'y_-2', 'y_-1']])



kmeans = MiniBatchKMeans(n_clusters=100, random_state=0).fit(X)

cluster_prev_max_speed = cluster_features.copy()

cluster_prev_max_speed['cluster_num'] = kmeans.labels_

cluster_prev_max_speed['cluster'] = 'cluster_'+cluster_prev_max_speed['cluster_num'].astype(str)

cluster_prev_max_speed = cluster_prev_max_speed.reset_index()[['PlayKey','cluster_num','cluster']]

cluster_prev_max_speed.groupby('cluster')['cluster'].count().describe()



X = StandardScaler().fit_transform(cluster_features[[

    'x_1', 'x_2', 'x_3','x_4', 'x_5', 'x_6', 'x_7', 'x_8', 'x_9', 'x_10',

    'y_1', 'y_2', 'y_3', 'y_4','y_5', 'y_6', 'y_7', 'y_8', 'y_9', 'y_10']])



kmeans = MiniBatchKMeans(n_clusters=100, random_state=0).fit(X)

cluster_post_max_speed = cluster_features.copy()

cluster_post_max_speed['cluster_num'] = kmeans.labels_

cluster_post_max_speed['cluster'] = 'cluster_'+cluster_post_max_speed['cluster_num'].astype(str)

cluster_post_max_speed = cluster_post_max_speed.reset_index()[['PlayKey','cluster_num','cluster']]

cluster_post_max_speed.groupby('cluster')['cluster'].count().describe()



play_list['player_speed_max'] = play_list[['PlayKey']].merge(speed_max, how='left',left_on=['PlayKey'],right_on=['PlayKey'],validate='many_to_one')['speed_max'].values

play_list['player_speed_avg'] = play_list[['PlayKey']].merge(speed_avg, how='left',left_on=['PlayKey'],right_on=['PlayKey'],validate='many_to_one')['speed_avg'].values



play_list['player_hv_distance'] = play_list[['PlayKey']].merge(hv_distance, how='left',left_on=['PlayKey'],right_on=['PlayKey'],validate='many_to_one')['hv_distance'].values

play_list['player_hv_direction'] = play_list[['PlayKey']].merge(hv_direction, how='left',left_on=['PlayKey'],right_on=['PlayKey'],validate='many_to_one')['hv_direction'].values



play_list['player_total_dir_change'] = play_list[['PlayKey']].merge(dir_change, how='left',left_on=['PlayKey'],right_on=['PlayKey'],validate='many_to_one')['total_dir_change'].values



play_list['player_max_lateral_speed'] = play_list[['PlayKey']].merge(max_lateral_speed, how='left',left_on=['PlayKey'],right_on=['PlayKey'],validate='many_to_one')['max_lateral_speed'].values

play_list['player_max_vertical_speed'] = play_list[['PlayKey']].merge(max_vertical_speed, how='left',left_on=['PlayKey'],right_on=['PlayKey'],validate='many_to_one')['max_vertical_speed'].values

play_list['player_min_vertical_speed'] = play_list[['PlayKey']].merge(min_vertical_speed, how='left',left_on=['PlayKey'],right_on=['PlayKey'],validate='many_to_one')['min_vertical_speed'].values



play_list['player_acceleration_max'] = play_list[['PlayKey']].merge(acceleration_max, how='left',left_on=['PlayKey'],right_on=['PlayKey'],validate='many_to_one')['acceleration_max'].values

play_list['player_seconds_until_max_speed'] = play_list[['PlayKey']].merge(seconds_until_max_speed, how='left',left_on=['PlayKey'],right_on=['PlayKey'],validate='many_to_one')['seconds_until_max_speed'].values

play_list['player_deceleration_max'] = -1*play_list[['PlayKey']].merge(deceleration_max, how='left',left_on=['PlayKey'],right_on=['PlayKey'],validate='many_to_one')['deceleration_max'].values



play_list['player_total_distance'] = play_list[['PlayKey']].merge(total_distance, how='left',left_on=['PlayKey'],right_on=['PlayKey'],validate='many_to_one')['total_distance'].values



play_list['play_length'] = play_list[['PlayKey']].merge(play_length, how='left',left_on=['PlayKey'],right_on=['PlayKey'],validate='many_to_one')['play_length'].values



play_list['player_total_distance'] = play_list[['PlayKey']].merge(total_distance, how='left',left_on=['PlayKey'],right_on=['PlayKey'],validate='many_to_one')['total_distance'].values

play_list['fingerprint_post_max_speed'] = play_list[['PlayKey']].merge(cluster_post_max_speed, how='left',left_on=['PlayKey'],right_on=['PlayKey'],validate='many_to_one')['cluster'].values

play_list['fingerprint_prev_max_speed'] = play_list[['PlayKey']].merge(cluster_prev_max_speed, how='left',left_on=['PlayKey'],right_on=['PlayKey'],validate='many_to_one')['cluster'].values



for c in ['player_speed_max','player_speed_avg','player_acceleration_max','player_deceleration_max','player_seconds_until_max_speed',

          'play_length','player_total_distance','player_total_dir_change','player_max_lateral_speed',

         'player_max_vertical_speed','player_min_vertical_speed']:

    play_list[c] = np.where(play_list[c].isnull(), np.nan, winsorize(play_list[c], limits=[0.01, 0.01]))

    

play_list['weather_field_type'] = play_list['FieldType'] + '*' + play_list['weather']



style.use('fivethirtyeight')

sns.set_context('paper')



def plot_plays(plays, groupby='PlayKey', col_wrap=4):

    grid = sns.FacetGrid(plays, col=groupby, col_order=sorted(set(plays[groupby])),

                  col_wrap=col_wrap, height=3, aspect=1.0, sharex=False, sharey=False,despine=False)



    grid.map(plt.axvline, x=0, ls=":", c=".5")

    grid.map(plt.axvline, x=53.3, c=".5")

    grid.map(plt.axvline, x=53.33-.5, ls=":", c=".5")

    grid.map(plt.axvline, x=23, ls=":", c=".5")

    grid.map(plt.axvline, x=53.33-23, ls=":", c=".5")



    grid.map(plt.axhline, y=0, c=".5")

    grid.map(plt.axhline, y=100, c=".5")

    grid.map(plt.axhline, y=50, c=".5")

    grid.map(plt.axhline, y=-10, c=".5")

    grid.map(plt.axhline, y=110, c=".5")



    grid.map(sns.scatterplot,'y','x', 'speed', size_norm=(1,10), size=plays['time_norm'], palette='Reds')

    grid.set(

            yticks=range(-10,120,10), xticks=[],

        yticklabels=['','G','10','20','30','40','50','40','30','20','10','G',''],

            ylabel='', xlabel='',

             ylim=(-11,111), xlim=(-0.5,53.3))



    grid.fig.tight_layout(w_pad=1)

    plt.show()

    

def plot_whisker(metric):

    sns.boxplot(x='injury', y=metric, hue='FieldType', data=plays)

    plt.legend(loc='center left',bbox_to_anchor=(1,0.5))

    plt.show()



def plot_heatmap(metric):

    play_list.groupby(['FieldType','injury'])[metric].mean().to_frame().round(3)

    ax = sns.heatmap(play_list.groupby(['FieldType','injury'])[metric].mean().to_frame().reset_index().pivot('injury','FieldType',metric),

         annot=True, fmt=".2f", cmap="Greens", cbar=True)

    ax.xaxis.tick_top()

    ax.set_title(metric)

    plt.show()

    play_list.hist(metric, by='injury', sharex=True)

    plt.show()

    

def plot_cluster_examples(cluster_plays, col):

    df = tracking_data[tracking_data.PlayKey.isin(cluster_plays.groupby('cluster').max()['PlayKey'])].copy()

    df = df.merge(plays[['PlayKey',col]], how='left',left_on=['PlayKey'],right_on=['PlayKey'],validate='many_to_one')

    df['cluster'] = df[col] + ' | ' + df['PlayKey'].astype(str)

    plot_plays(df, 'cluster', 5)

    

def plot_metric_examples(metric, unit=''):

    max_min = plays.drop_duplicates(subset=['PlayKey']).sort_values(metric)['PlayKey'].values

    samples = [max_min[int(len(max_min)*p)] for p in [.01,.25,.5,.75,.99]]

    df = tracking_data[tracking_data.PlayKey.isin(samples)].copy()

    df = df.merge(plays[['PlayKey',metric]], how='left',left_on=['PlayKey'],right_on=['PlayKey'],validate='many_to_one').sort_values(metric)

    df[metric] = df[metric].rank(method='dense').astype(int).map({1:'p1',2:'p25',3:'p50',4:'p75',5:'p99'}) + " (" + df[metric].apply(lambda m: "%.1f" % (m)) + ' %s)'%unit

    plot_plays(df, metric, 5)

    

def modify_cluster_df(df):

    df['cluster '] = df['cluster'] + '\nmax_speed = ~' + round(df['avg_speed'],1).astype(str) + ' yards/sec'

    # filter to 1000 plays (12 rows per play)

    return df.head(12*1000)



def plot_max_speed(df, color='Purples', ylim=None, title=''):

    grid = sns.FacetGrid(df, col='cluster ', col_order=sorted(set(df['cluster '])),

                  col_wrap=5, height=3, aspect=1.0, sharey=True, sharex=True,despine=True)



    grid.map(plt.axvline, x=0, c=".5")

    grid.map(plt.axhline, y=0, c=".5")



    grid.map(sns.lineplot,'x_norm_max_speed','y_norm_max_speed', 'PlayKey', 

             palette=color, sort=False, ci=None)



    grid.set(#xticks=[-1,0,1], yticks=[-3,-2,-1,0],

             xlim=(-2.5,2.5), 

        ylim=ylim

    )

    grid.fig.tight_layout()

    plt.suptitle(title)

    plt.subplots_adjust(top=.75)

    plt.show()

    

def basic_bar(category, sort=False):

    df = play_list.groupby([category,'injury'])['PlayKey'].count().to_frame().reset_index()

    df['% injuries'] = 100*(np.where(df['injury']=='No Injury', 0, df['PlayKey'] / len(play_list.query('has_injury==1'))))

    df['% plays'] = 100*(df['PlayKey'] / len(play_list))

    df = df.groupby(category)[['% injuries','% plays']].sum()

    if sort:

        df = df.sort_values('% plays', ascending=False)

    ax = df.plot.barh(color=['tab:red','gray'])

    ax.invert_yaxis()

    plt.legend(loc='center left',bbox_to_anchor=(1,0.5))

    plt.xlabel('%')

    plt.show()

    

def plot_ci(cur, title):

    g = sns.boxplot(x=list(cur[['0.975]','[0.025']].values), y=list(cur.index.values), hue=list(cur['Model'].values),

                   linewidth=.5)



    fig = plt.gcf()

    plt.title('%s (Confidence Interval)'%title)

#     plt.xlim(-xlim,xlim)

    plt.show()

    return cur.sort_values(['variable','Model'])



print(len(play_list), len(play_list.drop_duplicates(subset=['PlayerKey'])))

plays = play_list[~play_list.PlayType.isin(['Extra Point','Field Goal','0'])]

print(len(plays), len(plays.drop_duplicates(subset=['PlayerKey'])))

plays = plays.dropna(subset=['weather','PlayType'])

print(len(plays), len(plays.drop_duplicates(subset=['PlayerKey'])))

plays = plays.dropna(subset=['weather','PlayType','player_speed_max'])

print(len(plays), len(plays.drop_duplicates(subset=['PlayerKey'])))



plays.info()
plays['fixed_effects'] = plays.PlayerKey.astype(str)# + '/' + plays.PlayType

# plays['fixed_effects'] = plays.PositionGroup.astype(str) + '/' + plays.PlayType



X_cols = [

    # movement

#     'fingerprint_prev_max_speed','fingerprint_post_max_speed',

    'player_speed_max', #'player_speed_avg',

    'player_hv_distance', 'player_hv_direction', 'player_acceleration_max',

    'player_total_dir_change','player_max_lateral_speed', 'player_min_vertical_speed', #'player_max_vertical_speed',

    'player_seconds_until_max_speed', 'player_deceleration_max', 'player_total_distance',

    # context

    'play_length', 'workload_days_rest', 'workload_num_plays',    

#     'PositionGroup',

    'PlayType',

    # stadium/weather

    'weather',

    # playing surface

    'FieldType',

    'weather_field_type',

#     'field_type_previous_game','field_type_most_frequent',

    'field_type_previous_game_Natural_to_Synthetic','field_type_previous_game_Synthetic_to_Natural',

    'field_type_most_frequent_Natural_to_Synthetic','field_type_most_frequent_Synthetic_to_Natural',

    # FE

    'fixed_effects'

]

X = pd.get_dummies(plays[X_cols], drop_first=False)

X = X.drop(['workload_days_rest_7',#'fingerprint_prev_max_speed_cluster_0','fingerprint_post_max_speed_cluster_0',

            'PlayType_Rush','weather_Indoor','FieldType_Natural','weather_field_type_Natural*Indoor','fixed_effects_26624'],axis=1)

# X = pd.DataFrame(StandardScaler().fit_transform(X), columns=X.columns).astype(np.float32)

for c in ['player_speed_max', #'player_speed_avg',

    'player_hv_distance', 'player_hv_direction', 'player_acceleration_max',

    'player_total_dir_change','player_max_lateral_speed', 'player_min_vertical_speed', #'player_max_vertical_speed',

    'player_seconds_until_max_speed', 'player_deceleration_max', 'player_total_distance',

    'play_length', 

       'workload_num_plays']:

    X[c] = StandardScaler().fit_transform(X[[c]]).reshape(-1, 1)



# model = LinearRegression().fit(X,y)

# results = pd.DataFrame(model.coef_.T, X.columns, columns=['coef']).sort_values('coef',ascending=False)



y = plays['has_injury'].values

%time ols = sm.OLS(y, X).fit()

stats_injury = ols.summary().tables[0]

ols = ols.summary().tables[1].data[1:]

results_injury = pd.DataFrame(ols, columns=['variable','coef','std err','t','P>|t|','[0.025','0.975]'], dtype='float32')

results_injury['Model'] = 'Injury'

results_injury = results_injury.set_index('variable',drop=True).sort_values('coef',ascending=False)



y = np.where(plays['injury']=='Ankle/Foot', 1, 0)

%time ols = sm.OLS(y, X).fit()

stats_ankle = ols.summary().tables[0]

ols = ols.summary().tables[1].data[1:]

results_ankle = pd.DataFrame(ols, columns=['variable','coef','std err','t','P>|t|','[0.025','0.975]'], dtype='float32')

results_ankle['Model'] = 'Ankle'

results_ankle = results_ankle.set_index('variable',drop=True).sort_values('coef',ascending=False)



y = np.where(plays['injury']=='Knee', 1, 0)

%time ols = sm.OLS(y, X).fit()

stats_knee = ols.summary().tables[0]

ols = ols.summary().tables[1].data[1:]

results_knee = pd.DataFrame(ols, columns=['variable','coef','std err','t','P>|t|','[0.025','0.975]'], dtype='float32')

results_knee['Model'] = 'Knee'

results_knee = results_knee.set_index('variable',drop=True).sort_values('coef',ascending=False)



all_results = pd.concat([results_injury, results_ankle, results_knee])



X_cols = [

        # movement

        'fingerprint_prev_max_speed','fingerprint_post_max_speed',

        'player_speed_max', #'player_speed_avg',

        'player_hv_distance', 'player_hv_direction', 'player_acceleration_max',

        'player_total_dir_change','player_max_lateral_speed', 'player_min_vertical_speed', #'player_max_vertical_speed',

        'player_seconds_until_max_speed', 'player_deceleration_max', 'player_total_distance',

        # context

        'play_length', 'workload_days_rest', 'workload_num_plays',    

    #     'PositionGroup',

        'PlayType',

        # stadium/weather

        'weather',

        # playing surface

        'FieldType',

        'weather_field_type',

    #     'field_type_previous_game','field_type_most_frequent',

        'field_type_previous_game_Natural_to_Synthetic','field_type_previous_game_Synthetic_to_Natural',

        'field_type_most_frequent_Natural_to_Synthetic','field_type_most_frequent_Synthetic_to_Natural',

        # FE

        'fixed_effects'

]

X = pd.get_dummies(plays[X_cols], drop_first=False)

X = X.drop(['workload_days_rest_7','fingerprint_prev_max_speed_cluster_0','fingerprint_post_max_speed_cluster_0',

            'PlayType_Rush','weather_Indoor','FieldType_Natural','weather_field_type_Natural*Indoor','fixed_effects_26624'],axis=1)

# X = pd.DataFrame(StandardScaler().fit_transform(X), columns=X.columns).astype(np.float32)

for c in ['player_speed_max', #'player_speed_avg',

    'player_hv_distance', 'player_hv_direction', 'player_acceleration_max',

    'player_total_dir_change','player_max_lateral_speed', 'player_min_vertical_speed', #'player_max_vertical_speed',

    'player_seconds_until_max_speed', 'player_deceleration_max', 'player_total_distance',

    'play_length', 

       'workload_num_plays']:

    X[c] = StandardScaler().fit_transform(X[[c]]).reshape(-1, 1)



# model = LinearRegression().fit(X,y)

# results = pd.DataFrame(model.coef_.T, X.columns, columns=['coef']).sort_values('coef',ascending=False)



y = plays['has_injury'].values

%time ols = sm.OLS(y, X).fit()

stats_injury = ols.summary().tables[0]

ols = ols.summary().tables[1].data[1:]

results_injury_fp = pd.DataFrame(ols, columns=['variable','coef','std err','t','P>|t|','[0.025','0.975]'], dtype='float32')

results_injury_fp['Model'] = 'Injury'

results_injury_fp = results_injury_fp.set_index('variable',drop=True).sort_values('coef',ascending=False)



y = np.where(plays['injury']=='Ankle/Foot', 1, 0)

%time ols = sm.OLS(y, X).fit()

stats_ankle = ols.summary().tables[0]

ols = ols.summary().tables[1].data[1:]

results_ankle_fp = pd.DataFrame(ols, columns=['variable','coef','std err','t','P>|t|','[0.025','0.975]'], dtype='float32')

results_ankle_fp['Model'] = 'Ankle'

results_ankle_fp = results_ankle_fp.set_index('variable',drop=True).sort_values('coef',ascending=False)



y = np.where(plays['injury']=='Knee', 1, 0)

%time ols = sm.OLS(y, X).fit()

stats_knee = ols.summary().tables[0]

ols = ols.summary().tables[1].data[1:]

results_knee_fp = pd.DataFrame(ols, columns=['variable','coef','std err','t','P>|t|','[0.025','0.975]'], dtype='float32')

results_knee_fp['Model'] = 'Knee'

results_knee_fp = results_knee_fp.set_index('variable',drop=True).sort_values('coef',ascending=False)



all_results_fp = pd.concat([results_injury_fp, results_ankle_fp, results_knee_fp])
X_cols = [

    # movement

#     'fingerprint_prev_max_speed','fingerprint_post_max_speed',

#     'player_speed_max', #'player_speed_avg',

#     'player_hv_distance', 'player_hv_direction', 'player_acceleration_max',

#     'player_total_dir_change','player_max_lateral_speed', 'player_min_vertical_speed', #'player_max_vertical_speed',

#     'player_seconds_until_max_speed', 'player_deceleration_max', 'player_total_distance',

    # context

    'play_length', 'workload_days_rest', 'workload_num_plays',    

#     'PositionGroup',

    'PlayType',

    # stadium/weather

    'weather',

    # playing surface

    'FieldType',

    'weather_field_type',

#     'field_type_previous_game','field_type_most_frequent',

    'field_type_previous_game_Natural_to_Synthetic','field_type_previous_game_Synthetic_to_Natural',

    'field_type_most_frequent_Natural_to_Synthetic','field_type_most_frequent_Synthetic_to_Natural',

    # FE

    'fixed_effects'

]

X = pd.get_dummies(plays[X_cols], drop_first=False)

X = X.drop(['workload_days_rest_7',#'fingerprint_prev_max_speed_cluster_0','fingerprint_post_max_speed_cluster_0',

            'PlayType_Rush','weather_Indoor','FieldType_Natural','weather_field_type_Natural*Indoor','fixed_effects_26624'],axis=1)



def run_ols(y, type):

    ols = sm.OLS(y, X).fit()

    ols = ols.summary().tables[1].data[1:]

    results_cur = pd.DataFrame(ols, columns=['variable','coef','std err','t','P>|t|','[0.025','0.975]'], dtype='float32')

    results_cur['Model'] = type

    results_cur = results_cur.set_index('variable',drop=True).sort_values('coef',ascending=False)

    results_cur = results_cur[~results_cur.index.str.startswith('fixed_effects')]

    return results_cur



results_speed = run_ols(plays['player_speed_max'], 'Speed Max')

results_lateral = run_ols(plays['player_max_lateral_speed'], 'Lateral Speed Max')

results_acc = run_ols(plays['player_acceleration_max'], 'Acceleration Max')
tracking_injured = tracking_data[tracking_data.PlayKey.isin(injuries.PlayKey)].copy()

tracking_injured = tracking_injured.merge(play_list[['PlayKey','BodyPart','PlayType','PositionGroup']], how='left',left_on=['PlayKey'],right_on=['PlayKey'],validate='many_to_one')

tracking_injured['PlayType'] = tracking_injured.PlayType.str.replace(' Not Returned| Returned','')

tracking_injured['BodyPart'] = tracking_injured.BodyPart.map({'Ankle':'Ankle/Foot','Foot':'Ankle/Foot','Knee':'Knee'})



tracking_injured['Injury'] = tracking_injured['PlayType'] + ' | ' + tracking_injured['BodyPart']

plot_plays(tracking_injured, 'Injury')
style.use('fivethirtyeight')

sns.set_context('paper')



plot_metric_examples('player_speed_max', 'yd/s')

plot_metric_examples('player_speed_avg', 'yd/s')

plot_metric_examples('player_max_lateral_speed', 'yd/s')

plot_metric_examples('player_max_vertical_speed', 'yd/s')

plot_metric_examples('player_min_vertical_speed', 'yd/s')
plot_metric_examples('player_hv_distance', '')

plot_metric_examples('player_hv_direction', '')

plot_metric_examples('player_total_dir_change', '')
plot_metric_examples('player_acceleration_max', 'yd/s²')

plot_metric_examples('player_deceleration_max', 'yd/s²')
plot_metric_examples('player_total_distance', 'yards')
plot_metric_examples('player_seconds_until_max_speed', '')

plot_metric_examples('play_length', 'secs')
plot_plays(tracking_data[tracking_data.PlayKey.isin(['30068-24-22'])], 'PlayKey')



sns.scatterplot(data=max_speed_times[max_speed_times.PlayKey.isin(['30068-24-22'])], x = 'y', y = 'x', hue='PlayKey')

plt.ylim(0,120)

plt.xlim(0,53.3)

plt.title('Actual XY second before/after max speed')

plt.show()



sns.scatterplot(data=max_speed_times[max_speed_times.PlayKey.isin(['30068-24-22'])], x = 'x_norm_max_speed', y = 'y_norm_max_speed', hue='PlayKey')

plt.xlim(-2,2)

plt.title('Normalized XY second before/after max speed')

plt.show()
basic_bar('weather', sort=True)

basic_bar('workload_days_rest')

basic_bar('FieldType')

basic_bar('PositionGroup', sort=True)

basic_bar('PlayType', sort=True)

basic_bar('field_type_previous_game')

basic_bar('field_type_most_frequent')
plot_whisker('player_speed_max')

plot_whisker('player_speed_avg')

plot_whisker('player_hv_distance')

plot_whisker('player_hv_direction')

plot_whisker('player_acceleration_max')

plot_whisker('player_seconds_until_max_speed')

plot_whisker('player_deceleration_max')

plot_whisker('player_total_distance')

plot_whisker('play_length')

plot_whisker('player_total_dir_change')

plot_whisker('player_max_lateral_speed')

plot_whisker('player_max_vertical_speed')

plot_whisker('player_min_vertical_speed')
X = plays[[

    # movement

#         'fingerprint_prev_max_speed','fingerprint_post_max_speed',

        'player_speed_max', #'player_speed_avg',

        'player_hv_distance', 'player_hv_direction', 'player_acceleration_max',

        'player_total_dir_change','player_max_lateral_speed', 'player_min_vertical_speed', #'player_max_vertical_speed',

        'player_seconds_until_max_speed', 'player_deceleration_max', 'player_total_distance',

        # context

        'play_length', 'workload_days_rest', 'workload_num_plays',    

    #     'PositionGroup',

    #     'PlayType',

        # stadium/weather

        'weather',

        # playing surface

        'FieldType',

        'weather_field_type',

    #     'field_type_previous_game','field_type_most_frequent',

        'field_type_previous_game_Natural_to_Synthetic','field_type_previous_game_Synthetic_to_Natural',

        'field_type_most_frequent_Natural_to_Synthetic','field_type_most_frequent_Synthetic_to_Natural',

        # FE

#         'fixed_effects'

]].info()



X = plays[['player_speed_max', #'player_speed_avg',

        'player_hv_distance', 'player_hv_direction', 'player_acceleration_max',

        'player_total_dir_change','player_max_lateral_speed', 'player_min_vertical_speed', #'player_max_vertical_speed',

        'player_seconds_until_max_speed', 'player_deceleration_max', 'player_total_distance',]]

sns.heatmap(X.corr(), annot=True, fmt=".2f")

plt.show()
all_filtered = all_results[~((all_results.index.str.startswith('fixed_effects'))|(

    all_results.index.str.startswith('fingerprint')))].sort_values('0.975]',ascending=False).round(6)



plot_ci(all_filtered[all_filtered['P>|t|']<=.05], 'Statistically Significant Variables')
results_non_injured = pd.concat([results_speed, results_lateral, results_acc])

results_non_injured = results_non_injured[results_non_injured.index.str.startswith('Field')]

plot_ci(results_non_injured[results_non_injured['P>|t|']<=.05], 'Statistically Significant Variables')
prev_max_speed_clusters = max_speed_times[max_speed_times.PlayKey.isin(max_speed_times.query('time_norm_max_speed==-1 and x_norm_max_speed >= 0').PlayKey)].merge(cluster_prev_max_speed, 

                                                how='left',left_on=['PlayKey'],right_on=['PlayKey'],validate='many_to_one').copy()

post_max_speed_clusters = max_speed_times[max_speed_times.PlayKey.isin(max_speed_times.query('time_norm_max_speed==-1 and x_norm_max_speed >= 0').PlayKey)].merge(cluster_post_max_speed, 

                                                how='left',left_on=['PlayKey'],right_on=['PlayKey'],validate='many_to_one').copy()



avg_speed_cluster_prev = plays.groupby('fingerprint_prev_max_speed')['player_speed_max'].mean().reset_index()[['fingerprint_prev_max_speed','player_speed_max']]

avg_speed_cluster_post = plays.groupby('fingerprint_post_max_speed')['player_speed_max'].mean().reset_index()[['fingerprint_post_max_speed','player_speed_max']]



prev_max_speed_clusters['avg_speed'] = prev_max_speed_clusters[['cluster']].merge(avg_speed_cluster_prev, how='left',left_on=['cluster'],right_on=['fingerprint_prev_max_speed'],validate='many_to_one')['player_speed_max'].values

post_max_speed_clusters['avg_speed'] = post_max_speed_clusters[['cluster']].merge(avg_speed_cluster_post, how='left',left_on=['cluster'],right_on=['fingerprint_post_max_speed'],validate='many_to_one')['player_speed_max'].values



top_n = 5

high_risk_pre = prev_max_speed_clusters[prev_max_speed_clusters['cluster'].isin(

    results_injury_fp[results_injury_fp.index.str.startswith('fingerprint_prev')].head(top_n).index.str.replace('.*cluster','cluster'))].query('time_norm_max_speed<=1').copy()

high_risk_post = post_max_speed_clusters[post_max_speed_clusters['cluster'].isin(

    results_injury_fp[results_injury_fp.index.str.startswith('fingerprint_post')].head(top_n).index.str.replace('.*cluster','cluster'))].query('time_norm_max_speed>=-1').copy()



low_risk_pre = prev_max_speed_clusters[prev_max_speed_clusters['cluster'].isin(

    results_injury_fp[results_injury_fp.index.str.startswith('fingerprint_prev')].tail(top_n).index.str.replace('.*cluster','cluster'))].query('time_norm_max_speed<=1').copy()

low_risk_post = post_max_speed_clusters[post_max_speed_clusters['cluster'].isin(

    results_injury_fp[results_injury_fp.index.str.startswith('fingerprint_post')].tail(top_n).index.str.replace('.*cluster','cluster'))].query('time_norm_max_speed>=-1').copy()



high_risk_pre = modify_cluster_df(high_risk_pre)

high_risk_post = modify_cluster_df(high_risk_post)

low_risk_pre = modify_cluster_df(low_risk_pre)

low_risk_post = modify_cluster_df(low_risk_post)



plot_ci(results_injury_fp[(~results_injury_fp.index.str.startswith('fixed'))&(results_injury_fp['P>|t|']<=.05)], 'Statistically Significant Variables')
plot_max_speed(high_risk_pre, color='Oranges', ylim=(-10,1), title='High Risk Movement (pre-max speed)')

plot_cluster_examples(high_risk_pre, 'fingerprint_prev_max_speed')



plot_max_speed(low_risk_pre, color='Greens', ylim=(-10,1), title='Low Risk Movement (pre-max speed)')

plot_cluster_examples(low_risk_pre, 'fingerprint_prev_max_speed')



plot_max_speed(high_risk_post, color='Oranges', ylim=(-1, 10), title='High Risk Movement (post-max speed)')

plot_cluster_examples(high_risk_post, 'fingerprint_post_max_speed')



plot_max_speed(low_risk_post, color='Greens', ylim=(-1, 10), title='Low Risk Movement (post-max speed)')

plot_cluster_examples(low_risk_post, 'fingerprint_post_max_speed')
basic_bar('weather', sort=True)

basic_bar('workload_days_rest')

basic_bar('FieldType')

basic_bar('PositionGroup', sort=True)

basic_bar('PlayType', sort=True)

basic_bar('field_type_previous_game')

basic_bar('field_type_most_frequent')
tracking_injured['Injury'] = tracking_injured['PositionGroup'] + ' | ' + tracking_injured['BodyPart']

plot_plays(tracking_injured, 'Injury')
print(stats_injury)

print(stats_knee)

print(stats_ankle)
plot_ci(all_filtered[all_filtered.index.str.startswith('player_')], 'Player Movement')
plot_ci(all_filtered[all_filtered.index.str.startswith('FieldType')], 'Playing Surface')
plot_ci(all_filtered[all_filtered.index.str.startswith('field_type')], 'Field Type')
plot_ci(all_filtered[all_filtered.index.str.startswith('workload')], 'Workload')
plot_ci(all_filtered[all_filtered.index.str.startswith('weather')], 'Weather')
plot_ci(all_filtered[(all_filtered.index.str.startswith('play_'))|(all_filtered.index.str.startswith('PlayType'))], 'Play')