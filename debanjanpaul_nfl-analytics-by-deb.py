import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from matplotlib import pyplot as plt

from scipy.stats import chi2_contingency

from scipy.stats import chi2

#from sklearn.preprocessing import scale



%matplotlib inline

sns.set_style("darkgrid")

#pd.set_option('display.max_rows', 1000)



# Set your own project id here

PROJECT_ID = 'debanjan-bg'

from google.cloud import bigquery

bigquery_client = bigquery.Client(project=PROJECT_ID)
### Upload a csv

"""filename = r'/kaggle/input/nfl-playing-surface-analytics/PlayerTrackData.csv'

dataset_ref = bigquery_client.dataset('debanjan_bq_nfl_datasets')

table_ref = dataset_ref.table('PlayerTrackData')

job_config = bigquery.LoadJobConfig()

job_config.source_format = bigquery.SourceFormat.CSV

job_config.autodetect = True



with open(filename, 'rb') as src_file:

    job = bigquery_client.load_table_from_file(src_file, table_ref, job_config=job_config)

    

job.result()

"""



### Upload a dataframe

"""

dataset_ref = bigquery_client.dataset('debanjan_bq_nfl_datasets')

table_ref = dataset_ref.table('NotInjuredOffenseWideReceiver')

job_config = bigquery.LoadJobConfig()

job_config.autodetect = True

job = bigquery_client.load_table_from_dataframe(not_injured_offense_WideReceiver_df, table_ref, job_config=job_config)

job.result()

"""
def CalculateVariousFeatures(df):

    df.sort_values(by='time', inplace = True)

    

    ### Calculate Change in Instanteneous velocity

    df['estimated_change_in_instanteneous_velocity'] = df['s'].diff(+1)

    df['time_delta'] = df['time'].diff(+1)

    df['estimated_acceleration'] = df['estimated_change_in_instanteneous_velocity'] / df['time_delta']

    df['estimated_acceleration_x_component'] = df['estimated_acceleration']*np.sin(np.radians(df['dir']))

    df['estimated_acceleration_y_component'] = df['estimated_acceleration']*np.cos(np.radians(df['dir']))

    

    ### Calculate deviation in dir and o

    df['twist_dir_minus_o'] = df['dir'] - df['o']

    df['rate_of_change_in_twist'] = df['twist_dir_minus_o'] / df['time_delta']

    

    ### Calculate rate of change in direction (player motion)

    df['change_in_directional_motion'] = df['dir'].diff(+1)

    df['rate_of_change_in_directional_motion'] = df['change_in_directional_motion'] / df['time_delta']

    

    ### Calculate rate of change in orientation (player facing)

    df['change_in_orientation'] = df['o'].diff(+1)

    df['rate_of_change_in_orientation'] = df['change_in_orientation'] / df['time_delta']

    

    return df
def DeriveFeatures(PlayerTrackData):

    players = PlayerTrackData['PlayerKey'].unique()

    tracking_data_subset=pd.DataFrame()

    for p in players:

        temp_df = PlayerTrackData[PlayerTrackData['PlayerKey'] == p].groupby(['PlayKey']).apply(CalculateVariousFeatures)

        tracking_data_subset = tracking_data_subset.append(temp_df, ignore_index = True)

    return tracking_data_subset
def GetPlayerTrackData(fields, table_name, playerlist=None):



    

    if playerlist:

        PLAYERLIST = tuple(playerlist)

        query = f""" SELECT {fields}  

             FROM `{PROJECT_ID}.debanjan_bq_nfl_datasets.{table_name}`

             WHERE PlayerKey IN {PLAYERLIST}

         """

    else:

        query  = f"""SELECT {fields}

             FROM `{PROJECT_ID}.debanjan_bq_nfl_datasets.{table_name}`

          """





    query_job = bigquery_client.query(query)

    return query_job.to_dataframe()
def TestChiSquare(dist1, dist2):



    # contingency table

    table = [dist1, dist2]

    #print(table)

    stat, p, dof, expected = chi2_contingency(table)

    #print('dof=%d' % dof)

    #print(expected)



    alpha = 0.05

    test_result = {'significance' : alpha,'p' : p, 'dof' : dof}



    if p <= alpha:

        test_result['outcome'] = 'Reject H0'

        return test_result

    else:

        test_result['outcome'] = 'Fail to reject H0'

        return test_result
def CalculateCOV(PlayerTrackData):

    #players = PlayerTrackData['PlayerKey'].unique()

    #output = pd.DataFrame()

    output = PlayerTrackData.groupby(['PlayerKey','PlayKey']).apply(lambda x: np.std(x) / np.mean(x))

    #output.reset_index(drop = False, inplace = True)

    return output        
injury_data = pd.read_csv(r'/kaggle/input/nfl-playing-surface-analytics/InjuryRecord.csv')

playlist_data = pd.read_csv(r'/kaggle/input/nfl-playing-surface-analytics/PlayList.csv')

#tracking_data = pd.read_csv(r'/kaggle/input/nfl-playing-surface-analytics/PlayerTrackData.csv')
print('Sample has got : ' + str(len(injury_data['PlayerKey'].unique())) + ' players injury records')

print('Sample has got : ' +  str(injury_data.shape[0]) + ' injury records')
temp_series = injury_data['PlayerKey'].value_counts() > 1

plyers_with_2injuries = temp_series.index[temp_series == True]

injury_data[injury_data['PlayerKey'].isin(plyers_with_2injuries)].sort_values(by='PlayerKey')
print('Roaster Positions : ', playlist_data['RosterPosition'].unique())
offensive_position = ['Quarterback', 'Wide Receiver', 'Running Back', 'Tight End', 'Offensive Lineman', 'Kicker']

defensive_position = ['Linebacker','Defensive Lineman', 'Safety', 'Cornerback']

playlist_data.loc[playlist_data['RosterPosition'].isin(offensive_position), 'RosterPositionCategory'] = 'offense'

playlist_data.loc[playlist_data['RosterPosition'].isin(defensive_position), 'RosterPositionCategory'] = 'defense'
players_by_roasterpositioncategory = playlist_data[['PlayerKey', 'RosterPositionCategory']].drop_duplicates()

players_by_roasterpositioncategory.reset_index(drop=True)
print('Breakup of offensive/defensive players in injured group :')

pd.DataFrame(players_by_roasterpositioncategory[players_by_roasterpositioncategory['PlayerKey'].isin(injury_data['PlayerKey'].unique())]\

             ['RosterPositionCategory'].value_counts(dropna=False))
players_by_roasterpositioncategory = playlist_data[['PlayerKey', 'RosterPosition', 'RosterPositionCategory']].drop_duplicates()

print('Breakup of offensive/defensive players within not-injured group :')

pd.DataFrame(players_by_roasterpositioncategory[~players_by_roasterpositioncategory['PlayerKey'].isin(injury_data['PlayerKey'].unique())]\

             ['RosterPositionCategory'].value_counts(dropna=False))
injured_offense = list(players_by_roasterpositioncategory[(players_by_roasterpositioncategory['PlayerKey'].isin(injury_data['PlayerKey'].unique()))

                                  & (players_by_roasterpositioncategory['RosterPositionCategory'] == 'offense')]['PlayerKey'])



injured_defense = list(players_by_roasterpositioncategory[(players_by_roasterpositioncategory['PlayerKey'].isin(injury_data['PlayerKey'].unique()))

                                  & (players_by_roasterpositioncategory['RosterPositionCategory'] == 'defense')]['PlayerKey'])



not_injured_offense = list(players_by_roasterpositioncategory[(~players_by_roasterpositioncategory['PlayerKey'].isin(injury_data['PlayerKey'].unique()))

                                  & (players_by_roasterpositioncategory['RosterPositionCategory'] == 'offense')]['PlayerKey'])



not_injured_defense = list(players_by_roasterpositioncategory[(~players_by_roasterpositioncategory['PlayerKey'].isin(injury_data['PlayerKey'].unique()))

                                  & (players_by_roasterpositioncategory['RosterPositionCategory'] == 'defense')]['PlayerKey'])
f,axes=plt.subplots(1,2,figsize=(18,5),sharey=True)



sns.catplot(x="BodyPart", kind="count", palette="ch:.25", hue = 'Surface',

            data=injury_data[injury_data['PlayerKey'].isin(injured_offense)].sort_values(by='BodyPart'), ax = axes[0])

axes[0].set_title('Number of BodyPart Injuries : Injured Offense')



sns.catplot(x="BodyPart", kind="count", palette="ch:.25", hue = 'Surface',

            data=injury_data[injury_data['PlayerKey'].isin(injured_defense)].sort_values(by='BodyPart'), ax = axes[1])

axes[1].set_title('Number of BodyPart Injuries : Injured Defense')

plt.close(2)

plt.close(3)
injury_data[injury_data['PlayerKey'].isin(injured_offense)]['BodyPart'].value_counts()
injury_data[injury_data['PlayerKey'].isin(injured_defense)]['BodyPart'].value_counts()
print(TestChiSquare(np.array([27, 26, 4, 2, 1]), np.array([21, 16, 5, 3, 0])))
#offensive_position = ['Quarterback', 'Wide Receiver', 'Running Back', 'Tight End', 'Offensive Lineman', 'Kicker']



injured_offense_Quarterback = list(players_by_roasterpositioncategory[(players_by_roasterpositioncategory['PlayerKey'].isin(injury_data['PlayerKey'].unique()))

                                  & (players_by_roasterpositioncategory['RosterPositionCategory'] == 'offense')

                                                         & (players_by_roasterpositioncategory['RosterPosition'] == 'Quarterback')]['PlayerKey'])



injured_offense_WideReceiver = list(players_by_roasterpositioncategory[(players_by_roasterpositioncategory['PlayerKey'].isin(injury_data['PlayerKey'].unique()))

                                  & (players_by_roasterpositioncategory['RosterPositionCategory'] == 'offense')

                                                         & (players_by_roasterpositioncategory['RosterPosition'] == 'Wide Receiver')]['PlayerKey'])



injured_offense_RunningBack = list(players_by_roasterpositioncategory[(players_by_roasterpositioncategory['PlayerKey'].isin(injury_data['PlayerKey'].unique()))

                                  & (players_by_roasterpositioncategory['RosterPositionCategory'] == 'offense')

                                                         & (players_by_roasterpositioncategory['RosterPosition'] == 'Running Back')]['PlayerKey'])



injured_offense_TightEnd = list(players_by_roasterpositioncategory[(players_by_roasterpositioncategory['PlayerKey'].isin(injury_data['PlayerKey'].unique()))

                                  & (players_by_roasterpositioncategory['RosterPositionCategory'] == 'offense')

                                                         & (players_by_roasterpositioncategory['RosterPosition'] == 'Tight End')]['PlayerKey'])



injured_offense_OffensiveLineman = list(players_by_roasterpositioncategory[(players_by_roasterpositioncategory['PlayerKey'].isin(injury_data['PlayerKey'].unique()))

                                  & (players_by_roasterpositioncategory['RosterPositionCategory'] == 'offense')

                                                         & (players_by_roasterpositioncategory['RosterPosition'] == 'Offensive Lineman')]['PlayerKey'])



injured_offense_Kicker = list(players_by_roasterpositioncategory[(players_by_roasterpositioncategory['PlayerKey'].isin(injury_data['PlayerKey'].unique()))

                                  & (players_by_roasterpositioncategory['RosterPositionCategory'] == 'offense')

                                                         & (players_by_roasterpositioncategory['RosterPosition'] == 'Kicker')]['PlayerKey'])

print('injured_offense_Quarterback : ', len(injured_offense_Quarterback))

print('injured_offense_WideReceiver : ', len(injured_offense_WideReceiver))

print('injured_offense_RunningBack : ', len(injured_offense_RunningBack))

print('injured_offense_TightEnd : ', len(injured_offense_TightEnd))

print('injured_offense_OffensiveLineman : ',  len(injured_offense_OffensiveLineman))

print('injured_offense_Kicker : ', len(injured_offense_Kicker))
f,axes=plt.subplots(2,2,figsize=(20,10),sharey=True)



sns.catplot(x="BodyPart", kind="count", palette="ch:.25", hue = 'Surface',

            data=injury_data[injury_data['PlayerKey'].isin(injured_offense_WideReceiver)].sort_values(by='BodyPart'), ax = axes[0,0])

axes[0,0].set_title('Number of BodyPart Injuries : Injured Offense - Wide Receiver')



sns.catplot(x="BodyPart", kind="count", palette="ch:.25", hue = 'Surface',

            data=injury_data[injury_data['PlayerKey'].isin(injured_offense_RunningBack)].sort_values(by='BodyPart'), ax = axes[0,1])

axes[0,1].set_title('Number of BodyPart Injuries : Injured Offense - Running Back')



sns.catplot(x="BodyPart", kind="count", palette="ch:.25", hue = 'Surface',

            data=injury_data[injury_data['PlayerKey'].isin(injured_offense_TightEnd)].sort_values(by='BodyPart'), ax = axes[1,0])

axes[1,0].set_title('Number of BodyPart Injuries : Injured Offense - Tight End')



sns.catplot(x="BodyPart", kind="count", palette="ch:.25", hue = 'Surface',

            data=injury_data[injury_data['PlayerKey'].isin(injured_offense_OffensiveLineman)].sort_values(by='BodyPart'), ax = axes[1,1])

axes[1,1].set_title('Number of BodyPart Injuries : Injured Offense - Offensive Lineman')



plt.close(2)

plt.close(3)

plt.close(4)

plt.close(5)
#defensive_position = ['Linebacker','Defensive Lineman', 'Safety', 'Cornerback']



injured_defense_Linebacker = list(players_by_roasterpositioncategory[(players_by_roasterpositioncategory['PlayerKey'].isin(injury_data['PlayerKey'].unique()))

                                  & (players_by_roasterpositioncategory['RosterPositionCategory'] == 'defense')

                                  & (players_by_roasterpositioncategory['RosterPosition'] == 'Linebacker')]['PlayerKey'])



injured_defense_DefensiveLineman = list(players_by_roasterpositioncategory[(players_by_roasterpositioncategory['PlayerKey'].isin(injury_data['PlayerKey'].unique()))

                                  & (players_by_roasterpositioncategory['RosterPositionCategory'] == 'defense')

                                  & (players_by_roasterpositioncategory['RosterPosition'] == 'Defensive Lineman')]['PlayerKey'])



injured_defense_Safety = list(players_by_roasterpositioncategory[(players_by_roasterpositioncategory['PlayerKey'].isin(injury_data['PlayerKey'].unique()))

                                  & (players_by_roasterpositioncategory['RosterPositionCategory'] == 'defense')

                                  & (players_by_roasterpositioncategory['RosterPosition'] == 'Safety')]['PlayerKey'])



injured_defense_Cornerback = list(players_by_roasterpositioncategory[(players_by_roasterpositioncategory['PlayerKey'].isin(injury_data['PlayerKey'].unique()))

                                  & (players_by_roasterpositioncategory['RosterPositionCategory'] == 'defense')

                                  & (players_by_roasterpositioncategory['RosterPosition'] == 'Cornerback')]['PlayerKey'])
print('injured_defense_Linebacker : ',  len(injured_defense_Linebacker))

print('injured_defense_DefensiveLineman : ', len(injured_defense_DefensiveLineman))

print('injured_defense_Safety : ',  len(injured_defense_Safety))

print('injured_defense_Cornerback : ',     len(injured_defense_Cornerback))
f,axes=plt.subplots(2,2,figsize=(20,10),sharey=True)



sns.catplot(x="BodyPart", kind="count", palette="ch:.25", hue = 'Surface',

            data=injury_data[injury_data['PlayerKey'].isin(injured_defense_Linebacker)].sort_values(by='BodyPart'), ax = axes[0,0])

axes[0,0].set_title('Number of BodyPart Injuries : Injured Defense - Linebacker')



sns.catplot(x="BodyPart", kind="count", palette="ch:.25", hue = 'Surface',

            data=injury_data[injury_data['PlayerKey'].isin(injured_defense_DefensiveLineman)].sort_values(by='BodyPart'), ax = axes[0,1])

axes[0,1].set_title('Number of BodyPart Injuries : Injured Defense - DefensiveLineman')



sns.catplot(x="BodyPart", kind="count", palette="ch:.25", hue = 'Surface',

            data=injury_data[injury_data['PlayerKey'].isin(injured_defense_Safety )].sort_values(by='BodyPart'), ax = axes[1,0])

axes[1,0].set_title('Number of BodyPart Injuries : Injured Defense - Safety')



sns.catplot(x="BodyPart", kind="count", palette="ch:.25", hue = 'Surface',

            data=injury_data[injury_data['PlayerKey'].isin(injured_defense_Cornerback)].sort_values(by='BodyPart'), ax = axes[1,1])

axes[1,1].set_title('Number of BodyPart Injuries : Injured Defense - Cornerback')



plt.close(2)

plt.close(3)

plt.close(4)

plt.close(5)
# offensive_position = ['Quarterback', 'Wide Receiver', 'Running Back', 'Tight End', 'Offensive Lineman', 'Kicker']





not_injured_offense_Quarterback = list(players_by_roasterpositioncategory[(~players_by_roasterpositioncategory['PlayerKey'].isin(injury_data['PlayerKey'].unique()))

                                  & (players_by_roasterpositioncategory['RosterPositionCategory'] == 'offense')

                                                         & (players_by_roasterpositioncategory['RosterPosition'] == 'Quarterback')]['PlayerKey'])



not_injured_offense_WideReceiver = list(players_by_roasterpositioncategory[(~players_by_roasterpositioncategory['PlayerKey'].isin(injury_data['PlayerKey'].unique()))

                                  & (players_by_roasterpositioncategory['RosterPositionCategory'] == 'offense')

                                                         & (players_by_roasterpositioncategory['RosterPosition'] == 'Wide Receiver')]['PlayerKey'])



not_injured_offense_RunningBack = list(players_by_roasterpositioncategory[(~players_by_roasterpositioncategory['PlayerKey'].isin(injury_data['PlayerKey'].unique()))

                                  & (players_by_roasterpositioncategory['RosterPositionCategory'] == 'offense')

                                                         & (players_by_roasterpositioncategory['RosterPosition'] == 'Running Back')]['PlayerKey'])



not_injured_offense_TightEnd = list(players_by_roasterpositioncategory[(~players_by_roasterpositioncategory['PlayerKey'].isin(injury_data['PlayerKey'].unique()))

                                  & (players_by_roasterpositioncategory['RosterPositionCategory'] == 'offense')

                                                         & (players_by_roasterpositioncategory['RosterPosition'] == 'Tight End')]['PlayerKey'])



not_injured_offense_OffensiveLineman = list(players_by_roasterpositioncategory[(~players_by_roasterpositioncategory['PlayerKey'].isin(injury_data['PlayerKey'].unique()))

                                  & (players_by_roasterpositioncategory['RosterPositionCategory'] == 'offense')

                                                         & (players_by_roasterpositioncategory['RosterPosition'] == 'Offensive Lineman')]['PlayerKey'])



not_injured_offense_Kicker = list(players_by_roasterpositioncategory[(~players_by_roasterpositioncategory['PlayerKey'].isin(injury_data['PlayerKey'].unique()))

                                  & (players_by_roasterpositioncategory['RosterPositionCategory'] == 'offense')

                                                         & (players_by_roasterpositioncategory['RosterPosition'] == 'Kicker')]['PlayerKey'])
print('not_injured_offense_Quarterback : ', len(not_injured_offense_Quarterback))

print('not_injured_offense_WideReceiver : ', len(not_injured_offense_WideReceiver))

print('not_injured_offense_RunningBack : ', len(not_injured_offense_RunningBack))

print('not_injured_offense_TightEnd : ', len(not_injured_offense_TightEnd))

print('not_injured_offense_OffensiveLineman : ',  len(not_injured_offense_OffensiveLineman))

print('not_injured_offense_Kicker : ', len(not_injured_offense_Kicker))
#defensive_position = ['Linebacker','Defensive Lineman', 'Safety', 'Cornerback']



not_injured_defense_Linebacker = list(players_by_roasterpositioncategory[(~players_by_roasterpositioncategory['PlayerKey'].isin(injury_data['PlayerKey'].unique()))

                                  & (players_by_roasterpositioncategory['RosterPositionCategory'] == 'defense')

                                  & (players_by_roasterpositioncategory['RosterPosition'] == 'Linebacker')]['PlayerKey'])



not_injured_defense_DefensiveLineman = list(players_by_roasterpositioncategory[(~players_by_roasterpositioncategory['PlayerKey'].isin(injury_data['PlayerKey'].unique()))

                                  & (players_by_roasterpositioncategory['RosterPositionCategory'] == 'defense')

                                  & (players_by_roasterpositioncategory['RosterPosition'] == 'Defensive Lineman')]['PlayerKey'])



not_injured_defense_Safety = list(players_by_roasterpositioncategory[(~players_by_roasterpositioncategory['PlayerKey'].isin(injury_data['PlayerKey'].unique()))

                                  & (players_by_roasterpositioncategory['RosterPositionCategory'] == 'defense')

                                  & (players_by_roasterpositioncategory['RosterPosition'] == 'Safety')]['PlayerKey'])



not_injured_defense_Cornerback = list(players_by_roasterpositioncategory[(~players_by_roasterpositioncategory['PlayerKey'].isin(injury_data['PlayerKey'].unique()))

                                  & (players_by_roasterpositioncategory['RosterPositionCategory'] == 'defense')

                                  & (players_by_roasterpositioncategory['RosterPosition'] == 'Cornerback')]['PlayerKey'])
print('not_injured_defense_Linebacker : ', len(not_injured_defense_Linebacker))

print('not_injured_defense_DefensiveLineman : ', len(not_injured_defense_DefensiveLineman))

print('not_injured_defense_Safety : ', len(not_injured_defense_Safety))

print('not_injured_defense_Cornerback : ', len(not_injured_defense_Cornerback))
groups = {

                'player_group' : {

                                    'injured_offense_Quarterback' : injured_offense_Quarterback,

                                    'injured_offense_WideReceiver' : injured_offense_WideReceiver,

                                    'injured_offense_RunningBack' : injured_offense_RunningBack,

                                    'injured_offense_TightEnd' : injured_offense_TightEnd,

                                    'injured_offense_OffensiveLineman' : injured_offense_OffensiveLineman,

                                    'injured_offense_Kicker' : injured_offense_Kicker,

                    

                                    'injured_defense_Linebacker' : injured_defense_Linebacker,

                                    'injured_defense_DefensiveLineman' : injured_defense_DefensiveLineman,

                                    'injured_defense_Safety' : injured_defense_Safety,

                                    'injured_defense_Cornerback' : injured_defense_Cornerback,

                    

                                    'not_injured_offense_Quarterback' : not_injured_offense_Quarterback,

                                    'not_injured_offense_WideReceiver' : not_injured_offense_WideReceiver,

                                    'not_injured_offense_RunningBack' : not_injured_offense_RunningBack,

                                    'not_injured_offense_TightEnd' :  not_injured_offense_TightEnd,

                                    'not_injured_offense_OffensiveLineman' : not_injured_offense_OffensiveLineman,

                                    'not_injured_offense_Kicker' : not_injured_offense_Kicker,

                    

                                    'not_injured_defense_Linebacker' : not_injured_defense_Linebacker,

                                    'not_injured_defense_DefensiveLineman' : not_injured_defense_DefensiveLineman,

                                    'not_injured_defense_Safety' : not_injured_defense_Safety,

                                    'not_injured_defense_Cornerback' : not_injured_defense_Cornerback

                                    

                },

                'player_group_df' : {

                                    'injured_offense_Quarterback_df' : pd.DataFrame(),

                                    'injured_offense_WideReceiver_df' : pd.DataFrame(),

                                    'injured_offense_RunningBack_df' : pd.DataFrame(),

                                    'injured_offense_TightEnd_df' : pd.DataFrame(),

                                    'injured_offense_OffensiveLineman_df' : pd.DataFrame(),

                                    'injured_offense_Kicker_df' : pd.DataFrame(),

                    

                                    'injured_defense_Linebacker_df' : pd.DataFrame(),

                                    'injured_defense_DefensiveLineman_df' : pd.DataFrame(),

                                    'injured_defense_Safety_df' : pd.DataFrame(),

                                    'injured_defense_Cornerback_df' : pd.DataFrame(),

                    

                                    'not_injured_offense_Quarterback_df' : pd.DataFrame(),

                                    'not_injured_offense_WideReceiver_df' : pd.DataFrame(),

                                    'not_injured_offense_RunningBack_df' : pd.DataFrame(),

                                    'not_injured_offense_TightEnd_df' :  pd.DataFrame(),

                                    'not_injured_offense_OffensiveLineman_df' : pd.DataFrame(),

                                    'not_injured_offense_Kicker_df' : pd.DataFrame(),

                    

                                    'not_injured_defense_Linebacker_df' : pd.DataFrame(),

                                    'not_injured_defense_DefensiveLineman_df' : pd.DataFrame(),

                                    'not_injured_defense_Safety_df' : pd.DataFrame(),

                                    'not_injured_defense_Cornerback_df' : pd.DataFrame()

                },

                'player_group_cv' : {

                                    'injured_offense_Quarterback_cv' : pd.DataFrame(),

                                    'injured_offense_WideReceiver_cv' : pd.DataFrame(),

                                    'injured_offense_RunningBack_cv' : pd.DataFrame(),

                                    'injured_offense_TightEnd_cv' : pd.DataFrame(),

                                    'injured_offense_OffensiveLineman_cv' : pd.DataFrame(),

                                    'injured_offense_Kicker_cv' : pd.DataFrame(),

                    

                                    'injured_defense_Linebacker_cv' : pd.DataFrame(),

                                    'injured_defense_DefensiveLineman_cv' : pd.DataFrame(),

                                    'injured_defense_Safety_cv' : pd.DataFrame(),

                                    'injured_defense_Cornerback_cv' : pd.DataFrame(),

                    

                                    'not_injured_offense_Quarterback_cv' : pd.DataFrame(),

                                    'not_injured_offense_WideReceiver_cv' : pd.DataFrame(),

                                    'not_injured_offense_RunningBack_cv' : pd.DataFrame(),

                                    'not_injured_offense_TightEnd_cv' :  pd.DataFrame(),

                                    'not_injured_offense_OffensiveLineman_cv' : pd.DataFrame(),

                                    'not_injured_offense_Kicker_cv' : pd.DataFrame(),

                    

                                    'not_injured_defense_Linebacker_cv' : pd.DataFrame(),

                                    'not_injured_defense_DefensiveLineman_cv' : pd.DataFrame(),

                                    'not_injured_defense_Safety_cv' : pd.DataFrame(),

                                    'not_injured_defense_Cornerback_cv' : pd.DataFrame()

                }

}
sns.catplot(x="BodyPart", kind="count", palette="ch:.25", hue = 'Surface',

            data=injury_data[injury_data['BodyPart'].isin(['Toes', 'Foot'])])

### Get data from PlayerTrackData table



bgq = True



injured_player_group_name = 'injured_offense_WideReceiver'

not_injured_player_group_name = 'not_injured_offense_WideReceiver'

number_of_players = 2 ### Higher number will lead to longer processing time.



if bgq:

    groups['player_group_df'][injured_player_group_name + '_df'] = DeriveFeatures(GetPlayerTrackData('PlayKey, PlayerKey, time, event, dir, s, o, dis', 

                                                                      'PlayerTrackData', groups['player_group'][injured_player_group_name][0:number_of_players]))

    groups['player_group_df'][not_injured_player_group_name + '_df'] = DeriveFeatures(GetPlayerTrackData('PlayKey, PlayerKey, time, event, dir, s, o, dis', 

                                                                          'PlayerTrackData', groups['player_group'][not_injured_player_group_name][0:number_of_players]))

groups['player_group_df'][not_injured_player_group_name + '_df'][['estimated_acceleration_x_component',

                                                                  'estimated_acceleration_y_component', 

                                                                  'rate_of_change_in_twist', 

                                                                  'rate_of_change_in_directional_motion',

                                                                  'rate_of_change_in_orientation']].replace(0, np.NaN).mean()
groups['player_group_df'][injured_player_group_name + '_df'][['estimated_acceleration_x_component',

                                                                  'estimated_acceleration_y_component', 

                                                                  'rate_of_change_in_twist', 

                                                                  'rate_of_change_in_directional_motion',

                                                                  'rate_of_change_in_orientation']].replace(0, np.NaN).mean()
from scipy.stats import ttest_ind





ttest_ind(groups['player_group_df'][injured_player_group_name + '_df']['rate_of_change_in_twist'].dropna(),

          groups['player_group_df'][not_injured_player_group_name + '_df']['rate_of_change_in_twist'].dropna(), equal_var=False)
f,axes=plt.subplots(2,5,figsize=(24,10),sharey=False)

bins = 100

kde = True



############## Not Injured Group ##############

sns.distplot(groups['player_group_df'][not_injured_player_group_name + '_df'][groups['player_group_df'][not_injured_player_group_name + '_df']['estimated_acceleration_x_component']

                                                 .between(-100,100)]['estimated_acceleration_x_component'],

                                                 bins = bins,

                                                 kde = kde,

                                                 ax = axes[0,0])

axes[0,0].set_title(not_injured_player_group_name)



sns.distplot(groups['player_group_df'][not_injured_player_group_name + '_df'][groups['player_group_df'][not_injured_player_group_name + '_df']['estimated_acceleration_y_component']

                                                 .between(-100,100)]['estimated_acceleration_y_component'],

                                                 bins = bins,

                                                 kde = kde,

                                                 ax = axes[0,1])

axes[0,1].set_title(not_injured_player_group_name)



sns.distplot(groups['player_group_df'][not_injured_player_group_name + '_df'][groups['player_group_df'][not_injured_player_group_name + '_df']['rate_of_change_in_twist']

                                                 .between(-50,50)]['rate_of_change_in_twist'],

                                                 bins = bins,

                                                 kde = kde,

                                                 ax = axes[0,2])

axes[0,2].set_title(not_injured_player_group_name)



sns.distplot(groups['player_group_df'][not_injured_player_group_name + '_df'][groups['player_group_df'][not_injured_player_group_name + '_df']['rate_of_change_in_directional_motion']

                                                 .between(-300,300)]['rate_of_change_in_directional_motion'],

                                                 bins = bins,

                                                 kde = kde,

                                                 ax = axes[0,3])

axes[0,3].set_title(not_injured_player_group_name)



sns.distplot(groups['player_group_df'][not_injured_player_group_name + '_df'][groups['player_group_df'][not_injured_player_group_name + '_df']['rate_of_change_in_orientation']

                                                 .between(-300,300)]['rate_of_change_in_orientation'],

                                                 bins = bins,

                                                 kde = kde,

                                                 ax = axes[0,4])

axes[0,4].set_title(not_injured_player_group_name)







############## Injured Group ##############

sns.distplot(groups['player_group_df'][injured_player_group_name + '_df'][groups['player_group_df'][injured_player_group_name + '_df']['estimated_acceleration_x_component']

                                                 .between(-100,100)]['estimated_acceleration_x_component'],

                                                 bins = bins,

                                                 kde = kde,

                                                 ax = axes[1,0])

axes[1,0].set_title(injured_player_group_name)



sns.distplot(groups['player_group_df'][injured_player_group_name + '_df'][groups['player_group_df'][injured_player_group_name + '_df']['estimated_acceleration_y_component']

                                                 .between(-100,100)]['estimated_acceleration_y_component'],

                                                 bins = bins,

                                                 kde = kde,

                                                 ax = axes[1,1])

axes[1,1].set_title(injured_player_group_name)



sns.distplot(groups['player_group_df'][injured_player_group_name + '_df'][groups['player_group_df'][injured_player_group_name + '_df']['rate_of_change_in_twist']

                                                 .between(-50,50)]['rate_of_change_in_twist'],

                                                 bins = bins,

                                                 kde = kde,

                                                 ax = axes[1,2])

axes[1,2].set_title(injured_player_group_name)



sns.distplot(groups['player_group_df'][injured_player_group_name + '_df'][groups['player_group_df'][injured_player_group_name + '_df']['rate_of_change_in_directional_motion']

                                                 .between(-300,300)]['rate_of_change_in_directional_motion'],

                                                 bins = bins,

                                                 kde = kde,

                                                 ax = axes[1,3])

axes[1,3].set_title(injured_player_group_name)



sns.distplot(groups['player_group_df'][injured_player_group_name + '_df'][groups['player_group_df'][injured_player_group_name + '_df']['rate_of_change_in_orientation']

                                                 .between(-300,300)]['rate_of_change_in_orientation'],

                                                 bins = bins,

                                                 kde = kde,

                                                 ax = axes[1,4])

axes[1,4].set_title(injured_player_group_name)
groups['player_group_cv'][injured_player_group_name + '_cv'] = CalculateCOV(groups['player_group_df'][injured_player_group_name + '_df'][[  'PlayKey', 

                                                                          'PlayerKey', 

                                                                          'estimated_acceleration_x_component',

                                                                          'estimated_acceleration_y_component', 

                                                                          'rate_of_change_in_twist', 

                                                                          'rate_of_change_in_directional_motion',

                                                                          'rate_of_change_in_orientation']])
groups['player_group_cv'][not_injured_player_group_name + '_cv'] = CalculateCOV(groups['player_group_df'][not_injured_player_group_name + '_df'][['PlayKey', 

                                                                              'PlayerKey', 

                                                                              'estimated_acceleration_x_component',

                                                                              'estimated_acceleration_y_component', 

                                                                              'rate_of_change_in_twist', 

                                                                              'rate_of_change_in_directional_motion',

                                                                              'rate_of_change_in_orientation']])
f,axes=plt.subplots(2,5,figsize=(24,10),sharey=False)

bins = 100

kde = True



############## Not Injured Group ##############

sns.distplot(groups['player_group_cv'][not_injured_player_group_name + '_cv'][groups['player_group_cv'][not_injured_player_group_name + '_cv']['estimated_acceleration_x_component']

                                                 .between(-100,100)]['estimated_acceleration_x_component'],

                                                 bins = bins,

                                                 kde = kde,

                                                 ax = axes[0,0])

axes[0,0].set_title(not_injured_player_group_name)



sns.distplot(groups['player_group_cv'][not_injured_player_group_name + '_cv'][groups['player_group_cv'][not_injured_player_group_name + '_cv']['estimated_acceleration_y_component']

                                                 .between(-100,100)]['estimated_acceleration_y_component'],

                                                 bins = bins,

                                                 kde = kde,

                                                 ax = axes[0,1])

axes[0,1].set_title(not_injured_player_group_name)



sns.distplot(groups['player_group_cv'][not_injured_player_group_name + '_cv'][groups['player_group_cv'][not_injured_player_group_name + '_cv']['rate_of_change_in_twist']

                                                 .between(-50,50)]['rate_of_change_in_twist'],

                                                 bins = bins,

                                                 kde = kde,

                                                 ax = axes[0,2])

axes[0,2].set_title(not_injured_player_group_name)



sns.distplot(groups['player_group_cv'][not_injured_player_group_name + '_cv'][groups['player_group_cv'][not_injured_player_group_name + '_cv']['rate_of_change_in_directional_motion']

                                                 .between(-300,300)]['rate_of_change_in_directional_motion'],

                                                 bins = bins,

                                                 kde = kde,

                                                 ax = axes[0,3])

axes[0,3].set_title(not_injured_player_group_name)



sns.distplot(groups['player_group_cv'][not_injured_player_group_name + '_cv'][groups['player_group_cv'][not_injured_player_group_name + '_cv']['rate_of_change_in_orientation']

                                                 .between(-300,300)]['rate_of_change_in_orientation'],

                                                 bins = bins,

                                                 kde = kde,

                                                 ax = axes[0,4])

axes[0,4].set_title(not_injured_player_group_name)







############## Injured Group ##############

sns.distplot(groups['player_group_cv'][injured_player_group_name + '_cv'][groups['player_group_cv'][injured_player_group_name + '_cv']['estimated_acceleration_x_component']

                                                 .between(-100,100)]['estimated_acceleration_x_component'],

                                                 bins = bins,

                                                 kde = kde,

                                                 ax = axes[1,0])

axes[1,0].set_title(injured_player_group_name)



sns.distplot(groups['player_group_cv'][injured_player_group_name + '_cv'][groups['player_group_cv'][injured_player_group_name + '_cv']['estimated_acceleration_y_component']

                                                 .between(-100,100)]['estimated_acceleration_y_component'],

                                                 bins = bins,

                                                 kde = kde,

                                                 ax = axes[1,1])

axes[1,1].set_title(injured_player_group_name)



sns.distplot(groups['player_group_cv'][injured_player_group_name + '_cv'][groups['player_group_cv'][injured_player_group_name + '_cv']['rate_of_change_in_twist']

                                                 .between(-50,50)]['rate_of_change_in_twist'],

                                                 bins = bins,

                                                 kde = kde,

                                                 ax = axes[1,2])

axes[1,2].set_title(injured_player_group_name)



sns.distplot(groups['player_group_cv'][injured_player_group_name + '_cv'][groups['player_group_cv'][injured_player_group_name + '_cv']['rate_of_change_in_directional_motion']

                                                 .between(-300,300)]['rate_of_change_in_directional_motion'],

                                                 bins = bins,

                                                 kde = kde,

                                                 ax = axes[1,3])

axes[1,3].set_title(injured_player_group_name)



sns.distplot(groups['player_group_cv'][injured_player_group_name + '_cv'][groups['player_group_cv'][injured_player_group_name + '_cv']['rate_of_change_in_orientation']

                                                 .between(-300,300)]['rate_of_change_in_orientation'],

                                                 bins = bins,

                                                 kde = kde,

                                                 ax = axes[1,4])

axes[1,4].set_title(injured_player_group_name)
injured_player_group_name = 'injured_defense_Cornerback'

temp_df = groups['player_group_df'][injured_player_group_name + '_df'][groups['player_group_df'][injured_player_group_name + '_df']['PlayKey'] == '36559-1-1']

#temp_df = not_injured_defense_Linebacker_df[not_injured_defense_Linebacker_df['PlayKey'] == '30953-1-1']
f,axes=plt.subplots(3,1,figsize=(24,14),sharex=True)



axes[0].plot(temp_df['time'], temp_df['estimated_acceleration_x_component'].fillna(0))

for evnt in temp_df['event']:

    try:

        axes[0].axvline(x=temp_df[temp_df['event'] == evnt]['time'].values[0], color = 'r')

        axes[0].text(temp_df[temp_df['event'] == evnt]['time'].values[0]+0.11,0,evnt, rotation=90)

        axes[0].set_ylabel('estimated_acceleration_x_component')

        axes[0].set_xlabel('Time')

    except:

        pass

    

axes[1].plot(temp_df['time'], temp_df['estimated_acceleration_y_component'].fillna(0))

for evnt in temp_df['event']:

    try:

        axes[1].axvline(x=temp_df[temp_df['event'] == evnt]['time'].values[0], color = 'r')

        axes[1].text(temp_df[temp_df['event'] == evnt]['time'].values[0]+0.11,0,evnt, rotation=90)

        axes[1].set_ylabel('estimated_acceleration_y_component')

        axes[1].set_xlabel('Time')

    except:

        pass



axes[2].plot(temp_df['time'], temp_df['twist_dir_minus_o'].fillna(0))

for evnt in temp_df['event']:

    try:

        axes[2].axvline(x=temp_df[temp_df['event'] == evnt]['time'].values[0], color = 'r')

        axes[2].text(temp_df[temp_df['event'] == evnt]['time'].values[0]+0.11,0,evnt, rotation=90)

        axes[2].set_ylabel('Difference between direction of motion and player orientation')

        axes[2].set_xlabel('Time')

    except:

        pass
