# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
matches= pd.read_csv('../input/ipldata/matches.csv')
deliveries=pd.read_csv('../input/ipldata/deliveries.csv')
matches[matches['season'].isin([2017,2018,2019])]['id'].unique()
deliveries.shape
a_2017_2018_2019=deliveries[deliveries['match_id'].isin([ 1,     2,     3,     4,     5,     6,     7,     8,     9,
          10,    11,    12,    13,    14,    15,    16,    17,    18,
          19,    20,    21,    22,    23,    24,    25,    26,    27,
          28,    29,    30,    31,    32,    33,    34,    35,    36,
          37,    38,    39,    40,    41,    42,    43,    44,    45,
          46,    47,    48,    49,    50,    51,    52,    53,    54,
          55,    56,    57,    58,    59,  7894,  7895,  7896,  7897,
        7898,  7899,  7900,  7901,  7902,  7903,  7904,  7905,  7906,
        7907,  7908,  7909,  7910,  7911,  7912,  7913,  7914,  7915,
        7916,  7917,  7918,  7919,  7920,  7921,  7922,  7923,  7924,
        7925,  7926,  7927,  7928,  7929,  7930,  7931,  7932,  7933,
        7934,  7935,  7936,  7937,  7938,  7939,  7940,  7941,  7942,
        7943,  7944,  7945,  7946,  7947,  7948,  7949,  7950,  7951,
        7952,  7953, 11137, 11138, 11139, 11140, 11141, 11142, 11143,
       11144, 11145, 11146, 11147, 11148, 11149, 11150, 11151, 11152,
       11153, 11309, 11310, 11311, 11312, 11313, 11314, 11315, 11316,
       11317, 11318, 11319, 11320, 11321, 11322, 11323, 11324, 11325,
       11326, 11327, 11328, 11329, 11330, 11331, 11332, 11333, 11334,
       11335, 11336, 11337, 11338, 11339, 11340, 11341, 11342, 11343,
       11344, 11345, 11346, 11347, 11412, 11413, 11414, 11415])].index
a_2017_2018_2019
deliveries.drop(a_2017_2018_2019,inplace=True)
deliveries.shape
batsman=deliveries[['batsman','match_id']].groupby(['batsman']).agg('nunique')
batsman['id']=batsman.index
batsman.reset_index(drop=True,inplace=True)
batsman=batsman.rename(columns={'match_id':'matches','id':'batsman'})
batsman=batsman[['batsman','matches']]
batsman_total_runs=deliveries[['batsman','total_runs']].groupby(['batsman']).agg('sum')
batsman_total_runs['id']=batsman_total_runs.index
batsman_total_runs.reset_index(drop=True,inplace=True)
batsman['total_runs']=batsman_total_runs['total_runs']
batsman_high_score=deliveries[['batsman','match_id','total_runs']].groupby(['batsman','match_id']).agg('sum').groupby(['batsman']).agg('max')
batsman_high_score['id']=batsman_high_score.index
batsman_high_score.reset_index(drop=True,inplace=True)
batsman_high_score=batsman_high_score.rename(columns={'total_runs':'HS','id':'batsman'})
batsman['HS']=batsman_high_score['HS']
batsman_balls_faced=deliveries[['batsman','total_runs']].groupby(['batsman']).agg('count')
batsman_balls_faced['id']=batsman_balls_faced.index
batsman_balls_faced.reset_index(drop=True,inplace=True)
batsman_balls_faced=batsman_balls_faced.rename(columns={'total_runs':'ballsfaced','id':'batsman'})
batsman['ballsfaced']=batsman_balls_faced['ballsfaced']
batsman_4s=deliveries[deliveries['total_runs']==4][['batsman','total_runs']].groupby(['batsman']).agg('count')
batsman_4s['id']=batsman_4s.index
batsman_4s.reset_index(drop=True,inplace=True)
batsman_4s=batsman_4s.rename(columns={'total_runs':"4's",'id':'batsman'})
batsman=pd.merge(batsman,batsman_4s,how='left',on='batsman')
batsman_6s=deliveries[deliveries['total_runs']==6][['batsman','total_runs']].groupby(['batsman']).agg('count')
batsman_6s['id']=batsman_6s.index
batsman_6s.reset_index(drop=True,inplace=True)
batsman_6s=batsman_6s.rename(columns={'total_runs':"6's",'id':'batsman'})
batsman=pd.merge(batsman,batsman_6s,how='left',on='batsman')
batsman[["4's","6's"]]=batsman[["4's","6's"]].fillna(0)
batsman["4's"]=batsman["4's"].astype(int)
batsman["6's"]=batsman["6's"].astype(int)
batsman['Avg(TR/M)']=batsman['total_runs']/batsman['matches']
batsman['SR(TR/BF)']=(batsman['total_runs']/batsman['ballsfaced'])*100
player_dismissed_entire_record=deliveries[['batsman','player_dismissed']].groupby(['batsman']).count()
player_dismissed_entire_record['batsman']=player_dismissed_entire_record.index
player_dismissed_entire_record.reset_index(drop=True,inplace=True)
batsman1=pd.merge(batsman,player_dismissed_entire_record,how='left',on='batsman')
batsman1=batsman1.rename(columns={'matches':'batting_played'})
batsman1=batsman1[['batsman','batting_played','player_dismissed', 'total_runs', 'HS', 'ballsfaced', "4's",
       "6's", 'Avg(TR/M)','SR(TR/BF)', ]]
batsman1['battingAVG(tr/outs)']=batsman1['total_runs']/batsman1['player_dismissed']
batsman1['runs_by_balls(R/B)']=batsman1['total_runs']/batsman1['ballsfaced']
batsman1['Calc']=batsman1['battingAVG(tr/outs)']*batsman1['runs_by_balls(R/B)']
batsman1
bowler=deliveries[['bowler','match_id']].groupby(['bowler']).agg('nunique')
bowler['id']=bowler.index
bowler.reset_index(drop=True,inplace=True)
bowler=bowler.rename(columns={'match_id':'matches','id':'bowler'})
bowler=bowler[['bowler','matches']]
bowler_bowled=deliveries[['bowler','over']].groupby(['bowler']).agg('count')
bowler_bowled['id']=bowler_bowled.index
bowler_bowled.reset_index(drop=True,inplace=True)
bowler_bowled=bowler_bowled.rename(columns={'over':'ballsbowled','id':'bowler'})
bowler['ballsbowled']=bowler_bowled['ballsbowled']
bowler_wickets=deliveries[['bowler','player_dismissed']].groupby(['bowler']).count()
bowler_wickets['id']=bowler_wickets.index
bowler_wickets.reset_index(drop=True,inplace=True)
bowler_wickets=bowler_wickets.rename(columns={'player_dismissed':'wickets','id':'bowler'})
bowler['wkts']=bowler_wickets['wickets']
bowler_givenruns=deliveries[['bowler','total_runs']].groupby(['bowler']).agg('sum')
bowler_givenruns['id']=bowler_givenruns.index
bowler_givenruns.reset_index(drop=True,inplace=True)
bowler_givenruns=bowler_givenruns.rename(columns={'total_runs':'runs','id':'bowler'})
bowler['runs']=bowler_givenruns['runs']
bowler['Avg(TR/Wkts)']=bowler['runs']/bowler['wkts']
bowler['bowling_sr(bb/wkts)']=bowler['ballsbowled']/bowler['wkts']
bowler['economy_rate(runs/bb)']=bowler['runs']/bowler['ballsbowled']
bowler['(1/bowling_avg)']=1/bowler['Avg(TR/Wkts)']
bowler['(1/bowling_sr)']=1/bowler['bowling_sr(bb/wkts)']
bowler['(1/bowling_er)']=1/bowler['economy_rate(runs/bb)']
bowler['bowling_sum']=bowler['(1/bowling_avg)']+bowler['(1/bowling_sr)']+bowler['(1/bowling_er)']
bowler.columns
bowler['CombinedBowlingRate']=3/bowler['bowling_sum']
bowler.columns
bowler
matches_season_2017=matches[matches['season']==2017]
matches_season_2017['id'].unique()
deliveries_complete_data=pd.read_csv('../input/ipldata/deliveries.csv')
deliveries_season_2017=deliveries_complete_data[deliveries_complete_data['match_id'].isin([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17,
       18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34,
       35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51,
       52, 53, 54, 55, 56, 57, 58, 59])]
deliveries_season_2017.shape
matches[matches['id']==1]
deliveries_complete_data[(deliveries_complete_data['match_id']==1)&(deliveries_complete_data['batting_team']=='Sunrisers Hyderabad')]['batsman'].unique()
batsman1[batsman1['batsman'].isin(['MC Henriques','S Dhawan','Yuvraj Singh','DA Warner', 'DJ Hooda', 'BCJ Cutting'])]
batsman1[batsman1['batsman'].isin(['MC Henriques','S Dhawan','Yuvraj Singh','DA Warner', 'DJ Hooda', 'BCJ Cutting'])]['Calc'].agg('mean')
deliveries_complete_data[(deliveries_complete_data['match_id']==1)&(deliveries_complete_data['bowling_team']=='Sunrisers Hyderabad')]['bowler'].unique()
bowler[bowler['bowler'].isin(['A Nehra', 'B Kumar', 'BCJ Cutting', 'Rashid Khan', 'DJ Hooda',
       'MC Henriques', 'Bipul Sharma'])]
bowler[bowler['bowler'].isin(['A Nehra', 'B Kumar', 'BCJ Cutting', 'Rashid Khan', 'DJ Hooda',
       'MC Henriques', 'Bipul Sharma'])]['CombinedBowlingRate'].agg('mean')
deliveries_complete_data[(deliveries_complete_data['match_id']==1)&(deliveries_complete_data['batting_team']=='Royal Challengers Bangalore')]['batsman'].unique()
batsman1[batsman1['batsman'].isin(['CH Gayle', 'Mandeep Singh', 'TM Head', 'KM Jadhav', 'SR Watson',
       'Sachin Baby', 'STR Binny', 'S Aravind', 'YS Chahal', 'TS Mills',
       'A Choudhary'])]
batsman1[batsman1['batsman'].isin(['CH Gayle', 'Mandeep Singh', 'TM Head', 'KM Jadhav', 'SR Watson',
       'Sachin Baby', 'STR Binny'])]['Calc'].agg('mean')
deliveries_complete_data[(deliveries_complete_data['match_id']==1)&(deliveries_complete_data['bowling_team']=='Royal Challengers Bangalore')]['bowler'].unique()
bowler[bowler['bowler'].isin(['TS Mills', 'A Choudhary', 'YS Chahal', 'S Aravind', 'SR Watson',
       'TM Head', 'STR Binny'])]
bowler[bowler['bowler'].isin(['TS Mills', 'A Choudhary', 'YS Chahal', 'S Aravind', 'SR Watson',
       'TM Head', 'STR Binny'])]['CombinedBowlingRate'].agg('mean')
matches[matches['id']==2]
deliveries_complete_data[(deliveries_complete_data['match_id']==2)&(deliveries_complete_data['batting_team']=='Rising Pune Supergiant')]['batsman'].unique()
batsman1[batsman1['batsman'].isin(['AM Rahane', 'MA Agarwal', 'SPD Smith', 'BA Stokes', 'MS Dhoni'])]
batsman1[batsman1['batsman'].isin(['AM Rahane', 'MA Agarwal', 'SPD Smith', 'BA Stokes', 'MS Dhoni'])]['Calc'].agg('mean')
deliveries_complete_data[(deliveries_complete_data['match_id']==2)&(deliveries_complete_data['batting_team']=='Mumbai Indians')]['batsman'].unique()
batsman1[batsman1['batsman'].isin(['PA Patel', 'JC Buttler', 'RG Sharma', 'N Rana', 'AT Rayudu',
       'KH Pandya', 'KA Pollard', 'HH Pandya', 'TG Southee'])]
batsman1[batsman1['batsman'].isin(['PA Patel', 'JC Buttler', 'RG Sharma', 'N Rana', 'AT Rayudu',
       'KH Pandya', 'KA Pollard', 'HH Pandya', 'TG Southee'])]['Calc'].agg('mean')
deliveries_complete_data[(deliveries_complete_data['match_id']==2)&(deliveries_complete_data['bowling_team']=='Rising Pune Supergiant')]['bowler'].unique()
bowler[bowler['bowler'].isin(['AB Dinda', 'DL Chahar', 'BA Stokes', 'Imran Tahir', 'A Zampa',
       'R Bhatia'])]
bowler[bowler['bowler'].isin(['AB Dinda', 'DL Chahar', 'BA Stokes', 'Imran Tahir', 'A Zampa',
       'R Bhatia'])]['CombinedBowlingRate'].agg('mean')
deliveries_complete_data[(deliveries_complete_data['match_id']==2)&(deliveries_complete_data['bowling_team']=='Mumbai Indians')]['bowler'].unique()
bowler[bowler['bowler'].isin(['TG Southee', 'HH Pandya', 'MJ McClenaghan', 'JJ Bumrah',
       'KH Pandya', 'KA Pollard'])]
bowler[bowler['bowler'].isin(['TG Southee', 'HH Pandya', 'MJ McClenaghan', 'JJ Bumrah',
       'KH Pandya', 'KA Pollard'])]['CombinedBowlingRate'].agg('mean')
matches[matches['id']==3]
deliveries_complete_data[(deliveries_complete_data['match_id']==3)&(deliveries_complete_data['batting_team']=='Gujarat Lions')]['batsman'].unique()
batsman1[batsman1['batsman'].isin(['JJ Roy', 'BB McCullum', 'SK Raina', 'AJ Finch', 'KD Karthik'])]
batsman1[batsman1['batsman'].isin(['JJ Roy', 'BB McCullum', 'SK Raina', 'AJ Finch', 'KD Karthik'])]['Calc'].agg('mean')
deliveries_complete_data[(deliveries_complete_data['match_id']==3)&(deliveries_complete_data['bowling_team']=='Gujarat Lions')]['bowler'].unique()
bowler[bowler['bowler'].isin(['P Kumar', 'DS Kulkarni', 'MS Gony', 'S Kaushik', 'DR Smith',
       'SB Jakati'])]
bowler[bowler['bowler'].isin(['P Kumar', 'DS Kulkarni', 'MS Gony', 'S Kaushik', 'DR Smith',
       'SB Jakati'])]['CombinedBowlingRate'].agg('mean')
deliveries_complete_data[(deliveries_complete_data['match_id']==3)&(deliveries_complete_data['batting_team']=='Kolkata Knight Riders')]['batsman'].unique()
batsman1[batsman1['batsman'].isin(['G Gambhir', 'CA Lynn'])]
batsman1[batsman1['batsman'].isin(['G Gambhir', 'CA Lynn'])]['Calc'].agg('mean')
deliveries_complete_data[(deliveries_complete_data['match_id']==3)&(deliveries_complete_data['bowling_team']=='Kolkata Knight Riders')]['bowler'].unique()
bowler[bowler['bowler'].isin(['TA Boult', 'PP Chawla', 'SP Narine', 'CR Woakes', 'Kuldeep Yadav',
       'YK Pathan'])]
bowler[bowler['bowler'].isin(['TA Boult', 'PP Chawla', 'SP Narine', 'CR Woakes', 'Kuldeep Yadav',
       'YK Pathan'])]['CombinedBowlingRate'].agg('mean')
matches[matches['id']==4]
deliveries_complete_data[(deliveries_complete_data['match_id']==4)&(deliveries_complete_data['batting_team']=='Rising Pune Supergiant')]['batsman'].unique()
batsman1[batsman1['batsman'].isin(['AM Rahane', 'MA Agarwal', 'SPD Smith', 'BA Stokes', 'MS Dhoni',
       'MK Tiwary', 'DT Christian'])]
batsman1[batsman1['batsman'].isin(['AM Rahane', 'MA Agarwal', 'SPD Smith', 'BA Stokes', 'MS Dhoni',
       'MK Tiwary', 'DT Christian'])]['Calc'].agg('mean')
deliveries_complete_data[(deliveries_complete_data['match_id']==4)&(deliveries_complete_data['bowling_team']=='Rising Pune Supergiant')]['bowler'].unique()
bowler[bowler['bowler'].isin(['AB Dinda', 'DT Christian', 'BA Stokes', 'Imran Tahir',
       'RD Chahar', 'R Bhatia'])]
bowler[bowler['bowler'].isin(['AB Dinda', 'DT Christian', 'BA Stokes', 'Imran Tahir',
       'RD Chahar', 'R Bhatia'])]['CombinedBowlingRate'].agg('mean')
deliveries_complete_data[(deliveries_complete_data['match_id']==4)&(deliveries_complete_data['batting_team']=='Kings XI Punjab')]['batsman'].unique()
batsman1[batsman1['batsman'].isin(['HM Amla', 'M Vohra', 'WP Saha', 'AR Patel', 'GJ Maxwell',
       'DA Miller'])]
batsman1[batsman1['batsman'].isin(['HM Amla', 'M Vohra', 'WP Saha', 'AR Patel', 'GJ Maxwell',
       'DA Miller'])]['Calc'].agg('mean')
deliveries_complete_data[(deliveries_complete_data['match_id']==4)&(deliveries_complete_data['bowling_team']=='Kings XI Punjab')]['bowler'].unique()
bowler[bowler['bowler'].isin(['Sandeep Sharma', 'MM Sharma', 'AR Patel', 'T Natarajan',
       'MP Stoinis', 'Swapnil Singh'])]
bowler[bowler['bowler'].isin(['Sandeep Sharma', 'MM Sharma', 'AR Patel', 'T Natarajan',
       'MP Stoinis', 'Swapnil Singh'])]['CombinedBowlingRate'].agg('mean')
matches[matches['id']==5]
deliveries_complete_data[(deliveries_complete_data['match_id']==5)&(deliveries_complete_data['batting_team']=='Royal Challengers Bangalore')]['batsman'].unique()
batsman1[batsman1['batsman'].isin(['CH Gayle', 'SR Watson', 'Mandeep Singh', 'KM Jadhav', 'STR Binny',
       'Vishnu Vinod', 'Iqbal Abdulla', 'P Negi', 'TS Mills'])]
batsman1[batsman1['batsman'].isin(['CH Gayle', 'SR Watson', 'Mandeep Singh', 'KM Jadhav', 'STR Binny',
       'Vishnu Vinod', 'Iqbal Abdulla', 'P Negi', 'TS Mills'])]['Calc'].agg('mean')
deliveries_complete_data[(deliveries_complete_data['match_id']==5)&(deliveries_complete_data['bowling_team']=='Royal Challengers Bangalore')]['bowler'].unique()
bowler[bowler['bowler'].isin(['B Stanlake', 'YS Chahal', 'Iqbal Abdulla', 'TS Mills',
       'SR Watson', 'P Negi'])]
bowler[bowler['bowler'].isin(['B Stanlake', 'YS Chahal', 'Iqbal Abdulla', 'TS Mills',
       'SR Watson', 'P Negi'])]['CombinedBowlingRate'].agg('mean')
deliveries_complete_data[(deliveries_complete_data['match_id']==5)&(deliveries_complete_data['batting_team']=='Delhi Daredevils')]['batsman'].unique()
batsman1[batsman1['batsman'].isin(['AP Tare', 'SW Billings', 'KK Nair', 'SV Samson', 'RR Pant',
       'CH Morris', 'CR Brathwaite', 'PJ Cummins', 'A Mishra', 'S Nadeem',
       'Z Khan'])]
batsman1[batsman1['batsman'].isin(['AP Tare', 'SW Billings', 'KK Nair', 'SV Samson', 'RR Pant',
       'CH Morris', 'CR Brathwaite', 'PJ Cummins', 'A Mishra', 'S Nadeem',
       'Z Khan'])]['Calc'].agg('mean')
deliveries_complete_data[(deliveries_complete_data['match_id']==5)&(deliveries_complete_data['bowling_team']=='Delhi Daredevils')]['bowler'].unique()
bowler[bowler['bowler'].isin(['Z Khan', 'CH Morris', 'PJ Cummins', 'S Nadeem', 'A Mishra',
       'CR Brathwaite'])]
bowler[bowler['bowler'].isin(['Z Khan', 'CH Morris', 'PJ Cummins', 'S Nadeem', 'A Mishra',
       'CR Brathwaite'])]['CombinedBowlingRate'].agg('mean')
batsman1[batsman1['Calc'].isin(['inf'])]
