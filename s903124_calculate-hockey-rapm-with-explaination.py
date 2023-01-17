import pandas as pd

import numpy as np

pd.options.mode.chained_assignment = None 

pd.options.display.max_rows= 20

from sklearn.preprocessing import MultiLabelBinarizer

from scipy.sparse import hstack,csr_matrix

from sklearn.linear_model import RidgeCV



nhl_pbp = pd.read_csv('/kaggle/input/nhl-playbyplay-data-from-2007/hockey_scraper_data/pbp/nhl_pbp20182019.csv')

nhl_pbp = nhl_pbp[nhl_pbp['Game_Id'] < 30000 ] #Regular season only
nhl_pbp['poss_time'] = pd.to_numeric(nhl_pbp['Seconds_Elapsed']).diff().fillna(0).clip(lower=0)



nhl_pbp['home_shot'] = 0

nhl_pbp['away_shot'] = 0



nhl_pbp.loc[((nhl_pbp['Event'] == 'SHOT') | (nhl_pbp['Event'] == 'MISS') | (nhl_pbp['Event'] == 'GOAL')) & ((nhl_pbp[['homePlayer1_id','homePlayer2_id','homePlayer3_id','homePlayer4_id','homePlayer5_id','homePlayer6_id']].isin(nhl_pbp['p1_ID']).any(1))),'home_shot'] = 1

nhl_pbp.loc[((nhl_pbp['Event'] == 'SHOT') | (nhl_pbp['Event'] == 'MISS') | (nhl_pbp['Event'] == 'GOAL')) & ((nhl_pbp[['awayPlayer1_id','awayPlayer2_id','awayPlayer3_id','awayPlayer4_id','awayPlayer5_id','awayPlayer6_id']].isin(nhl_pbp['p1_ID']).any(1))),'away_shot'] = 1



nhl_pbp.loc[(nhl_pbp['Event'] == 'BLOCK' ) & ((nhl_pbp[['awayPlayer1_id','awayPlayer2_id','awayPlayer3_id','awayPlayer4_id','awayPlayer5_id','awayPlayer6_id']].isin(nhl_pbp['p1_ID']).any(1))),'home_shot'] = 1

nhl_pbp.loc[(nhl_pbp['Event'] == 'BLOCK' ) & ((nhl_pbp[['homePlayer1_id','homePlayer2_id','homePlayer3_id','homePlayer4_id','homePlayer5_id','homePlayer6_id']].isin(nhl_pbp['p1_ID']).any(1))),'away_shot'] = 1



nhl_pbp.loc[(nhl_pbp['home_shot'] == 0 ) & ((nhl_pbp['Event'] == 'SHOT') | (nhl_pbp['Event'] == 'MISS') | (nhl_pbp['Event'] == 'GOAL')) & (nhl_pbp['Home_Zone'] == 'Off'), 'home_shot'] = 1

nhl_pbp.loc[(nhl_pbp['away_shot'] == 0 ) & ((nhl_pbp['Event'] == 'SHOT') | (nhl_pbp['Event'] == 'MISS') | (nhl_pbp['Event'] == 'GOAL')) & (nhl_pbp['Home_Zone'] == 'Def'), 'away_shot'] = 1



nhl_pbp.loc[(nhl_pbp['home_shot'] == 0 ) & (nhl_pbp['Event'] == 'BLOCK') & (nhl_pbp['Home_Zone'] == 'Def'), 'home_shot'] = 1

nhl_pbp.loc[(nhl_pbp['away_shot'] == 0 ) & (nhl_pbp['Event'] == 'BLOCK') & (nhl_pbp['Home_Zone'] == 'Off'), 'away_shot'] = 1
nhl_pbp['faceoff'] = (np.select(condlist=[((nhl_pbp['Event'] == 'FAC') & (nhl_pbp['Ev_Zone'] == 'Neu')), ((nhl_pbp['Event'] == 'FAC') & (nhl_pbp['Ev_Zone'] == 'Off')),((nhl_pbp['Event'] == 'FAC') & (nhl_pbp['Ev_Zone'] == 'Def'))], 

           choicelist=[0,1,-1], default=np.nan))



nhl_pbp['faceoff'] = nhl_pbp['faceoff'].ffill().fillna(0)
nhl_pbp.loc[nhl_pbp['homePlayer6_id'].astype(str) == '', 'homePlayer6_id'] ='home,SH_1'

nhl_pbp.loc[nhl_pbp['homePlayer5_id'].astype(str) == '', 'homePlayer5_id'] ='home,SH_2'





nhl_pbp.loc[nhl_pbp['awayPlayer6_id'].astype(str) == '', 'awayPlayer6_id'] ='away,SH_1'

nhl_pbp.loc[nhl_pbp['awayPlayer5_id'].astype(str) == '', 'awayPlayer5_id'] ='away,SH_2'



nhl_pbp = nhl_pbp[(nhl_pbp.Home_Players >= 4) & (nhl_pbp.Away_Players >= 4)]
nhl_pbp.loc[nhl_pbp['Home_Goalie_Id'] == nhl_pbp['homePlayer6_id'], 'homePlayer6_id'] = 'goalie'

nhl_pbp.loc[nhl_pbp['Home_Goalie_Id'] == nhl_pbp['homePlayer5_id'], 'homePlayer5_id'] = 'goalie'

nhl_pbp.loc[nhl_pbp['Home_Goalie_Id'] == nhl_pbp['homePlayer4_id'], 'homePlayer4_id'] = 'goalie'



nhl_pbp.loc[nhl_pbp['Away_Goalie_Id'] == nhl_pbp['awayPlayer6_id'], 'awayPlayer6_id'] = 'goalie'

nhl_pbp.loc[nhl_pbp['Away_Goalie_Id'] == nhl_pbp['awayPlayer5_id'], 'awayPlayer5_id'] = 'goalie'

nhl_pbp.loc[nhl_pbp['Away_Goalie_Id'] == nhl_pbp['awayPlayer4_id'], 'awayPlayer4_id'] = 'goalie'
home_offense = nhl_pbp[nhl_pbp.poss_time > 0][['homePlayer1_id','homePlayer2_id','homePlayer3_id','homePlayer4_id','homePlayer5_id','homePlayer6_id',

                       'awayPlayer1_id','awayPlayer2_id','awayPlayer3_id','awayPlayer4_id','awayPlayer5_id','awayPlayer6_id','faceoff',

                                               'home_shot','poss_time']]



home_offense.columns =  home_offense.columns.str.replace('home', 'offense')

home_offense.columns =  home_offense.columns.str.replace('away', 'defense')

home_offense.columns =  home_offense.columns.str.replace('home_shot', 'shot')

home_offense['Home'] = 1



away_offense = nhl_pbp[nhl_pbp.poss_time > 0][['homePlayer1_id','homePlayer2_id','homePlayer3_id','homePlayer4_id','homePlayer5_id','homePlayer6_id',

                       'awayPlayer1_id','awayPlayer2_id','awayPlayer3_id','awayPlayer4_id','awayPlayer5_id','awayPlayer6_id','faceoff',

                                               'away_shot','poss_time']]

away_offense.columns =  away_offense.columns.str.replace('away', 'offense')

away_offense.columns =  away_offense.columns.str.replace('home', 'defense')

away_offense.columns =  away_offense.columns.str.replace('away_shot', 'shot')



away_offense['Home'] = 0

away_offense['faceoff'] = -away_offense['faceoff']



stint_df = pd.concat([home_offense,away_offense]).reset_index()

stint_df = stint_df.groupby(['offensePlayer1_id','offensePlayer2_id','offensePlayer3_id','offensePlayer4_id','offensePlayer5_id','offensePlayer6_id','defensePlayer1_id','defensePlayer2_id','defensePlayer3_id','defensePlayer4_id','defensePlayer5_id','defensePlayer6_id','faceoff']).sum().reset_index()



stint_df['faceoff_str'] = np.nan

stint_df.loc[stint_df.faceoff == 1, 'faceoff_str'] = 'Off'

stint_df.loc[stint_df.faceoff == -1, 'faceoff_str'] = 'Def'
stint_df['combined_off_player'] = stint_df[['offensePlayer1_id','offensePlayer2_id','offensePlayer3_id','offensePlayer4_id','offensePlayer5_id','offensePlayer6_id']].astype(str).values.tolist()

stint_df['combined_def_player'] = stint_df[['defensePlayer1_id','defensePlayer2_id','defensePlayer3_id','defensePlayer4_id','defensePlayer5_id','defensePlayer6_id']].astype(str).values.tolist()



off_mlb = MultiLabelBinarizer(sparse_output=True)

off_array = off_mlb.fit_transform(stint_df['combined_off_player'] )

def_mlb = MultiLabelBinarizer(sparse_output=True)

def_array = def_mlb.fit_transform(stint_df['combined_def_player'] )



stint_array = hstack([off_array,def_array*-1,csr_matrix(stint_df['faceoff'].values).transpose(),csr_matrix(stint_df['Home'].values).transpose()])

clf = RidgeCV(alphas=[1e3,1e4,1e5])

clf.fit(csr_matrix(stint_array),3600*stint_df['offense_shot']/stint_df['poss_time'],sample_weight=stint_df['poss_time']) 
player_df = pd.DataFrame({'player_id':nhl_pbp[['homePlayer1_id','homePlayer2_id','homePlayer3_id','homePlayer4_id','homePlayer5_id','homePlayer6_id', 'awayPlayer1_id','awayPlayer2_id','awayPlayer3_id','awayPlayer4_id','awayPlayer5_id','awayPlayer6_id']].values.flatten(),

                         'player_name':nhl_pbp[['homePlayer1','homePlayer2','homePlayer3','homePlayer4','homePlayer5','homePlayer6', 'awayPlayer1','awayPlayer2','awayPlayer3','awayPlayer4','awayPlayer5','awayPlayer6']].values.flatten()})

player_df = player_df.drop_duplicates().dropna()

player_df['player_id'] = player_df['player_id'].astype(str)



class_list = ['{}|{}'.format(a, 'offense')for a in off_mlb.classes_] + ['{}|{}'.format(a, 'defense')for a in def_mlb.classes_] + ['faceoff', 'home']

rapm_df = pd.DataFrame({'class':class_list,'coef':clf.coef_})

rapm_df['player_id'] = rapm_df['class'].str.split('|').str[0]

rapm_df['offense'] = rapm_df['class'].str.split('|').str[1]



rapm_df = pd.merge(rapm_df[['player_id','offense','coef']], player_df,on='player_id',how='inner').dropna()

rapm_df['player_name'] = rapm_df['player_name'].replace('', np.nan)

rapm_df = rapm_df.dropna()

rapm_df.loc[rapm_df['offense'] == 'defense', 'coef'] *= -1

rapm_df = rapm_df[(rapm_df.player_id != 'goalie') & (rapm_df.player_id != 'NA')].sort_values(by='coef',ascending=False)

rapm_df.head(10)