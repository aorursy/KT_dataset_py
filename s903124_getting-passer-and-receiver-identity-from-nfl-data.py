import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

pd.options.display.max_columns = 999

import codecs

decode_hex = codecs.getdecoder("hex_codec")
fastr_18 = pd.read_csv('/kaggle/input/nflfastr/play_by_play_2018.csv')
fastr_18.head()
roster_data =pd.read_csv('https://raw.githubusercontent.com/guga31bb/nflfastR-data/master/roster-data/roster.csv')

roster_data = roster_data[['teamPlayers.gsisId','teamPlayers.nflId']].drop_duplicates().dropna()
fastr_18['passer_gsis_id'] =  (fastr_18['passer_id'].str.split('-').str[2].str[-2:] + fastr_18['passer_id'].str.split('-').str[3] + fastr_18['passer_id'].str.split('-').str[4].str[:4]).apply(lambda x: decode_hex(x)[0].decode("utf-8") if(pd.notnull(x))  else x )

fastr_18['passer_gsis_id'] ='00-' +  fastr_18.loc[~pd.isna(fastr_18['passer_gsis_id']), 'passer_gsis_id'].astype(str).str.zfill(7)
fastr_18['receiver_gsis_id'] =  (fastr_18['receiver_id'].str.split('-').str[2].str[-2:] + fastr_18['receiver_id'].str.split('-').str[3] + fastr_18['receiver_id'].str.split('-').str[4].str[:4]).apply(lambda x: decode_hex(x)[0].decode("utf-8") if(pd.notnull(x))  else x )

fastr_18['receiver_gsis_id'] ='00-' +  fastr_18.loc[~pd.isna(fastr_18['receiver_gsis_id']), 'receiver_gsis_id'].astype(str).str.zfill(7)
week1_df = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2021/week1.csv')
fastr_18['passer_nflId'] =  pd.merge(fastr_18[['passer_gsis_id' ]],roster_data[['teamPlayers.nflId','teamPlayers.gsisId']].dropna(),left_on='passer_gsis_id',right_on='teamPlayers.gsisId',how='left')['teamPlayers.nflId']

fastr_18['receiver_nflId'] =  pd.merge(fastr_18[['receiver_gsis_id' ]],roster_data[['teamPlayers.nflId','teamPlayers.gsisId']].dropna(),left_on='receiver_gsis_id',right_on='teamPlayers.gsisId',how='left')['teamPlayers.nflId']
week1_df = pd.merge(week1_df,fastr_18[['play_id', 'old_game_id','passer_nflId','receiver_nflId']],left_on=['gameId','playId'],right_on=['old_game_id','play_id'],how='left')
week1_df['IsPasser'] = week1_df['nflId'] == week1_df['passer_nflId']

week1_df['IsReceiver'] = week1_df['nflId'] == week1_df['receiver_nflId']
week1_df.head()