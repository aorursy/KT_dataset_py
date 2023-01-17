import sys

import numpy as np

import pandas as pd

pd.options.display.float_format = '{:,.5f}'.format

from tqdm.notebook import tqdm



import skmem #utility script


Eventdict = {"huddle_start_offense":0,

"huddle_break_offense":0,

"line_set":0,

"ball_snap":1,

"pass_forward":1,

"pass_arrived":1,

"pass_outcome_incomplete":0,

"pass_outcome_caught":0,

"first_contact":1,

"out_of_bounds":0,

"man_in_motion":0,

"handoff":1,

"tackle":0,

"penalty_flag":0,

"penalty_accepted":0,

"touchdown":0,

"shift":0,

"qb_kneel":0,

"fumble":1,

"fumble_offense_recovered":0,

"lateral":1,

"penalty_declined":0,

"qb_sack":0,

"pass_shovel":1,

"pass_outcome_touchdown":0,

"run":1,

"pass_outcome_interception":1,

"qb_strip_sack":1,

"two_point_conversion":1,

"pass_tipped":1,

"fumble_defense_recovered":0,

"two_minute_warning":0,

"two_point_play":1,

"snap_direct":1,

"play_action":1,

"qb_spike":0,

"pass_lateral":1,

"touchback":0,

"timeout_tv":0,

"timeout":0,

"kickoff_play":1,

"onside_kick":1,

"kick_received":1,

"safety":0,

"field_goal_attempt":1,

"field_goal":0,

"punt_play":1,

"punt":1,

"punt_land":1,

"fair_catch":0,

"punt_downed":0,

"punt_received":1,

"punt_fake":1,

"kickoff":1,

"kickoff_land":1,

"timeout_away":0,

"punt_muffed":1,

"timeout_booth_review":0,

"field_goal_play":1,

"run_pass_option":1,

"timeout_injury":0,

"kick_recovered":0,

"extra_point_attempt":1,

"extra_point":0,

"field_goal_blocked":1,

"field_goal_missed":0,

"timeout_home":0,

"extra_point_blocked":0,

"extra_point_missed":0,

"punt_blocked":0,

"timeout_quarter":0,

"end_path":0,

"field_goal_fake":1,

"xp_fake":1,

"extra_point_fake":1,

"timeout_halftime":0,

"free_kick":1,

"free_kick_play":1,

"0_kick":1,

"drop_kick":1,

"play_submit\t":0}

inj = pd.read_csv('../input/nfl-playing-surface-analytics/InjuryRecord.csv')

id_array = inj.PlayKey.str.split('-', expand=True).to_numpy()

inj['PlayerKey'] = id_array[:,0]

inj['GameID'] = id_array[:,1]

inj['PlayKey'] = id_array[:,2]

inj = inj.dropna().astype({'PlayerKey': 'int32',

           'GameID': 'int32',

           'PlayKey': 'int32'})




csize = 2_000_000 #set this to fit your situation

samplerate = 0.01

chunker = pd.read_csv('../input/nfl-playing-surface-analytics/PlayerTrackData.csv',

                      chunksize=csize)

track_list = []

player_list = []

inj_list = []

mr = skmem.MemReducer()

maxdis = 1

i = 0

for chunk in tqdm(chunker, total = int(80_000_000/csize)):

    chunk['PlayKey'] = chunk.PlayKey.fillna('0-0-0')

    id_array = chunk.PlayKey.str.split('-', expand=True).to_numpy()

    chunk['PlayerKey'] = id_array[:,0].astype(int)

    chunk['GameID'] = id_array[:,1].astype(int)

    chunk['PlayKey'] = id_array[:,2].astype(int)

    chunk = chunk.astype({'PlayerKey': 'int32',

           'GameID': 'int32',

           'PlayKey': 'int32'})

    chunk = chunk.replace({"event": Eventdict})

    chunk['event'] = chunk.event.ffill().fillna(0)

    chunk['dY'] = np.cos(chunk['dir'].shift(1) * (np.pi/180)) 

    chunk['dX'] = np.sin(chunk['dir'].shift(1) * (np.pi/180)) 

    chunk['Pred_X'] = chunk.dX * chunk.dis.shift(1) + chunk.x.shift(1)

    chunk['Pred_Y'] = chunk.dY *chunk.dis.shift(1) + chunk.y.shift(1)

    chunk['Acc_X'] = chunk.Pred_X - chunk.x

    chunk['Acc_Y'] = chunk.Pred_Y - chunk.y

    chunk['Total_Acc'] = (chunk.Acc_X**2 + chunk.Acc_Y**2)*0.5

    chunk['Angle'] = chunk['o']-chunk['dir']

    chunk['Angle'] = np.where(chunk['Angle']>180,360-chunk['Angle'],chunk['Angle'])

    chunk['Angle'] = np.where(chunk['Angle']<-180,360+chunk['Angle'],chunk['Angle'])

    chunk = chunk[(abs(chunk.x - chunk.x.shift(1))<maxdis) &  (abs(chunk.y - chunk.y.shift(1))<maxdis)].copy()

    chunk = chunk.reset_index()

    for lag in range(1,10):

        if lag>1:

            chunk['Delta_Dis_' + str(lag)] = chunk.dis.shift(lag) - chunk.dis.shift(lag-1)

            chunk['Delta_Dir_' + str(lag)] = chunk.dir.shift(lag) - chunk.dir.shift(lag-1)

            chunk['Delta_Angle_' + str(lag)] = chunk['Angle'].shift(lag) - chunk['Angle'].shift(lag-1)

            chunk['Delta_O_' + str(lag)] = chunk.o.shift(lag) - chunk.o.shift(lag-1)

            chunk['Delta_Total_Acc_' + str(lag)] = chunk.Total_Acc.shift(lag) - chunk.Total_Acc.shift(lag-1)

            #Adjust change in direction

            chunk['Delta_Dir_' + str(lag)] = np.where(chunk['Delta_Dir_' + str(lag)]>180,360-chunk['Delta_Dir_' + str(lag)],chunk['Delta_Dir_' + str(lag)])

            chunk['Delta_Dir_' + str(lag)] = np.where(chunk['Delta_Dir_' + str(lag)]<-180,360+chunk['Delta_Dir_' + str(lag)],chunk['Delta_Dir_' + str(lag)])

            chunk['Delta_Angle_' + str(lag)] = np.where(chunk['Delta_Angle_' + str(lag)]>180,360-chunk['Delta_Angle_' + str(lag)],chunk['Delta_Angle_' + str(lag)])

            chunk['Delta_Angle_' + str(lag)] = np.where(chunk['Delta_Angle_' + str(lag)]<-180,360+chunk['Delta_Angle_' + str(lag)],chunk['Delta_Angle_' + str(lag)])

            chunk['Delta_O_' + str(lag)] = np.where(chunk['Delta_O_' + str(lag)]>180,360-chunk['Delta_O_' + str(lag)],chunk['Delta_O_' + str(lag)])

            chunk['Delta_O_' + str(lag)] = np.where(chunk['Delta_O_' + str(lag)]<-180,360+chunk['Delta_O_' + str(lag)],chunk['Delta_O_' + str(lag)])

            #Kludgy

            chunk['abs_Delta_Dis_'+ str(lag)] = abs(chunk['Delta_Dis_' + str(lag)])

            chunk['abs_Delta_Dir_'+ str(lag)] = abs(chunk['Delta_Dir_' + str(lag)])

            chunk['abs_Delta_Angle_'+ str(lag)] = abs(chunk['Delta_Angle_' + str(lag)])

            chunk['abs_Delta_O_'+ str(lag)] = abs(chunk['Delta_O_' + str(lag)])

            chunk['abs_Delta_Total_Acc_'+ str(lag)] = abs(chunk['Delta_Total_Acc_' + str(lag)])

            

    for lag in [5,10]:



        chunk['Rolling_'+str(lag)+'_Dis_std']=chunk['Delta_Dis_2'].rolling(lag).std().reset_index(drop=True)

        chunk['Rolling_'+str(lag)+'_Dir_std']=chunk['Delta_Dir_2'].rolling(lag).std().reset_index(drop=True)

        chunk['Rolling_'+str(lag)+'_Angle_std']=chunk['Delta_Angle_2'].rolling(lag).std().reset_index(drop=True)

        chunk['Rolling_'+str(lag)+'_O_std']=chunk['Delta_O_2'].rolling(lag).std().reset_index(drop=True)

        chunk['Rolling_'+str(lag)+'_Total_Acc_std']=chunk['Delta_Total_Acc_2'].rolling(lag).std().reset_index(drop=True)

        chunk['Rolling_abs_'+str(lag)+'_Dis_mean']=abs(chunk['Delta_Dis_2']).rolling(lag).mean().reset_index(drop=True)

        chunk['Rolling_abs_'+str(lag)+'_Dir_mean']=abs(chunk['Delta_Dir_2']).rolling(lag).mean().reset_index(drop=True)

        chunk['Rolling_abs_'+str(lag)+'_Angle_mean']=abs(chunk['Delta_Angle_2']).rolling(lag).mean().reset_index(drop=True)

        chunk['Rolling_abs_'+str(lag)+'_O_mean']=abs(chunk['Delta_O_2']).rolling(lag).mean().reset_index(drop=True)

        chunk['Rolling_abs_'+str(lag)+'_Total_Acc_mean']=abs(chunk['Delta_Total_Acc_2']).rolling(lag).mean().reset_index(drop=True)

    

        cols = [c for c in chunk.columns if c.lower()[5:] != 'Delta']

        chunk = chunk[cols]



    chunk = chunk.replace([np.inf,-np.inf],np.nan).fillna(0)  

    #floaters = chunk.select_dtypes('float').columns.tolist()

    #chunk = mr.fit_transform(chunk, float_cols=floaters) #float downcast is optional

    chunk = chunk[chunk['event']==1]

    chunk= chunk.drop(columns=['event'])

    injchunk = inj.merge(chunk, on=['PlayerKey','GameID','PlayKey'])

    player_list.append(chunk[chunk['PlayerKey']==43483])

    track_list.append(chunk.sample(frac=samplerate))

    inj_list.append(injchunk)
tracks = pd.concat(track_list)

tracks.to_parquet('track.parq')

player = pd.concat(player_list)

player.to_parquet('oneplayer.parq')

injtrack = pd.concat(inj_list)

injtrack.to_parquet('injtrack.parq')