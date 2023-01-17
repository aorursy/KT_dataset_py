import numpy as np 
import pandas as pd
df = pd.read_csv('../input/NGS-2016-pre.csv')
vr = pd.read_csv('../input/video_review.csv')

df.Time = pd.to_datetime(df.Time, format ='%Y-%m-%d %H:%M:%S.%f')
df_list = ['Season_Year', 'GameKey', 'PlayID', 'GSISID', 'Time']
df.sort_values(df_list, inplace=True)
def calculate_speeds(df, dt=None, SI=False):
    data_selected = df[['Time', 'x','y']]
    if SI==True:
        data_selected.x = data_selected.x / 1.0936132983 # 1 m/s == 1.0936132983  yd/s
        data_selected.y = data_selected.y / 1.0936132983 # 1.0936132983 = 1 m/s equivale 1.0936132983  yd/s
    # Might have used shift pd function ?
    data_selected_diff = data_selected.diff()
    if dt==None:
        # Time is now a timedelta and need to be converted
        data_selected_diff.Time = data_selected_diff.Time.apply(lambda x: (x.total_seconds()))
        data_selected_diff['Speed'] = (data_selected_diff.x **2 + data_selected_diff.y **2).astype(np.float64).apply(np.sqrt) / data_selected_diff.Time
    else:
        # Need to be sure about the time step...
        data_selected_diff['Speed'] = (data_selected_diff.x **2 + data_selected_diff.y **2).astype(np.float64).apply(np.sqrt) / dt
    #data_selected_diff.rename(columns={'Time':'TimeDelta'}, inplace=True)
    #return data_selected_diff
    df['TimeDelta'] = data_selected_diff.Time
    df['Speed'] = data_selected_diff.Speed
    return df[1:]

def remove_wrong_values(df, tested_columns=['Season_Year', 'GameKey', 'PlayID', 'GSISID', 'TimeDelta'], cutspeed=None):
    dump = df.copy()
    colums = dump.columns
    mask = []
    for col in tested_columns:
        dump['shift_'+col] = dump[col].shift(-1)
        mask.append("( dump['shift_"+col+"'] == dump['"+col+"'])")
    mask =eval(" & ".join(mask))
    # Keep results where next rows is equally space
    dump = dump[mask]
    dump = dump[colums]
    if cutspeed!=None:
        dump = dump[dump.Speed < cutspeed]
    return dump
df_speed = calculate_speeds(df, SI=True)
df_speed.head()
vr_GameKey = vr['GameKey']
vr_PlayID = vr['PlayID']
vr_GSISID = vr['GSISID']
df_GameKey = df_speed.loc[df_speed['GameKey'] == 5]
df_GameKey_PlayID = df_GameKey.loc[df_GameKey['PlayID'] == 3129]
df_GameKey_PlayID_GSISID = df_GameKey_PlayID.loc[df_GameKey_PlayID['GSISID'] == 31057]
df_GameKey_PlayID_GSISID
df_speed_max = df_GameKey_PlayID_GSISID.loc[df_GameKey_PlayID_GSISID['Speed'] == max(df_GameKey_PlayID_GSISID['Speed'])]
df_speed_max 
# yds per seg
df_speed_line_set = df_GameKey_PlayID_GSISID.loc[df_GameKey_PlayID_GSISID['Event'] == 'line_set']
df_speed_line_set
df_speed_punt = df_GameKey_PlayID_GSISID.loc[df_GameKey_PlayID_GSISID['Event'] == 'punt']
df_speed_punt
df_speed_ball_snap = df_GameKey_PlayID_GSISID.loc[df_GameKey_PlayID_GSISID['Event'] == 'ball_snap']
df_speed_ball_snap
df_speed_punt_received = df_GameKey_PlayID_GSISID.loc[df_GameKey_PlayID_GSISID['Event'] == 'punt_received']
df_speed_punt_received
df_speed_punt_play = df_GameKey_PlayID_GSISID.loc[df_GameKey_PlayID_GSISID['Event'] == 'punt_play']
df_speed_punt_play
df_speed_penalty_flag = df_GameKey_PlayID_GSISID.loc[df_GameKey_PlayID_GSISID['Event'] == 'penalty_flag']
df_speed_penalty_flag
df_speed_play_submit = df_GameKey_PlayID_GSISID.loc[df_GameKey_PlayID_GSISID['Event'] == 'play_submit']
df_speed_play_submit