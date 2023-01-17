# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import glob
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
listNGS_CSV = glob.glob("../input/how-to-import-large-csv-files-and-save-efficiently/NGS*pkl")
import feather
# Any results you write to the current directory are saved as output.
def calculate_speeds(df, dt=None, SI=False):
    data_selected = df[['Time', 'x','y']]
    if SI==True:
        data_selected.x = data_selected.x / 1.0936132983
        data_selected.y = data_selected.y / 1.0936132983
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

def get_speed(df):
    df_with_speed = df.copy()
    date_format = '%Y-%m-%d %H:%M:%S.%f'
    sortBy = ['Season_Year', 'GameKey', 'PlayID', 'GSISID', 'Time']
    df_with_speed.Time = pd.to_datetime(df_with_speed.Time, format =date_format)
    df_with_speed.sort_values(sortBy, inplace=True)
    df_with_speed = calculate_speeds(df_with_speed, SI=True)
    cut_speed=100 / 9.58 # World record 9,857232 m/s for NFL
    df_with_speed = remove_wrong_values(df_with_speed, cutspeed=cut_speed)
    return df_with_speed

def memory(df):
    if isinstance(df,pd.DataFrame):
        value = df.memory_usage(deep=True).sum() / 1024 ** 2
    else: # we assume if not a df it's a series
        value = df.memory_usage(deep=True) / 1024 ** 2
    return value, "{:03.2f} MB".format(value)

def downCast(df, verbose=True):
    mem, _ =  memory(df)
    date_format = '%Y-%m-%d %H:%M:%S.%f'
    converted_obj = df.select_dtypes(include=['int']).astype('category')
    df[converted_obj.columns] = converted_obj
    converted_obj = df.select_dtypes(include=['float']).apply(pd.to_numeric,downcast='float')
    df[converted_obj.columns] = converted_obj
    if 'Time' in df: df.Time = pd.to_datetime(df.Time,format=date_format)
    memDown, _ = memory(df)
    if verbose: print("Gain {:03.2f} %".format(mem/memDown*100.0))
    return df

def downCastAllCSV(listBigCSV, verbose=True):
    for csvFile in listBigCSV:
        dataframe = pd.read_csv(csvFile, engine='c')
        dataframe = downCast(dataframe)
        dataframe.to_pickle(os.path.basename(csvFile[:-4]+'.pkl'))
        del dataframe
        
import inspect

def retrieve_name(var):
    callers_local_vars = inspect.currentframe().f_back.f_locals.items()
    return [var_name for var_name, var_val in callers_local_vars if var_val is var][0]

def filter_expr(key,filtered):
    return '(df["'+key+'"] == '+str(filtered)+')'

def filter_from_dict(df, filters, inplace=False):
    mask = []
    maskList=[]
    for key, filtered in filters.items():
        if type(filtered) == str:
            filtered = '"'+filtered+'"'
            mask.append(filter_expr(key,filtered))
        else:
            if type(filtered) == list:
                for fil in filtered:
                    maskList.append(filter_expr(key,fil))
                mask.append('('+'|'.join(maskList)+')')
            else:
                mask.append(filter_expr(key,filtered))
    print( 'Mask generated: ',' & '.join(mask))
    if inplace:
        df = df[(eval('&'.join(mask)))]
    else:
        return df[(eval('&'.join(mask)))]

df = pd.read_csv('../input/NFL-Punt-Analytics-Competition/NGS-2016-pre.csv')
date_format = '%Y-%m-%d %H:%M:%S.%f'
sortBy = ['Season_Year', 'GameKey', 'PlayID', 'GSISID', 'Time']
df.Time = pd.to_datetime(df.Time, format =date_format)
df.sort_values(sortBy, inplace=True)
%timeit calculate_speeds(df[:10000])
df_with_speed = calculate_speeds(df, SI=True)
df_with_speed.head()
cut_speed=100 / 9.58 # World record 9,857232 m/s for NFL
df_remove_bad_values = remove_wrong_values(df, cutspeed=cut_speed)
df_remove_bad_values.Speed.hist()
df_remove_bad_values.describe()

game_play_player = df_remove_bad_values[((df_remove_bad_values.GameKey==5)&(df_remove_bad_values.PlayID==3129)&(df_remove_bad_values.GSISID==31057))]
game_play_player.plot(x='Time',y='Speed')
game_play_player.Speed.max() *2.23694
game_play_player.Speed.mean()*2.23694
def calculate_speeds2(df, dt=None, SI=False):
    data_selected = df[['Time', 'x','y']]
    if SI==True:
        data_selected.x = data_selected.x.values / 1.0936132983
        data_selected.y = data_selected.y.values / 1.0936132983
    # Might have used shift pd function ?
    data_selected_diff = data_selected.diff()
    if dt==None:
        # Time is now a timedelta and need to be converted
        data_selected_diff.Time = data_selected_diff.Time.apply(lambda x: (x.total_seconds()))
        data_selected_diff['Speed'] = np.sqrt(data_selected_diff.x.values **2 + data_selected_diff.y.values **2) / data_selected_diff.Time.values
    else:
        # Need to be sure about the time step...
        data_selected_diff['Speed'] = (data_selected_diff.x.values **2 + data_selected_diff.y.values **2).astype(np.float64).apply(np.sqrt) / dt.values
    #data_selected_diff.rename(columns={'Time':'TimeDelta'}, inplace=True)
    #return data_selected_diff
    df['TimeDelta'] = data_selected_diff.Time
    df['Speed'] = data_selected_diff.Speed
    return df[1:]

def get_speed(df):
    df_with_speed = df.copy()
    date_format = '%Y-%m-%d %H:%M:%S.%f'
    sortBy = ['Season_Year', 'GameKey', 'PlayID', 'GSISID', 'Time']
    df_with_speed.Time = pd.to_datetime(df_with_speed.Time, format =date_format)
    df_with_speed.sort_values(sortBy, inplace=True)
    df_with_speed = calculate_speeds2(df_with_speed, SI=True)
    cut_speed=100 / 9.58 # World record 9,857232 m/s for NFL
    df_with_speed = remove_wrong_values(df_with_speed, cutspeed=cut_speed)
    return df_with_speed


df_test = get_speed(df)
df_remove_bad_values == df_test



df = get_speed(df)
