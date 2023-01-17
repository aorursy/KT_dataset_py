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
df = pd.concat([pd.read_pickle('/kaggle/input/m5-simple-fe/grid_part_1.pkl'),
               # pd.read_pickle('/kaggle/input/m5-simple-fe/grid_part_2.pkl').iloc[:,2:],
                pd.read_pickle('/kaggle/input/m5-simple-fe/grid_part_3.pkl').iloc[:,2:]],axis=1)
def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns: #columns毎に処理
        col_type = df[col].dtypes
        if col_type in numerics: #numericsのデータ型の範囲内のときに処理を実行. データの最大最小値を元にデータ型を効率的なものに変更
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df
event_day = df[df.id == 'HOBBIES_1_008_CA_1_evaluation'][['d','event_name_1']][df.event_name_1 == df.event_name_1]
event_days = []
for i in range(-7,7):
    tmp = event_day.copy()
    tmp.d = tmp.d + i
    tmp.columns = ['d','event_lag_' + str(i)]
    event_days.append(tmp)
event_lag_df = df[['id','d','sales','event_name_1']]
for i in range(-7, 7):
    event_lag_df = pd.merge(event_lag_df, event_days[i], on=['d'], how='left')
event_lag_df = reduce_mem_usage(event_lag_df)
event_lag_df.to_pickle('event_lag_df.pkl')
for i in range(-7, 7):
    print(event_lag_df[['sales','event_lag_' + str(i)]].groupby(['event_lag_' + str(i)]).mean())
