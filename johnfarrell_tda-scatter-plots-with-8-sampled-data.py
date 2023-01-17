%matplotlib inline
import os
import gc
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

#sample_data = pd.read_csv( '../input/train_sample.csv' )
dtype = {
        'ip'               : 'uint32',
        'app'              : 'uint16',
        'device'           : 'uint16',
        'os'               : 'uint16',
        'channel'          : 'uint16',
        'is_attributed'    : 'uint8',
        #'click_id'        : 'uint32',
        'click_hour'       : 'uint8',
        'click_minute'     : 'uint8',
        'click_second'     : 'uint8'        
        }
DATA_DIR = '../input/'
target_col = 'is_attributed'

def read_by_chunk(path, chunksize=10**6, subsample=0.2, read_csv_params={}):
    df = pd.DataFrame()
    for chunk in tqdm(pd.read_csv(path, chunksize=chunksize, **read_csv_params)):
        if 0.0<subsample<1.0:
            size = int(subsample*len(chunk))
            chunk = chunk.sample(frac=1).iloc[:size]
        df = pd.concat([df, chunk], ignore_index=True, axis=0)
    return df
key_li = ['ip', 'app', 'device', 'os', 'channel', 'click_time', 'is_attributed']
read_csv_params = {'dtype':dtype, 'usecols':key_li}
subsample = 0.08
df_train = read_by_chunk(DATA_DIR+'train.csv', subsample=subsample, read_csv_params=read_csv_params)
print(df_train.shape)
df_train.head()
key_li = ['ip', 'app', 'device', 'os', 'channel', 'click_time']
read_csv_params = {'dtype':dtype, 'usecols':key_li}
df_test = read_by_chunk(DATA_DIR+'test.csv', subsample=subsample, read_csv_params=read_csv_params)
print(df_test.shape)
df_test.head()
df_test[target_col] = -1
df_test = df_test[key_li+[target_col]]
sample_data = pd.concat([df_train, df_test], axis=0, ignore_index=True)
del df_train, df_test; gc.collect()
print(sample_data.shape)
sample_data.head()
sample_data[ 'click_time' ] = pd.to_datetime( sample_data[ 'click_time' ] )
clicktime = np.array( [ t.timestamp() for t in sample_data[ 'click_time' ] ] )
#clicktime -= np.min( clicktime )
clicktime -= pd.to_datetime( '2017-11-07 00:00:00' ).timestamp()

#idx_download = sample_data.index[ sample_data[ 'is_attributed' ] == 1 ]
#idx_test = sample_data.index[ sample_data[ 'is_attributed' ] == -1 ]
sample_data[ 'time' ] = clicktime.astype( int )
sample_data.sort_values( by = [ 'time' ], inplace = True )
sample_data.reset_index( drop = True, inplace = True )

del clicktime; gc.collect()
def plot_attribute_vs_time( sdata, attr_name ):
    idx = sdata['is_attributed'] == 1
    idx_test = sdata['is_attributed'] == -1
    plt.figure(figsize=(128, 48))
    plt.plot( sdata[ 'time' ] / 3600, sdata[ attr_name ], 'b.', alpha = 0.1, label = 'is_attributed = 0' )
    plt.plot( sdata[ 'time' ][idx] / 3600, sdata[ attr_name ][idx], 'ro', alpha = 0.6, label = 'is_attributed = 1' )
    plt.plot( sdata[ 'time' ][idx_test] / 3600, sdata[ attr_name ][idx_test], 'g.', alpha = 0.6, label = 'test' )
    plt.xlabel( 'Time [hours]', fontsize = 16 )
    plt.ylabel( attr_name, fontsize = 16 )
    leg  = plt.legend( fontsize = 16 )
    for l in leg.get_lines():
        l.set_alpha( 1 )
        l.set_marker( '.' )
    plt.show()
for c in ['ip', 'app', 'device', 'os', 'channel']:
    print('plotting', c, '...')
    plot_attribute_vs_time(sample_data, c)
CUT = {}
CUT['ip'] = sample_data['ip'][sample_data['is_attributed']==-1].max()+1
CUT['app'] = 300
CUT['device'] = 500
CUT['os'] = 200
CUT['channel'] = sample_data['channel'].max()+1 #NO CUT
for c in ['ip', 'app', 'device', 'os']:
    print('plotting cut', c, '...')
    plot_attribute_vs_time(sample_data[sample_data[c]<CUT[c]], c)