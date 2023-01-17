import numpy as np

import pandas as pd
%time df_train = pd.read_csv('../input/train.csv', dtype = {'acoustic_data': np.int16, 'time_to_failure': np.float32} ) # float32 is enough :)
ttf = df_train['time_to_failure'].values

index_start = np.nonzero(np.diff(ttf) > 0)[0] + 1

index_start = np.insert(index_start, 0, 0) # insert 1st period start manually

print(index_start)
chunk_length = np.diff(np.append(index_start, df_train.shape[0]))

print(chunk_length)
t_start = ttf[index_start]

t_end = ttf[index_start + chunk_length - 1]

print('t_start =', t_start)

print('t_end   =', t_end)
dt_step = (t_start - t_end) / chunk_length

dt_step
linsp_diff_max = []

for i in range(len(index_start)):

    ttf_orig  = ttf[index_start[i] : index_start[i] + chunk_length[i]]

    ttf_linsp = np.linspace(t_start[i], t_end[i], chunk_length[i])

    linsp_diff_max.append(np.abs(ttf_orig - ttf_linsp).max())

linsp_diff_max = np.array(linsp_diff_max)

print(linsp_diff_max)

print('Max error =', linsp_diff_max.max())
df_train_info = pd.DataFrame({

    'index_start':index_start,

    'chunk_length':chunk_length,

    't_start':t_start,

    't_end':t_end,

    'dt_step':dt_step,

    'linsp_diff_max':linsp_diff_max

})

df_train_info.to_csv('train_info.csv', index=False)

df_train_info
%time np.savez_compressed('train_acoustic_data.npz', acoustic_data=df_train['acoustic_data'].values)

%ls -lh
%%time

df_train_info = pd.read_csv('train_info.csv')

ac_data = np.load('train_acoustic_data.npz')['acoustic_data']



def get_quake_period(i):

    index_start, chunk_length = df_train_info['index_start'][i], df_train_info['chunk_length'][i]

    t_start, t_end = df_train_info['t_start'][i], df_train_info['t_end'][i]

    ac_data_period = ac_data[ index_start : index_start + chunk_length ]

    ttf_data_period = np.linspace(t_start, t_end, chunk_length, dtype=np.float32)

    return ac_data_period, ttf_data_period
get_quake_period(3)
df_ssub = pd.read_csv('../input/sample_submission.csv')

df_ssub.tail()
ac_data = []



for fname in df_ssub['seg_id'].values:

    ac_data.append(pd.read_csv('../input/test/' + fname + '.csv').acoustic_data.values.astype(np.int16))

    

ac_data = np.array(ac_data)



np.savez_compressed('test_acoustic_data.npz', acoustic_data=ac_data)



print('test data shape :', ac_data.shape)