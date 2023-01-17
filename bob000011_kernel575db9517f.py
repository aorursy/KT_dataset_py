import pandas as pd

import numpy as np



import datetime

#beatiful first submission
df = pd.read_csv("/kaggle/input/sputnik/train.csv")
df = df.iloc[0:]

df['epoch'] = pd.to_datetime(df.epoch,format='%Y-%m-%dT%H:%M:%S') 

df.index  = df.epoch

df.drop('epoch', axis = 1, inplace = True)
train = df[df.type == 'train']

test = df[df.type == 'test']



# количество спутников

sat_sz = len(np.unique(list(df['sat_id'].values)))

# уберу чуть позже

sz = 24
one_sat = train[train.sat_id == 42]#take random sattelite

one_sat = one_sat.iloc[:]
# tmp_train[0:30].plot.line(y = 'y')

one_sat['minute'] = one_sat.index.minute

one_sat['second'] = one_sat.index.second

one_sat['hour'] = one_sat.index.hour

one_sat['dom'] = one_sat.index.day

one_sat['time'] = one_sat.index.time



one_sat_piv = pd.pivot_table(one_sat, values = ["x", "y", "z"], columns = "dom",index = "hour")

one_sat_piv['x'].stack().reset_index()[:96].plot(y = 0)
one_sat[:40].plot(y = ['x','y','z'])

one_sat[::24].plot(y = ['x','y','z'])
def copy (trn, tst, col):

    colmn = list(trn[col][-sz:])

    copy = colmn*int(tst.shape[0]/sz + 1)

    return copy[:tst.shape[0]]
result = pd.DataFrame()

for i in range(sat_sz):

    tmp_train = train[train.sat_id == i]

    tmp_test = test[test.sat_id == i]

    

    tmp_test['x'] = copy(tmp_train, tmp_test, 'x')

    tmp_test['y'] = copy(tmp_train, tmp_test, 'y')

    tmp_test['z'] = copy(tmp_train, tmp_test, 'z')



    result = result.append(tmp_test)

    

result
df2 = pd.DataFrame()

df2['id'] = result['id'].values

df2['error']  = np.linalg.norm(result[['x', 'y', 'z']].values - result[['x_sim', 'y_sim', 'z_sim']].values, axis=1)

test.shape
df2.index = df2['id']

df2.drop('id', axis = 1, inplace = True)
df2
df2.to_csv('subm_00.csv')