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
# read data

def read_data(inp1, inp2):

    train = pd.read_csv(inp1 + 'train.csv', dtype={'time': np.float32, 'signal': np.float32, 'open_channels':np.int32})

    test  = pd.read_csv(inp1 + 'test.csv', dtype={'time': np.float32, 'signal': np.float32})

#     sub  = pd.read_csv('/kaggle/input/liverpool-ion-switching/sample_submission.csv', dtype={'time': np.float32})

    

    Y_train_proba = np.load(inp2 + "Y_train_proba.npy")

    Y_test_proba = np.load(inp2 + "Y_test_proba.npy")

    

    for i in range(11):

        train[f"proba_{i}"] = Y_train_proba[:, i]

        test[f"proba_{i}"] = Y_test_proba[:, i]



    return train, test

from sklearn.metrics import f1_score

proba_tr, proba_te  = read_data('/kaggle/input/liverpool-ion-switching/','/kaggle/input/ion-shifted-rfc-proba/')
proba_tr.sort_values('time', ignore_index = True, inplace = True)

proba_tr_value = proba_tr[['proba_0', 'proba_1', 'proba_2','proba_3', 'proba_4', 'proba_5', 'proba_6', 'proba_7', 'proba_8','proba_9', 'proba_10']].values

proba_tr_pred = np.argmax(proba_tr_value, axis=-1)
proba_tr['pred'] = proba_tr_pred
for i in [0,1],[2,6],[3,7],[5,8],[4,9]:

    j1 = i[0] + 1

    j2 = i[1] + 1

    batch = j1; a = 500000*(batch-1); b = 500000*batch

    batch = j2; c = 500000*(batch-1); d = 500000*batch

    print(j1,j2)

    pred_temp = np.concatenate([proba_tr.pred.values[a:b], proba_tr.pred.values[c:d]]).reshape((-1,1))

    real_temp = np.concatenate([proba_tr.open_channels.values[a:b], proba_tr.open_channels.values[c:d]]).reshape((-1,1))

    #     print(len(pred_temp),len(real_temp))

    print(i);print('f1 score =',f1_score(pred_temp,real_temp,average='macro'))
f1_score(proba_tr.pred,proba_tr.open_channels,average='macro')
# submission 

proba_te.sort_values('time', ignore_index = True, inplace = True)



temp = proba_te[['proba_0','proba_1','proba_2','proba_3','proba_4','proba_5','proba_6','proba_7','proba_8','proba_9','proba_10']].values



temp_pred = pd.DataFrame(np.argmax(temp, axis=-1))

temp_pred.columns = ['open_channels']

import datetime



x = datetime.datetime.now()

x = x.strftime("%Y%m%d%H%M%S")

test_ori = pd.read_csv('/kaggle/input/liverpool-ion-switching/' + 'test.csv')

pd.concat([test_ori[['time']],temp_pred], axis = 1).to_csv('sub'+x+'.csv.gz', compression='gzip', index = False, float_format='%.4f')

pd.concat([test_ori[['time']],temp_pred], axis = 1).to_csv('sub'+x+'.csv', index = False, float_format='%.4f')