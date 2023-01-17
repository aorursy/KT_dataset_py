# https://docs.google.com/presentation/d/1Jg5trwTuV4zuUamQunIpompl51hoUvEFE9Xk9od-bTU/edit#slide=id.gc6fa3c898_0_0
!pip install -I tensorflow-gpu==1.14
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

from sklearn.preprocessing import StandardScaler

from keras.utils import to_categorical

from keras.models import Model, Sequential, load_model

from keras.layers import CuDNNLSTM, Dense, Input, Dropout, Activation, Concatenate

from keras.optimizers import Adam



from keras.callbacks import TensorBoard, ModelCheckpoint

from keras.utils import multi_gpu_model

from tqdm import tqdm

from scipy.special import softmax

from sklearn.model_selection import train_test_split




demo = pd.read_csv("../input/tj19data/demo.csv")

demo = demo.fillna(0)

demo['c1'] = demo['c1'].astype('int')

test_set = pd.read_csv("../input/tj19data/test.csv")

train_set = pd.read_csv("../input/tj19data/train.csv")

txn = pd.read_csv("../input/tj19data/txn.csv")

_txn = txn.copy()

txn.rename({

    'n3':'date_idx'

}, axis=1, inplace=True)

txn.drop('t0', axis=1, inplace=True)

scaler = StandardScaler()
class_weights = [1.09518711e+01, 3.07683547e+00, 3.58078232e-01, 2.61727260e-01,

       4.38833908e-01, 1.35073077e+03, 6.72563039e+00, 9.89546351e+00,

       3.99230769e+00, 1.03902367e+01, 4.49295078e-01, 4.26546559e+01,

       1.18329459e+00]
id_ccno = txn[['id', 'old_cc_no']].drop_duplicates()

train_id_ccno = pd.merge(id_ccno, train_set, on='id')



txn_n = txn[['id', 'old_cc_no', 'date_idx', 'n4', 'n5', 'n6', 'n7']].copy()

txn_n['count'] = 1



LENGTH_SEQ = 365

LENGTH_SEQ = 53



txn_n['date_idx'] = txn_n['date_idx'] // 7 + 1 # WEEKLY

txn_n = txn_n.groupby(['id', 'old_cc_no', 'date_idx']).sum()



scaled_txn_n = txn_n

txn_scaler = StandardScaler()

scaled_txn_n[['n4', 'n5', 'n6', 'n7', 'count']] = txn_scaler.fit_transform(txn_n)

scaled_txn_n = scaled_txn_n.reset_index().set_index('old_cc_no')
def prepare_demo_data(ids, is_test=False):

    filtered_demo = demo.set_index('id').loc[ids]

    filtered_txn = txn.set_index('id').loc[ids]

    sum_amount = filtered_txn.reset_index()[['id', 'n6']].groupby('id').sum().to_numpy()

    old_cc_label_onehot = to_categorical(filtered_txn.groupby('id')['old_cc_label'].first(), 14)

    demo_data = filtered_demo[['n0', 'n1', 'n2']].to_numpy()

    job_onehot = to_categorical(filtered_demo['c1'].to_numpy(), 14)

    x = np.concatenate((job_onehot, old_cc_label_onehot, demo_data, sum_amount), axis=1)

    

    if is_test:

        return x

    

    x[:, 28:] = scaler.fit_transform(x[:, 28:])

    

    return x, to_categorical(train_set.set_index('id').loc[ids]['label'].to_numpy(), 13)

i = 0

def preprocess_credit_class(old_cc_no_ids):



    txn_joined = _txn.drop_duplicates(subset='old_cc_no').set_index('old_cc_no')

    txn_joined = txn_joined.loc[old_cc_no_ids]



    txn_joined = txn_joined.drop(['n3', 'n4', 'n5', 'n6', 'n7', 't0'], axis=1)

    txn_joined = txn_joined.reset_index()

    

    _x = txn_joined[['old_cc_no','old_cc_label', 'c5', 'c6', 'c7']]

#     return _x, txn_joined

    global i

    i = 0



    dfs = []

    itera = _x.itertuples()

    for (_, old_cc_no, old_cc_label, c5, c6, c7) in itera:

        i += 1

        if i % 100 == 0:

            print(i)

        old_cc_label = to_categorical(old_cc_label, 13)

        c5 = to_categorical(c5, 100)

        c6 = to_categorical(c6, 79)

        c7 = to_categorical(c7, 95)

        onehot = np.concatenate((old_cc_label, c5, c6, c7))



        df = pd.DataFrame({

            'old_cc_no': old_cc_no,

        }, index=[old_cc_no])

        df['onehot'] = [onehot]

        dfs.append(df)

    return pd.concat(dfs,ignore_index=False)



padding_value = txn_scaler.transform([[0, 0, 0, 0, 0]])

def create_seq_from_group(group):

    seq = np.ones((LENGTH_SEQ, 5)) * padding_value

    group = group.sort_values('date_idx')

    for (_, _, date_idx, n4, n5, n6, n7, count) in group.itertuples():

        seq[int(date_idx) - 1] = [n4, n5, n6, n7, count]

    return seq



def create_seq(txn):

    grouped = txn.groupby(['old_cc_no'])

    seqs = []

    for name, group in tqdm(grouped):

        seq = create_seq_from_group(group)

        seqs.append(seq)

    seqs = np.array(seqs)

    return seqs



def create_seq_dataframe(old_cc_nos):



    txn = scaled_txn_n.loc[old_cc_nos]

    txn = txn.sort_values(['old_cc_no', 'date_idx'])

    grouped = txn.groupby(['old_cc_no'])

    output_df = pd.DataFrame(columns=['old_cc_no', 'seq'])

    dfs = []

    for old_cc_no, group in tqdm(grouped):

        seq = create_seq_from_group(group)

#         output_df = output_df.append({

#             'old_cc_no': old_cc_no,

#             'seq': seq

#         }, ignore_index=True, )

        df = pd.DataFrame({

            'old_cc_no': old_cc_no,

        }, index=[old_cc_no])

        df['seq'] = [seq]

        dfs.append(df)

    

#     return output_df

    return pd.concat(dfs, ignore_index=False)



def create_test_dataframe(ids):

#     ids = test_set.iloc[:500]['id']

    test_df = pd.merge(id_ccno, ids, on='id')



    cc_seq = create_seq_dataframe(test_df['old_cc_no'])

    cc_class = preprocess_credit_class(test_df['old_cc_no'])



    test_df = pd.merge(test_df, cc_seq, on='old_cc_no')

    test_df = pd.merge(test_df, cc_class, on='old_cc_no')

    return test_df



batch_size = 32

def create_data(ids, is_test=False):

    

    train_old_cc_nos = pd.merge(train_id_ccno, ids, on='id')['old_cc_no']



    train_txn = scaled_txn_n.loc[train_old_cc_nos]

    train_txn = train_txn.sort_values(['old_cc_no', 'date_idx'])



    seqs = create_seq(train_txn)

#     return seqs

    onehot = preprocess_credit_class(train_old_cc_nos.to_list())

    onehot = np.array([v for v in onehot['onehot'].to_numpy()])

    



    demo_x, _ = prepare_demo_data(ids, False)

    

    if is_test:

        return onehot, demo_x, seqs

    else:

        labels = train_id_ccno.set_index('old_cc_no').loc[train_old_cc_nos].sort_values('old_cc_no')['label']

        labels = np.array([to_categorical(label, 13) if not np.isnan(label) else to_categorical(0, 13) for label in labels])

        return onehot, demo_x, seqs, labels

    
model = load_model('../input/weight/ep020-val_loss1.5714234885013298-val_acc0.4072950482368469.hdf5')
ids = test_set['id']

cc_nos = pd.merge(id_ccno, ids, on='id')['old_cc_no']

test_df = create_test_dataframe(ids)

# _x, joined = preprocess_credit_class(cc_nos)
seqs = test_df['seq'].to_numpy()

seqs = np.array([s for s in seqs])

onehot = test_df['onehot'].to_numpy()

onehot = np.array([v for v in onehot])
prob = model.predict([onehot, seqs], verbose=1)
output_test_df = test_df
output_test_df['prob'] = prob.tolist()
def ensemble(group):

#     group_prob = np.array(group['prob'].tolist()).sum(axis=0)

#     group_prob = softmax(group_prob)

    group_prob = np.array(group['prob'].tolist())[0]

    return group_prob
grouped = output_test_df.groupby('id')

dfs = []

for name, group in tqdm(grouped):

    class_prob = ensemble(group)

    data = {f'class{class_index}':p for class_index, p in enumerate(class_prob)}

    data['id'] = name

    dfs.append(pd.DataFrame(data, index=[name]))

output = pd.concat(dfs, ).set_index('id')
output.to_csv('output.csv', float_format='%.4f')