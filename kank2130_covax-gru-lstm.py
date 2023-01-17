import pandas as pd

import numpy as np

import json

import tensorflow.keras.layers as L

import keras.backend as K

import tensorflow as tf

import plotly.express as px

from sklearn.model_selection import StratifiedKFold, KFold, GroupKFold

from sklearn.cluster import KMeans

import os

from collections import Counter as count

#import cond_rnn



os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def allocate_gpu_memory(gpu_number=0):

    physical_devices = tf.config.experimental.list_physical_devices('GPU')



    if physical_devices:

        try:

            print("Found {} GPU(s)".format(len(physical_devices)))

            tf.config.set_visible_devices(physical_devices[gpu_number], 'GPU')

            tf.config.experimental.set_memory_growth(physical_devices[gpu_number], True)

            print("#{} GPU memory is allocated".format(gpu_number))

        except RuntimeError as e:

            print(e)

    else:

        print("Not enough GPU hardware devices available")

allocate_gpu_memory()



Ver='GRU_LSTM1'

debug = False
def gru_layer(hidden_dim, dropout):

    return L.Bidirectional(L.GRU(hidden_dim, dropout=dropout, return_sequences=True, kernel_initializer = 'orthogonal'))



def lstm_layer(hidden_dim, dropout):

    return L.Bidirectional(L.LSTM(hidden_dim, dropout=dropout, return_sequences=True, kernel_initializer = 'orthogonal'))



def build_model(seq_len=107, pred_len=68, dropout=0.5, embed_dim=100, hidden_dim=256, type=0):

    inputs1 = L.Input(shape=(seq_len, 6)) #[(None, 107, 7)]

    inputs2 = L.Input(shape = (1,18)) ###Number 12 needs to be changed according to the feature engineering

    inputs3 = L.Dense(6, activation = 'relu')(inputs2)

    inputs = 0.97*inputs1 + 0.03*inputs3

    

    # split categorical and numerical features and concatenate them later.

    categorical_feat_dim = 3

    categorical_fea = inputs[:, :, :categorical_feat_dim] #(Tens [(None, 107, 4)]

    numerical_fea = inputs[:, :, 3:] #(Te [(None, 107, 3)]



    embed = L.Embedding(input_dim=len(token2int), output_dim=embed_dim)(categorical_fea) #(None, 107, 4, 100)

    reshaped = tf.reshape(embed, shape=(-1, embed.shape[1],  embed.shape[2] * embed.shape[3])) #(TensorFlow [(None, 107, 400)]

    reshaped = L.concatenate([reshaped, numerical_fea], axis=2) #(None, 107, 403)

    

    if type == 0:

        hidden = gru_layer(hidden_dim, dropout)(reshaped)

        hidden = gru_layer(hidden_dim, dropout)(hidden)

    elif type == 1:

        hidden = lstm_layer(hidden_dim, dropout)(reshaped)

        hidden = gru_layer(hidden_dim, dropout)(hidden)

    elif type == 2:

        hidden = gru_layer(hidden_dim, dropout)(reshaped)

        hidden = lstm_layer(hidden_dim, dropout)(hidden)

    elif type == 3:

        hidden = lstm_layer(hidden_dim, dropout)(reshaped) #(None, 107, 512)

        hidden = lstm_layer(hidden_dim, dropout)(hidden) #(None, 107, 512)

    

    truncated = hidden[:, :pred_len] #(Te [(None, 68, 512)]

    #out1 = L.Dense(128, activation='relu')(truncated) #(None, 68, 5) ###the number 15 becomes kind of a hyperparameter here

    #out2 = L.concatenate([inputs1, truncated])

    out = L.Dense(5, activation = 'linear')(truncated)

    model = tf.keras.Model(inputs=[inputs1, inputs2], outputs=out)

    model.compile(tf.keras.optimizers.Adam(), loss=mcrmse)

    return model
token2int = {x:i for i, x in enumerate('().ACGUBEHIMSXPK')}

pred_cols = ['reactivity', 'deg_Mg_pH10', 'deg_pH10', 'deg_Mg_50C', 'deg_50C']

#pair_dict = {'A':'K','U':'K','G':'P','C':'P'}



def preprocess_inputs(df):

    #df['pairs'] = df[['sequence']].applymap(lambda seq: ('').join([pair_dict[x] for x in seq]))

    cols=['sequence', 'structure', 'predicted_loop_type']

    base_fea = np.transpose(

        np.array(

            df[cols]

            .applymap(lambda seq: [token2int[x] for x in seq])

            .values

            .tolist()

        ),

        (0, 2, 1)

    )

    bpps_sum_fea = np.array(df['bpps_sum'].to_list())[:,:,np.newaxis]

    bpps_max_fea = np.array(df['bpps_max'].to_list())[:,:,np.newaxis]

    bpps_nb_fea = np.array(df['bpps_nb'].to_list())[:,:,np.newaxis]

    return np.concatenate([base_fea,bpps_sum_fea,bpps_max_fea,bpps_nb_fea], 2)



def rmse(y_actual, y_pred):

    mse = tf.keras.losses.mean_squared_error(y_actual, y_pred)

    return K.sqrt(mse)



def mcrmse(y_actual, y_pred, num_scored=len(pred_cols)):

    score = 0

    for i in range(num_scored):

        score += rmse(y_actual[:, :, i], y_pred[:, :, i]) / num_scored

    return score
train = pd.read_json('../input/stanford-covid-vaccine/train.json', lines=True)

test = pd.read_json('../input/stanford-covid-vaccine/test.json', lines=True)
def process_inputs_2(df):

    df1 = df.copy()

    df2 = df.copy()

    df3 = df.copy()

    df4 = df.copy()

    df5 = df.copy()

    from collections import Counter as count

    bases = []

    for j in range(len(df1)):

        counts = dict(count(df1.iloc[j]['sequence']))

        bases.append((

            counts['A'] / df1.iloc[j]['seq_length'],

            counts['G'] / df1.iloc[j]['seq_length'],

            counts['C'] / df1.iloc[j]['seq_length'],

            counts['U'] / df1.iloc[j]['seq_length']

        ))



    bases = pd.DataFrame(bases, columns=['A_percent', 'G_percent', 'C_percent', 'U_percent'])

    del df1

    

    pairs = []

    all_partners = []

    for j in range(len(df2)):

        partners = [-1 for i in range(130)]

        pairs_dict = {}

        queue = []

        for i in range(0, len(df2.iloc[j]['structure'])):

            if df2.iloc[j]['structure'][i] == '(':

                queue.append(i)

            if df2.iloc[j]['structure'][i] == ')':

                first = queue.pop()

                try:

                    pairs_dict[(df2.iloc[j]['sequence'][first], df2.iloc[j]['sequence'][i])] += 1

                except:

                    pairs_dict[(df2.iloc[j]['sequence'][first], df2.iloc[j]['sequence'][i])] = 1



                partners[first] = i

                partners[i] = first



        all_partners.append(partners)



        pairs_num = 0

        pairs_unique = [('U', 'G'), ('C', 'G'), ('U', 'A'), ('G', 'C'), ('A', 'U'), ('G', 'U')]

        for item in pairs_dict:

            pairs_num += pairs_dict[item]

        add_tuple = list()

        for item in pairs_unique:

            try:

                add_tuple.append(pairs_dict[item]/pairs_num)

            except:

                add_tuple.append(0)

        pairs.append(add_tuple)



    pairs = pd.DataFrame(pairs, columns=['U-G', 'C-G', 'U-A', 'G-C', 'A-U', 'G-U'])

    del df2

    

    pairs_rate = []

    for j in range(len(df3)):

        res = dict(count(df3.iloc[j]['structure']))

        pairs_rate.append(res['('] / (df3.iloc[j]['seq_length']/2))



    pairs_rate = pd.DataFrame(pairs_rate, columns=['pairs_rate'])

    del df3

    

    loops = []

    for j in range(len(df4)):

        counts = dict(count(df4.iloc[j]['predicted_loop_type']))

        available = ['E', 'S', 'H', 'B', 'X', 'I', 'M']

        row = []

        for item in available:

            try:

                row.append(counts[item] / df4.iloc[j]['seq_length'])

            except:

                row.append(0)

        loops.append(row)



    loops = pd.DataFrame(loops, columns=available)

    del df4

    

    return pd.concat([df5, bases, pairs, loops, pairs_rate], axis=1)
train = process_inputs_2(train)

train.head()
test = process_inputs_2(test)

test.head()
# additional features



def read_bpps_sum(df):

    bpps_arr = []

    for mol_id in df.id.to_list():

        bpps_arr.append(np.load(f"../input/stanford-covid-vaccine/bpps/{mol_id}.npy").sum(axis=1))

    return bpps_arr



def read_bpps_max(df):

    bpps_arr = []

    for mol_id in df.id.to_list():

        bpps_arr.append(np.load(f"../input/stanford-covid-vaccine/bpps/{mol_id}.npy").max(axis=1))

    return bpps_arr



def read_bpps_min(df):

    bpps_arr = []

    for mol_id in df.id.to_list():

        bpps_arr.append(np.load(f"../input/stanford-covid-vaccine/bpps/{mol_id}.npy").max(axis=1))

    return bpps_arr



def read_bpps_nb(df):

    # normalized non-zero number

    # from https://www.kaggle.com/symyksr/openvaccine-deepergcn 

    bpps_nb_mean = 0.077522 # mean of bpps_nb across all training data

    bpps_nb_std = 0.08914   # std of bpps_nb across all training data

    bpps_arr = []

    for mol_id in df.id.to_list():

        bpps = np.load(f"../input/stanford-covid-vaccine/bpps/{mol_id}.npy")

        bpps_nb = (bpps > 0).sum(axis=0) / bpps.shape[0]

        bpps_nb = (bpps_nb - bpps_nb_mean) / bpps_nb_std

        bpps_arr.append(bpps_nb)

    return bpps_arr 





train['bpps_sum'] = read_bpps_sum(train)

test['bpps_sum'] = read_bpps_sum(test)

train['bpps_max'] = read_bpps_max(train)

test['bpps_max'] = read_bpps_max(test)

train['bpps_nb'] = read_bpps_nb(train)

test['bpps_nb'] = read_bpps_nb(test)
# clustering for GroupKFold

# expecting more accurate CV by putting similar RNAs into the same fold.

kmeans_model = KMeans(n_clusters=100, random_state=110).fit(preprocess_inputs(train)[:,:,0])

train['cluster_id'] = kmeans_model.labels_
aug_df = pd.read_csv('../input/augmented-data-covax-1/aug_data1.csv')

display(aug_df.head())
def aug_data(df):

    target_df = df.copy()

    new_df = aug_df[aug_df['id'].isin(target_df['id'])]

                         

    del target_df['structure']

    del target_df['predicted_loop_type']

    new_df = new_df.merge(target_df, on=['id','sequence'], how='left')



    df['cnt'] = df['id'].map(new_df[['id','cnt']].set_index('id').to_dict()['cnt'])

    df['log_gamma'] = 100

    df['score'] = 1.0

    df = df.append(new_df[df.columns]) #each id now has 2 rows

    return df

train = aug_data(train)

test = aug_data(test)
if debug:

    train = train[:200]

    test = test[:200]
def preprocess_ns(df, pred_len = 1):

    ns_columns = ['A_percent',

       'G_percent', 'C_percent', 'U_percent', 'U-G', 'C-G', 'U-A', 'G-C',

       'A-U', 'G-U', 'E', 'S', 'H', 'B', 'X', 'I', 'M', 'pairs_rate']

    z = np.array(df[ns_columns])

    b = np.repeat(z[:, np.newaxis,:], pred_len, axis=1)

    return b
preprocess_ns(train).shape
model = build_model()

model.summary()
def train_and_predict(type = 0, FOLD_N = 5):

    

    gkf = GroupKFold(n_splits=FOLD_N)



    public_df = test.query("seq_length == 107").copy()

    private_df = test.query("seq_length == 130").copy()



    public_inputs1 = preprocess_inputs(public_df)

    public_inputs2 = preprocess_ns(public_df)

    private_inputs1 = preprocess_inputs(private_df)

    private_inputs2 = preprocess_ns(private_df)





    holdouts = []

    holdout_preds = []



    for cv, (tr_idx, vl_idx) in enumerate(gkf.split(train,  train['reactivity'], train['cluster_id'])):

        trn = train.iloc[tr_idx]

        x_trn1 = preprocess_inputs(trn)

        x_trn2 = preprocess_ns(trn)        

        y_trn = np.array(trn[pred_cols].values.tolist()).transpose((0, 2, 1))

        w_trn = np.log(trn.signal_to_noise+1.1)/2.5



        val = train.iloc[vl_idx]

        x_val_all1 = preprocess_inputs(val)

        x_val_all2 = preprocess_ns(val)

        val = val[val.SN_filter == 1]

        x_val1 = preprocess_inputs(val)

        x_val2 = preprocess_ns(val)

        y_val = np.array(val[pred_cols].values.tolist()).transpose((0, 2, 1))



        model = build_model(type=type)

        model_short = build_model(seq_len=107, pred_len=107,type=type)

        model_long = build_model(seq_len=130, pred_len=130,type=type)



        history = model.fit(

            [x_trn1, x_trn2], y_trn,

            validation_data = ([x_val1, x_val2], y_val),

            batch_size=64,

            epochs=65,

            sample_weight=w_trn,

            callbacks=[

                tf.keras.callbacks.ReduceLROnPlateau(),

                tf.keras.callbacks.ModelCheckpoint(f'model{Ver}_cv{cv}.h5')

            ]

        )



        fig = px.line(

            history.history, y=['loss', 'val_loss'], 

            labels={'index': 'epoch', 'value': 'Mean Squared Error'}, 

            title='Training History')

        fig.show()



        model.load_weights(f'model{Ver}_cv{cv}.h5')

        model_short.load_weights(f'model{Ver}_cv{cv}.h5')

        model_long.load_weights(f'model{Ver}_cv{cv}.h5')



        holdouts.append(train.iloc[vl_idx])

        holdout_preds.append(model.predict([x_val_all1,x_val_all2]))

        if cv == 0:

            public_preds = model_short.predict([public_inputs1, public_inputs2])/FOLD_N

            private_preds = model_long.predict([private_inputs1, private_inputs2])/FOLD_N

        else:

            public_preds += model_short.predict([public_inputs1, public_inputs2])/FOLD_N

            private_preds += model_long.predict([private_inputs1, private_inputs2])/FOLD_N

    return holdouts, holdout_preds, public_df, public_preds, private_df, private_preds
val_df, val_preds, test_df, test_preds = [], [], [], []

if debug:

    nmodel = 1

else:

    nmodel = 4

for i in range(nmodel):

    holdouts, holdout_preds, public_df, public_preds, private_df, private_preds = train_and_predict(i)

    val_df += holdouts

    val_preds += holdout_preds

    test_df.append(public_df)

    test_df.append(private_df)

    test_preds.append(public_preds)

    test_preds.append(private_preds)
preds_ls = []

for df, preds in zip(test_df, test_preds):

    for i, uid in enumerate(df.id):

        single_pred = preds[i]

        single_df = pd.DataFrame(single_pred, columns=pred_cols)

        single_df['id_seqpos'] = [f'{uid}_{x}' for x in range(single_df.shape[0])]

        preds_ls.append(single_df)

preds_df = pd.concat(preds_ls).groupby('id_seqpos').mean().reset_index()

# .mean() is for

# 1, Predictions from multiple models

# 2, TTA (augmented test data)



preds_ls = []

for df, preds in zip(val_df, val_preds):

    for i, uid in enumerate(df.id):

        single_pred = preds[i]

        single_df = pd.DataFrame(single_pred, columns=pred_cols)

        single_df['id_seqpos'] = [f'{uid}_{x}' for x in range(single_df.shape[0])]

        single_df['SN_filter'] = df[df['id'] == uid].SN_filter.values[0]

        preds_ls.append(single_df)

holdouts_df = pd.concat(preds_ls).groupby('id_seqpos').mean().reset_index()
submission = preds_df[['id_seqpos', 'reactivity', 'deg_Mg_pH10', 'deg_pH10', 'deg_Mg_50C', 'deg_50C']]

submission.to_csv(f'submission.csv', index=False)

print(f'wrote to submission.csv')
def print_mse(prd):

    val = pd.read_json('../input/stanford-covid-vaccine/train.json', lines=True)



    val_data = []

    for mol_id in val['id'].unique():

        sample_data = val.loc[val['id'] == mol_id]

        sample_seq_length = sample_data.seq_length.values[0]

        for i in range(68):

            sample_dict = {

                           'id_seqpos' : sample_data['id'].values[0] + '_' + str(i),

                           'reactivity_gt' : sample_data['reactivity'].values[0][i],

                           'deg_Mg_pH10_gt' : sample_data['deg_Mg_pH10'].values[0][i],

                           'deg_Mg_50C_gt' : sample_data['deg_Mg_50C'].values[0][i],

                           }

            val_data.append(sample_dict)

    val_data = pd.DataFrame(val_data)

    val_data = val_data.merge(prd, on='id_seqpos')



    rmses = []

    mses = []

    for col in ['reactivity', 'deg_Mg_pH10', 'deg_Mg_50C']:

        rmse = ((val_data[col] - val_data[col+'_gt']) ** 2).mean() ** .5

        mse = ((val_data[col] - val_data[col+'_gt']) ** 2).mean()

        rmses.append(rmse)

        mses.append(mse)

        print(col, rmse, mse)

    print(np.mean(rmses), np.mean(mses))
print_mse(holdouts_df)
print_mse(holdouts_df[holdouts_df.SN_filter == 1])