# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import json
import tensorflow as tf
from matplotlib import pyplot as plt
os.chdir('/kaggle/')
os.getcwd()
train_data = pd.read_json('/kaggle/input/stanford-covid-vaccine/train.json', lines = True)
test_data = pd.read_json('/kaggle/input/stanford-covid-vaccine/test.json', lines = True)
submission_format = pd.read_csv('/kaggle/input/stanford-covid-vaccine/sample_submission.csv', encoding = 'utf-8-sig')
train_data.head()
train_data.shape
train_data.groupby(['SN_filter']).size()
test_data.head()
test_data.shape
submission_format.head()
print(train_data.shape)
print(test_data.shape)
print(submission_format.shape)
print('Training data:\n',train_data['seq_scored'].value_counts())
print('Test data:\n',test_data['seq_scored'].value_counts())
len(train_data['reactivity'].iloc[0])
len(train_data['sequence'].iloc[0])
#Checking if error is negative in any cell
flag = False
for i in range(0,len(train_data)):
    if(([x<0 for x in train_data['reactivity_error'].iloc[i]].count(True) > 0) |
       ([x<0 for x in train_data['deg_error_Mg_pH10'].iloc[i]].count(True) > 0) |
       ([x<0 for x in train_data['deg_error_pH10'].iloc[i]].count(True) > 0) |
       ([x<0 for x in train_data['deg_error_Mg_50C'].iloc[i]].count(True) > 0) |
       ([x<0 for x in train_data['deg_error_50C'].iloc[i]].count(True) > 0)):
        flag = True
print(flag)
#Checking if result is negative in any cell
min_reactivity_value = min(train_data['reactivity'].iloc[0])
min_deg_Mg_pH10_value = min(train_data['deg_Mg_pH10'].iloc[0])
min_deg_pH10_value = min(train_data['deg_pH10'].iloc[0])
min_deg_Mg_50C_value = min(train_data['deg_Mg_50C'].iloc[0])
min_deg_Mg_50C_value = min(train_data['deg_50C'].iloc[0])

for i in range(0,len(train_data)):
    if(min(train_data['reactivity'].iloc[i]) < min_reactivity_value):
        min_reactivity_value = min(train_data['reactivity'].iloc[i])   

    if(min(train_data['deg_Mg_pH10'].iloc[i]) < min_deg_Mg_pH10_value):
        min_deg_Mg_pH10_value = min(train_data['deg_Mg_pH10'].iloc[i])

    if(min(train_data['deg_pH10'].iloc[i]) < min_deg_pH10_value):
        min_deg_pH10_value = min(train_data['deg_pH10'].iloc[i])

    if(min(train_data['deg_Mg_50C'].iloc[i]) < min_deg_Mg_50C_value):
        min_deg_Mg_50C_value = min(train_data['deg_Mg_50C'].iloc[i])

    if(min(train_data['deg_50C'].iloc[i]) < min_deg_Mg_50C_value):
        min_deg_Mg_50C_value = min(train_data['deg_50C'].iloc[i])
        
print(min_reactivity_value, min_deg_Mg_pH10_value, min_deg_pH10_value, min_deg_Mg_50C_value, min_deg_Mg_50C_value)
train_data.head()
# # subtracting errors from target cols
# for i in range(0,len(train_data)):
#     num_time_steps = len(train_data['reactivity'].iloc[i])
#     for j in range(num_time_steps):
#         train_data['reactivity'][i][j] =  train_data['reactivity'][i][j] - train_data['reactivity_error'][i][j]
#         train_data['deg_Mg_pH10'][i][j] =  train_data['deg_Mg_pH10'][i][j] - train_data['deg_error_Mg_pH10'][i][j]
#         train_data['deg_pH10'][i][j] =  train_data['deg_pH10'][i][j] - train_data['deg_error_pH10'][i][j]
#         train_data['deg_Mg_50C'][i][j] =  train_data['deg_Mg_50C'][i][j] - train_data['deg_error_Mg_50C'][i][j]
#         train_data['deg_50C'][i][j] =  train_data['deg_50C'][i][j] - train_data['deg_error_50C'][i][j]
train_data.columns
# For embedding layer
token2int = {x:i for i, x in enumerate('().ACGUBEHIMSX')}
target_cols = ['reactivity', 'deg_Mg_pH10', 'deg_pH10', 'deg_Mg_50C', 'deg_50C']
token2int
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

def read_bpps_nb(df):
    #mean and std from https://www.kaggle.com/symyksr/openvaccine-deepergcn 
    bpps_nb_mean = 0.077522
    bpps_nb_std = 0.08914
    bpps_arr = []
    for mol_id in df.id.to_list():
        bpps = np.load(f"../input/stanford-covid-vaccine/bpps/{mol_id}.npy")
        bpps_nb = (bpps > 0).sum(axis=0) / bpps.shape[0]
        bpps_nb = (bpps_nb - bpps_nb_mean) / bpps_nb_std
        bpps_arr.append(bpps_nb)
    return bpps_arr 

os.chdir("/kaggle/working/")
train_data['bpps_sum'] = read_bpps_sum(train_data)
test_data['bpps_sum'] = read_bpps_sum(test_data)
train_data['bpps_max'] = read_bpps_max(train_data)
test_data['bpps_max'] = read_bpps_max(test_data)
train_data['bpps_nb'] = read_bpps_nb(train_data)
test_data['bpps_nb'] = read_bpps_nb(test_data)

#sanity check
train_data.head()
import plotly.express as px
from collections import Counter as count

def get_bases(data):
    bases = []

    for j in range(len(data)):
        counts = dict(count(data.iloc[j]['sequence']))
        bases.append((
            counts['A'] / 107,
            counts['G'] / 107,
            counts['C'] / 107,
            counts['U'] / 107
        ))

    bases = pd.DataFrame(bases, columns=['A_percent', 'G_percent', 'C_percent', 'U_percent'])
    return bases
def get_pairs_rate(data):
    pairs_rate = []

    for j in range(len(data)):
        res = dict(count(data.iloc[j]['structure']))
        pairs_rate.append(res['('] / 53.5)

    pairs_rate = pd.DataFrame(pairs_rate, columns=['pairs_rate'])
    return pairs_rate
def get_pairs(data):
    pairs = []
    all_partners = []
    for j in range(len(data)):
        partners = [-1 for i in range(130)]
        pairs_dict = {}
        queue = []
        for i in range(0, len(data.iloc[j]['structure'])):
            if data.iloc[j]['structure'][i] == '(':
                queue.append(i)
            if data.iloc[j]['structure'][i] == ')':
                first = queue.pop()
                try:
                    pairs_dict[(data.iloc[j]['sequence'][first], data.iloc[j]['sequence'][i])] += 1
                except:
                    pairs_dict[(data.iloc[j]['sequence'][first], data.iloc[j]['sequence'][i])] = 1

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
    return pairs
def get_loops(data):
    loops = []
    for j in range(len(data)):
        counts = dict(count(data.iloc[j]['predicted_loop_type']))
        available = ['E', 'S', 'H', 'B', 'X', 'I', 'M']
        row = []
        for item in available:
            try:
                row.append(counts[item] / 107)
            except:
                row.append(0)
        loops.append(row)

    loops = pd.DataFrame(loops, columns=available)
    return loops
from tqdm.notebook import tqdm

def get_structure_adj(train):
    ## get adjacent matrix from structure sequence
    
    ## here I calculate adjacent matrix of each base pair, 
    ## but eventually ignore difference of base pair and integrate into one matrix
    Ss = []
    for i in tqdm(range(len(train))):
        seq_length = train["seq_length"].iloc[i]
        structure = train["structure"].iloc[i]
        sequence = train["sequence"].iloc[i]

        cue = []
        a_structures = {
            ("A", "U") : np.zeros([seq_length, seq_length]),
            ("C", "G") : np.zeros([seq_length, seq_length]),
            ("U", "G") : np.zeros([seq_length, seq_length]),
            ("U", "A") : np.zeros([seq_length, seq_length]),
            ("G", "C") : np.zeros([seq_length, seq_length]),
            ("G", "U") : np.zeros([seq_length, seq_length]),
        }
        a_structure = np.zeros([seq_length, seq_length])
        for j in range(seq_length):
            if structure[j] == "(":
                cue.append(j)
            elif structure[j] == ")":
                start = cue.pop()
#                 a_structure[start, i] = 1
#                 a_structure[i, start] = 1
                a_structures[(sequence[start], sequence[j])][start, j] = 1
                a_structures[(sequence[j], sequence[start])][j, start] = 1
        
        a_strc = np.stack([a for a in a_structures.values()], axis = 2)
        a_strc = np.sum(a_strc, axis = 2, keepdims = True)
        Ss.append(a_strc)
    
    Ss = np.array(Ss)
    print(Ss.shape)
    return Ss
As = []
data = train_data[train_data['signal_to_noise'] > 1].copy()
for id in tqdm(data['id']):
    a = np.load(f"/kaggle/input/stanford-covid-vaccine/bpps/{id}.npy")
    As.append(a)
As = np.array(As)
def get_distance_matrix(As):
    ## adjacent matrix based on distance on the sequence
    ## D[i, j] = 1 / (abs(i - j) + 1) ** pow, pow = 1, 2, 4
    
    idx = np.arange(As.shape[1])
    Ds = []
    for i in range(len(idx)):
        d = np.abs(idx[i] - idx)
        Ds.append(d)

    Ds = np.array(Ds) + 1
    Ds = 1/Ds
    Ds = Ds[None, :,:]
    Ds = np.repeat(Ds, len(As), axis = 0)
    
    Dss = []
    for i in [1, 2, 4]: 
        Dss.append(Ds ** i)
    Ds = np.stack(Dss, axis = 3)
    print(Ds.shape)
    return Ds
def preprocess_inputs(df, cols=['sequence', 'structure', 'predicted_loop_type'], seq_length = 107, flag = 'train'):
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

    Ss = get_structure_adj(df)
    Ss = Ss.sum(axis = 1)
    
    if flag == 'train':
        Ds = get_distance_matrix(As)
    elif flag == 'test_private':
        Ds = get_distance_matrix(As_private)
    elif flag == 'test_public':
        Ds = get_distance_matrix(As_public)
    Ds = Ds.sum(axis = 1)
    
    data = np.concatenate([base_fea,bpps_sum_fea,bpps_max_fea,bpps_nb_fea, Ss, Ds], 2)
    
    array_data = (np.reshape(([([list(df['A_percent'])[0]] * seq_length)]),(1,seq_length,1)))
    for i in range(1,len(df)):
        array_data_i = (np.reshape(([([list(df['A_percent'])[i]] * seq_length)]),(1,seq_length,1)))
        array_data = np.concatenate([array_data,array_data_i], axis = 0)

    data = np.concatenate([data, array_data], 2)

    for col in ['G_percent','C_percent','U_percent','U-G','C-G','U-A','G-C','A-U','G-U',
                'E','S','H','B','X','I','M','pairs_rate']:
        array_data = (np.reshape(([([list(df[col])[0]] * seq_length)]),(1,seq_length,1)))
        for i in range(1,len(df)):
            arraydom_data_i = (np.reshape(([([list(df[col])[i]] * seq_length)]),(1,seq_length,1)))
            array_data = np.concatenate([array_data,array_data_i], axis = 0)

        data = np.concatenate([data, array_data], 2)

    return data
bases = get_bases(train_data)
pairs = get_pairs(train_data)
loops = get_loops(train_data)
pairs_rate = get_pairs_rate(train_data)
train_data = pd.concat([train_data, bases, pairs, loops, pairs_rate], axis=1)

bases = get_bases(test_data)
pairs = get_pairs(test_data)
loops = get_loops(test_data)
pairs_rate = get_pairs_rate(test_data)
test_data = pd.concat([test_data, bases, pairs, loops, pairs_rate], axis=1)
train_inputs = preprocess_inputs(train_data.loc[train_data['signal_to_noise'] > 1], seq_length = 107, flag = 'train')
train_labels = np.array(train_data.loc[train_data['signal_to_noise'] > 1][target_cols].values.tolist()).transpose((0, 2, 1))
train_inputs.shape
from keras.losses import mean_squared_error

def root_mean_squared_error(y_true, y_pred):
    return tf.sqrt(mean_squared_error(y_true, y_pred))

def MCRMSE(y_true, y_pred):
    colwise_mse = tf.reduce_mean(tf.square(y_true - y_pred), axis=1)
    return tf.reduce_mean(tf.sqrt(colwise_mse), axis=1)

def lstm_layer(hidden_dim, dropout):
    return tf.keras.layers.Bidirectional(
                                tf.keras.layers.LSTM(hidden_dim,
                                dropout=dropout,
                                return_sequences=True,
                                kernel_initializer = 'orthogonal'))

def gru_layer(hidden_dim, dropout):
    return tf.keras.layers.Bidirectional(
                                tf.keras.layers.GRU(hidden_dim,
                                dropout=dropout,
                                return_sequences=True,
                                kernel_initializer='orthogonal'))

def build_model(n_layers = 2, seq_len = 107, num_features = 28, embed_dim = 200, sp_dropout = 0.2, hidden_dim = 512, dropout = 0.5, pred_len = 68, gru_flag = False):
    
    inputs = tf.keras.layers.Input(shape=(seq_len, num_features))
    categorical_feats = inputs[:, :, :3]
    numerical_feats = inputs[:, :, 3:10]
    overall_gene_feats = inputs[:, :, 10:]

    embed = tf.keras.layers.Embedding(input_dim=len(token2int), output_dim=embed_dim)(categorical_feats)
   
    reshaped = tf.reshape(embed, shape=(-1, embed.shape[1],  embed.shape[2] * embed.shape[3]))
    
    reshaped_1 = tf.keras.layers.concatenate([reshaped, numerical_feats], axis=2)
          
    normalized_layer_1 = tf.keras.layers.BatchNormalization()(reshaped_1)
    
    spatial_dropout = tf.keras.layers.SpatialDropout1D(sp_dropout)(normalized_layer_1)
  
    if gru_flag:
        for x in range(n_layers):
            normalized_layer_1 = gru_layer(hidden_dim, dropout)(normalized_layer_1)
        normalized_layer_2 = tf.keras.layers.BatchNormalization()(normalized_layer_1)
    else:
        for x in range(n_layers):
            normalized_layer_1 = lstm_layer(hidden_dim, dropout)(normalized_layer_1)
        normalized_layer_2 = tf.keras.layers.BatchNormalization()(normalized_layer_1)
    
    concat_layer = tf.keras.layers.concatenate([normalized_layer_2, overall_gene_feats], axis=2)
    
    dense_layer_1 = tf.keras.layers.Dense(100, activation = 'linear')(concat_layer)
    normalized_layer_3 = tf.keras.layers.BatchNormalization()(dense_layer_1)
    dropout_layer_1 = tf.keras.layers.SpatialDropout1D(sp_dropout)(normalized_layer_3)
   
    dense_layer_2 = tf.keras.layers.Dense(100, activation = 'linear')(dropout_layer_1)
    normalized_layer_4 = tf.keras.layers.BatchNormalization()(dense_layer_2)
    dropout_layer_2 = tf.keras.layers.SpatialDropout1D(sp_dropout)(normalized_layer_4)
    
    #only making predictions on the first part of each sequence
    truncated = dropout_layer_2[:, :pred_len]

    out = tf.keras.layers.Dense(5, activation='linear')(truncated)

    model = tf.keras.Model(inputs=inputs, outputs=out)

    #some optimizers
    adam = tf.optimizers.Adam()

    model.compile(optimizer = adam, loss = MCRMSE)
    
    return model
train_inputs.shape
# EPOCHS = 60
# BATCH_SIZE = 32

# model_GRU_on_train_data = build_model(gru_flag = True)
# model_GRU_on_train_data.summary()
# model_GRU_callback = tf.keras.callbacks.ModelCheckpoint(f'GRU model.h5')

# history_GRU = model_GRU_on_train_data.fit(train_inputs, train_labels,
#                   batch_size=BATCH_SIZE,
#                   epochs=EPOCHS,
#                   verbose = 2,
#                   callbacks=[model_GRU_callback])  
# EPOCHS = 60
# BATCH_SIZE = 32

# model_LSTM_on_train_data = build_model(gru_flag = False)
# model_LSTM_on_train_data.summary()
# model_LSTM_callback = tf.keras.callbacks.ModelCheckpoint(f'LSTM model.h5')

# history_LSTM = model_LSTM_on_train_data.fit(train_inputs, train_labels,
#                   batch_size=BATCH_SIZE,
#                   epochs=EPOCHS,
#                   verbose = 2,
#                   callbacks=[model_LSTM_callback])
# print(f" LSTM loss: {min(history_LSTM.history['loss'])}")
# print(f" GRU loss: {min(history_GRU.history['loss'])}")

# fig, ax = plt.subplots(1, 1, figsize = (20, 10))

# ax.plot(history_LSTM.history['loss'])
# ax.plot(history_GRU.history['loss'])

# ax.set_title('Model - LSTM vs GRU')

# ax.set_ylabel('Loss')
# ax.set_xlabel('Epoch')
# ax.legend()
# plt.show()
public_df = test_data.query("seq_length == 107").copy()
private_df = test_data.query("seq_length == 130").copy()
As_public = []
for id in tqdm(public_df["id"]):
    a = np.load(f"/kaggle/input/stanford-covid-vaccine/bpps/{id}.npy")
    As_public.append(a)
As_public = np.array(As_public)
As_private = []
for id in tqdm(private_df["id"]):
    a = np.load(f"/kaggle/input/stanford-covid-vaccine/bpps/{id}.npy")
    As_private.append(a)
As_private = np.array(As_private)
public_inputs = preprocess_inputs(public_df, seq_length = 107, flag = 'test_public')
private_inputs = preprocess_inputs(private_df, seq_length = 130, flag = 'test_private')
model_LSTM_on_test_data_public = build_model(seq_len=107, pred_len=107, gru_flag = False)
model_LSTM_on_test_data_public.load_weights('../input/openvaccine-covid-model-weights/LSTM model.h5')
pred_test_data_public_LSTM = model_LSTM_on_test_data_public.predict(public_inputs)

model_GRU_on_test_data_public = build_model(seq_len=107, pred_len=107, gru_flag = True)
model_GRU_on_test_data_public.load_weights('../input/openvaccine-covid-model-weights/GRU model (1).h5')
pred_test_data_public_GRU = model_GRU_on_test_data_public.predict(public_inputs)
model_LSTM_on_test_data_private = build_model(seq_len=130, pred_len=130, gru_flag = False)
model_LSTM_on_test_data_private.load_weights('../input/openvaccine-covid-model-weights/LSTM model.h5')
pred_test_data_private_LSTM = model_LSTM_on_test_data_private.predict(private_inputs)

model_GRU_on_test_data_private = build_model(seq_len=130, pred_len=130, gru_flag = True)
model_GRU_on_test_data_private.load_weights('../input/openvaccine-covid-model-weights/GRU model (1).h5')
pred_test_data_private_GRU = model_GRU_on_test_data_private.predict(private_inputs)
def format_predictions(public_preds, private_preds):
    preds = []
    
    for df, preds_ in [(public_df, public_preds), (private_df, private_preds)]:
        for i, uid in enumerate(df.id):
            single_pred = preds_[i]

            single_df = pd.DataFrame(single_pred, columns=target_cols)
            single_df['id_seqpos'] = [f'{uid}_{x}' for x in range(single_df.shape[0])]

            preds.append(single_df)
    return pd.concat(preds).reset_index(drop = True)
lstm_preds = format_predictions(pred_test_data_public_LSTM, pred_test_data_private_LSTM)
gru_preds = format_predictions(pred_test_data_public_GRU, pred_test_data_private_GRU)
lstm_preds.head()
gru_preds.head()
submission_LSTM = submission_format[['id_seqpos']].merge(lstm_preds, how = 'inner', on = 'id_seqpos')
submission_GRU = submission_format[['id_seqpos']].merge(gru_preds, how = 'inner', on = 'id_seqpos')
print(submission_LSTM.shape)
submission_LSTM.head()
print(submission_GRU.shape)
submission_GRU.head()
target_cols
submission_lstm_gru_combined = submission_GRU.merge(submission_LSTM, how = 'inner', on = 'id_seqpos')

gru_weight = 0.5
lstm_weight = 0.5
for i in range(len(target_cols)):
    submission_lstm_gru_combined[target_cols[i]] = submission_lstm_gru_combined[target_cols[i]+'_x']*gru_weight + submission_lstm_gru_combined[target_cols[i]+'_y']*lstm_weight
submission_lstm_gru_combined = submission_lstm_gru_combined[['id_seqpos'] + target_cols]
submission_lstm_gru_combined.head()
os.chdir("/kaggle/working/")
submission_LSTM.to_csv('submission_LSTM.csv', index = False)
submission_GRU.to_csv('submission_GRU.csv', index = False)
submission_lstm_gru_combined.to_csv('submission_lstm_gru_combined.csv', index = False)