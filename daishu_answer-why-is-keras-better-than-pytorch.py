import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



import random

import torch

import torch.nn as nn

import torch.nn.functional as F

from torch.nn import Linear, LayerNorm, ReLU, Dropout

from sklearn.model_selection import StratifiedKFold

from tqdm import tqdm

import os

import copy

from sklearn.cluster import KMeans

from sklearn.model_selection import StratifiedKFold, KFold, GroupKFold

from torch.utils.data import Dataset,TensorDataset, DataLoader,RandomSampler

import time,datetime

import tensorflow as tf

import keras.backend as K

import tensorflow.keras.layers as L

#gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)

#sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))



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
token2int = {x:i for i, x in enumerate('().ACGUBEHIMSX')}

pred_cols = ['reactivity', 'deg_Mg_pH10', 'deg_pH10', 'deg_Mg_50C', 'deg_50C']



def rmse(y_actual, y_pred):

    mse = tf.keras.losses.mean_squared_error(y_actual, y_pred)

    return K.sqrt(mse)



def mcrmse(y_actual, y_pred, num_scored=len(pred_cols)):

    score = 0

    for i in range(num_scored):

        score += rmse(y_actual[:, :, i], y_pred[:, :, i]) / num_scored

    return score



def preprocess_inputs(df, cols=['sequence', 'structure', 'predicted_loop_type']):

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





train = pd.read_json('../input/stanford-covid-vaccine/train.json', lines=True)

test = pd.read_json('../input/stanford-covid-vaccine/test.json', lines=True)



def read_bpps_sum(df):

    bpps_arr = []

    for mol_id in df.id.to_list():

        bpps_arr.append(np.load(f"../input/stanford-covid-vaccine/bpps/{mol_id}.npy").max(axis=1))

    return bpps_arr



def read_bpps_max(df):

    bpps_arr = []

    for mol_id in df.id.to_list():

        bpps_arr.append(np.load(f"../input/stanford-covid-vaccine/bpps/{mol_id}.npy").sum(axis=1))

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



from sklearn.cluster import KMeans



kmeans_model = KMeans(n_clusters=200, random_state=110).fit(preprocess_inputs(train)[:,:,0])

train['cluster_id'] = kmeans_model.labels_
def gru_layer(hidden_dim, dropout):

    return L.Bidirectional(L.GRU(hidden_dim, dropout=dropout, return_sequences=True, kernel_initializer = 'orthogonal'))



def lstm_layer(hidden_dim, dropout):

    return L.Bidirectional(L.LSTM(hidden_dim, dropout=dropout, return_sequences=True, kernel_initializer = 'orthogonal'))



def build_model(seq_len=107, pred_len=68, dropout=0.5, embed_dim=100, hidden_dim=256, type=0):

    inputs = L.Input(shape=(seq_len, 6))

    

    # split categorical and numerical features and concatenate them later.

    categorical_feat_dim = 3

    categorical_fea = inputs[:, :, :categorical_feat_dim]

    numerical_fea = inputs[:, :, 3:]



    embed = L.Embedding(input_dim=len(token2int), output_dim=embed_dim)(categorical_fea)

    reshaped = tf.reshape(embed, shape=(-1, embed.shape[1],  embed.shape[2] * embed.shape[3]))

    reshaped = L.concatenate([reshaped, numerical_fea], axis=2)

    

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

        hidden = lstm_layer(hidden_dim, dropout)(reshaped)

        hidden = lstm_layer(hidden_dim, dropout)(hidden)

    

    truncated = hidden[:, :pred_len]

    out = L.Dense(5, activation='linear')(truncated)

    model = tf.keras.Model(inputs=inputs, outputs=out)

    model.compile(tf.keras.optimizers.Adam(), loss=mcrmse)

    return model



keras_model = build_model()

keras_model.load_weights('../input/gru-lstm-with-feature-engineering-and-augmentation/modelGRU_LSTM1_cv0.h5')

keras_model.layers
device = torch.device('cuda:%s'%0 if torch.cuda.is_available() else 'cpu')

def Init_params(shape,w=None,b=None):

    if w is None:

        w = torch.nn.Parameter(torch.empty(*shape))

        nn.init.xavier_uniform_(w)

    else:

        w = torch.nn.Parameter(w)

    if b is None:

        b = torch.nn.Parameter(torch.zeros(shape[1]))

    else:

        b = torch.nn.Parameter(b)

    return w,b



class GRU(nn.Module):

    def __init__(self,input_dim,hidden_dim,w_i=None,b_i=None,w_h=None,b_h=None):

        super(GRU, self).__init__()

        self.w_i,self.b_i = Init_params([input_dim,3*hidden_dim],w_i,b_i)

        self.w_h,self.b_h = Init_params([hidden_dim,3*hidden_dim],w_h,b_h)

        self.hd = hidden_dim

    def forward(self,x):

        hidden = torch.zeros((x.shape[0], self.hd)).to(device)

        output = []

        for i in range(x.shape[1]):

            x_z = torch.matmul(x[:,i,:],self.w_i[:,:self.hd]) + self.b_i[:self.hd]

            x_r = torch.matmul(x[:,i,:],self.w_i[:,self.hd:2*self.hd]) + self.b_i[self.hd:2*self.hd]

            x_n = torch.matmul(x[:,i,:],self.w_i[:,2*self.hd:]) + self.b_i[2*self.hd:]



            h_z = torch.matmul(hidden,self.w_h[:,:self.hd]) + self.b_h[:self.hd]

            h_r = torch.matmul(hidden,self.w_h[:,self.hd:2*self.hd]) + self.b_h[self.hd:2*self.hd]

            h_n = torch.matmul(hidden,self.w_h[:,2*self.hd:]) + self.b_h[2*self.hd:]



            z = torch.sigmoid(x_z+h_z)

            r = torch.sigmoid(x_r+h_r)

            n = torch.tanh(x_n+r*h_n)

            h = (1-z)*n + z*hidden

            hidden = h

            output.append(h.unsqueeze(1))

        return torch.cat(output,1)



class Net(nn.Module):

    def __init__(self):

        super(Net, self).__init__()

        num_target=5

        w = torch.Tensor(keras_model.layers[2].get_weights()[0])

        self.cate_emb = nn.Embedding.from_pretrained(w,freeze=False)

        self.gru = GRU(100*3+3, 256, torch.Tensor(keras_model.layers[-4].get_weights()[0]).contiguous(),

                       torch.Tensor(keras_model.layers[-4].get_weights()[2][0]).contiguous(),

                       torch.Tensor(keras_model.layers[-4].get_weights()[1]).contiguous(),

                       torch.Tensor(keras_model.layers[-4].get_weights()[2][1]).contiguous())

        self.reverse_gru = GRU(100*3+3, 256, torch.Tensor(keras_model.layers[-4].get_weights()[3]).contiguous(),

                       torch.Tensor(keras_model.layers[-4].get_weights()[5][0]).contiguous(),

                       torch.Tensor(keras_model.layers[-4].get_weights()[4]).contiguous(),

                       torch.Tensor(keras_model.layers[-4].get_weights()[5][1]).contiguous())

        self.gru1 = GRU(512, 256, torch.Tensor(keras_model.layers[-3].get_weights()[0]).contiguous(),

                       torch.Tensor(keras_model.layers[-3].get_weights()[2][0]).contiguous(),

                       torch.Tensor(keras_model.layers[-3].get_weights()[1]).contiguous(),

                       torch.Tensor(keras_model.layers[-3].get_weights()[2][1]).contiguous())

        self.reverse_gru1 = GRU(512, 256, torch.Tensor(keras_model.layers[-3].get_weights()[3]).contiguous(),

                       torch.Tensor(keras_model.layers[-3].get_weights()[5][0]).contiguous(),

                       torch.Tensor(keras_model.layers[-3].get_weights()[4]).contiguous(),

                       torch.Tensor(keras_model.layers[-3].get_weights()[5][1]).contiguous())

        self.predict = nn.Linear(512,num_target)

        for i,(n,p) in enumerate(self.predict.named_parameters()):

            if i == 0:

                p.data = torch.nn.Parameter(torch.Tensor(keras_model.layers[-1].get_weights()[0].T).contiguous())

            if i == 1:

                p.data = torch.nn.Parameter(torch.Tensor(keras_model.layers[-1].get_weights()[1]).contiguous())



    def forward(self, cateX,contX):

        cate_x = self.cate_emb(cateX).view(cateX.shape[0],cateX.shape[1],-1)

        sequence = torch.cat([cate_x,contX],-1)

        x = self.gru(sequence)

        reverse_x = torch.flip(self.reverse_gru(torch.flip(sequence,[1])),[1])

        sequence = torch.cat([x,reverse_x],-1)

        x = self.gru1(sequence)

        reverse_x = torch.flip(self.reverse_gru1(torch.flip(sequence,[1])),[1])

        x = torch.cat([x,reverse_x],-1)

        x = F.dropout(x,0.5,training=self.training)

        predict = self.predict(x)

        return predict

pytorch_model = Net()

pytorch_model.to(device)
# check 1 samples

x = preprocess_inputs(train[:1])

cate_x = torch.LongTensor(x[:,:,:3]).to(device)

cont_x = torch.Tensor(x[:,:,3:]).to(device)

y = np.array(train[:1][pred_cols].values.tolist()).transpose((0, 2, 1))
keras_y = keras_model.predict(x)

keras_y
pytorch_model.eval()

pytorch_y = pytorch_model(cate_x,cont_x).detach().cpu().numpy()

pytorch_y
# output difference between Keras and Pytorch

np.mean(np.abs(keras_y-pytorch_y[:,:68,:]))
# check all samples



gkf = GroupKFold(n_splits=5)

keras_predict = []

pytorch_predict = []

targets = []



for fold, (train_index, valid_index) in enumerate(gkf.split(train,  train['reactivity'], train['cluster_id'])):

    keras_model.load_weights('../input/gru-lstm-with-feature-engineering-and-augmentation/modelGRU_LSTM1_cv%s.h5'%fold)

    t_valid = train.iloc[valid_index]

    t_valid = t_valid[t_valid['SN_filter'] == 1]

    valid_x = preprocess_inputs(t_valid)

    valid_count = valid_x.shape[0]

    valid_cate_x = torch.LongTensor(valid_x[:,:,:3])

    valid_cont_x = torch.Tensor(valid_x[:,:,3:])

    valid_y = torch.Tensor(np.array(t_valid[pred_cols].values.tolist()).transpose((0, 2, 1)))



    valid_data = TensorDataset(valid_cate_x,valid_cont_x,valid_y)

    valid_data_loader = DataLoader(dataset=valid_data,shuffle=False,batch_size=32,num_workers=1)

    valid_y = valid_y.numpy()

    targets.append(valid_y)

    

    # Keras predict

    keras_predict.append(keras_model.predict(valid_x))

    

    # Pytorch predict and save oof

    pytorch_model = Net()

    pytorch_model.to(device)

    

    pytorch_model.eval()

 

    all_pred = []



    for data in valid_data_loader:

        cate_x,cont_x,y = [x.to(device) for x in data]

        outputs = pytorch_model(cate_x,cont_x)

        all_pred.append(outputs.detach().cpu().numpy())

    all_pred = np.concatenate(all_pred,0)[:,:68,:]

    pytorch_predict.append(all_pred)
for i in range(5):

    print('fold %s output difference between Keras and Pytorch:'%i,np.mean(np.abs(keras_predict[i]-pytorch_predict[i])))
def Metric(target,pred):

    metric = 0

    for i in range(target.shape[-1]):

        metric += (np.sqrt(np.mean((target[:,:,i]-pred[:,:,i])**2))/target.shape[-1])

    return metric
for i in range(5):

    print('fold %s'%i,'|','metric of keras outputs:%.6f'%Metric(targets[i],keras_predict[i]),'|','metric of pytorch outputs:%.6f'%Metric(targets[i],pytorch_predict[i]))
# let's check keras loss fuction



def rmse(y_actual, y_pred):

    mse = tf.keras.losses.mean_squared_error(y_actual, y_pred)

    return K.sqrt(mse)



def mcrmse(y_actual, y_pred, num_scored=5):

    score = 0

    for i in range(num_scored):

        score += rmse(y_actual[:, :, i], y_pred[:, :, i]) / num_scored

    return score



for i in range(5):

    print('fold %s'%i,mcrmse(targets[i],keras_predict[i]))
# let's check the average of keras loss fuction outputs

for i in range(5):

    print('fold %s'%i,K.mean(mcrmse(targets[i],keras_predict[i])))
# Finaly, let's check the submission of pytorch is same as keras.



# predict test

def Pred(df):

    test_x = preprocess_inputs(df)

    test_cate_x = torch.LongTensor(test_x[:,:,:3])

    test_cont_x = torch.Tensor(test_x[:,:,3:])

    test_data = TensorDataset(test_cate_x,test_cont_x)

    test_data_loader = DataLoader(dataset=test_data,shuffle=False,batch_size=64,num_workers=1)

    all_id = []

    for i,row in df.iterrows():

        for j in range(row['seq_length']):

            all_id.append(row['id']+'_%s'%j)



    all_id = np.array(all_id).reshape(-1,1)

    all_pred = np.zeros(len(all_id)*5).reshape(len(all_id),5)

    for fold in range(5):

        keras_model.load_weights('../input/gru-lstm-with-feature-engineering-and-augmentation/modelGRU_LSTM1_cv%s.h5'%fold)

        model = Net()

        model.to(device)

        model.eval()

        t_all_pred = []

        for data in test_data_loader:

            cate_x,cont_x = [x.to(device) for x in data]

            outputs = model(cate_x,cont_x)

            t_all_pred.append(outputs.detach().cpu().numpy())

        t_all_pred = np.concatenate(t_all_pred,0)

        all_pred += t_all_pred.reshape(-1,5)

    all_pred /= 5

    sub = pd.DataFrame(all_pred,columns=['reactivity','deg_Mg_pH10','deg_pH10','deg_Mg_50C','deg_50C'])

    sub['id_seqpos'] = all_id

    return sub

public_sub = Pred(test.loc[test['seq_length']==107])

private_sub = Pred(test.loc[test['seq_length']==130])

pytorch_sub = pd.concat([public_sub,private_sub]).reset_index(drop=True)

pytorch_sub = pytorch_sub[['id_seqpos']+pred_cols]
pytorch_sub = pytorch_sub.sort_values(by=['id_seqpos']).reset_index(drop=True)
keras_sub = pd.read_csv('../input/gru-lstm-with-feature-engineering-and-augmentation/submission.csv')
keras_sub = keras_sub.sort_values(by=['id_seqpos']).reset_index(drop=True)
keras_sub.head()
pytorch_sub.head()
# submission difference between Keras and Pytorch

np.mean(np.abs(keras_sub[pred_cols].values-pytorch_sub[pred_cols].values))
pytorch_sub.to_csv('./submission.csv',index=False)