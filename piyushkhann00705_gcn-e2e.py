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
import torch
CUDA=torch.version.cuda
TORCH=torch.__version__
print(CUDA)
print(TORCH)
!pip install torch-scatter
!pip install torch-sparse
!pip install torch-cluster
!pip install torch-spline-conv
!pip install torch-geometric
import torch
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import numpy as np 
import pandas as pd
import pickle
import json
import os
import networkx as nx
import datetime
from tqdm import tqdm
from statistics import mean 
import matplotlib.pyplot as plt 
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Flatten, LSTM, Conv1D, MaxPooling1D, Dropout, Activation , Masking, Bidirectional, GlobalAvgPool1D, GlobalMaxPool1D, Conv1D, TimeDistributed, Input, Concatenate, GRU, dot, multiply, concatenate, add, Lambda 
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.models import Sequential
import keras
import scipy.stats as stats
from scipy.stats import pearsonr
from scipy.stats import spearmanr
import tensorflow as tf
import logging
import gc
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch import optim
import random
import torch.nn as nn
base_dir = '../input/gcn-data/'

with open(os.path.join(base_dir, 'earning_calls_ticker_index.json'), 'rb') as f:
    ticker_indices = json.load(f)
    
with open('../input/audiolstm/final_audio_dict.pkl', 'rb') as f:
    final_audio_dict=pickle.load(f)
    
with open('../input/gcn-embd/finbert_earnings.pkl', 'rb') as f:
    text_dict=pickle.load(f)
    
    
traindf= pd.read_csv("../input/stockspeechgraph/train_split_Avg_Series_WITH_LOG.csv")
testdf=pd.read_csv("../input/stockspeechgraph/test_split_Avg_Series_WITH_LOG.csv")
valdf=pd.read_csv("../input/stockspeechgraph/val_split_Avg_Series_WITH_LOG.csv")

# single_traindf= pd.read_csv("../input/stockspeechgraph/train_split_SeriesSingleDayVol3.csv")
# single_testdf=pd.read_csv("../input/stockspeechgraph/test_split_SeriesSingleDayVol3.csv")
# single_valdf=pd.read_csv("../input/stockspeechgraph/val_split_SeriesSingleDayVol3.csv")

adj_wiki = pd.read_csv(os.path.join(base_dir, 'adj_wiki.csv'))
adj_wiki = adj_wiki.values
adj_wiki = adj_wiki[:,1:]
print('Shape of wiki-company based adjacenecy matrix is : ', adj_wiki.shape)

adj_graph = adj_wiki
ticker_indices
traindf.head()
# call2nodeindex=index of earning call in adjacency matrix
def convert_calls2graph(df):
    call2nodeindex = {} 
#     call_stock_ind = {}
    i=277
    start = i
    for index, row in df.iterrows():
        call2nodeindex[row['text_file_name']] = i
#         if row['ticker'].upper() in ticker_indices.keys():
#             call_stock_ind[row['text_file_name']] = ticker_indices[row['ticker'].upper()]
#         else:
#             call_stock_ind[row['text_file_name']] = -1
        i+=1
    set_end = i
    set_idx_range = range(start, set_end)
    
    
    print('Number of unique calls:', len(call2nodeindex))
    print('Total number of graph nodes:', i)
    print('Number of values', len(set_idx_range))
#     print('Number of test values', len(test_idx))
#     print('Number of validation values', len(val_idx))
    new_adj_matrix_shape = (i, i)

    set_adj = np.zeros(new_adj_matrix_shape,dtype=np.float32)
    set_adj[:adj_graph.shape[0], :adj_graph.shape[1]]=adj_graph
    ## add calls to graph

    for idx, row in df.iterrows():
        call_name = row['text_file_name']
        ticker=row['ticker']
        set_adj[call2nodeindex[call_name], call2nodeindex[call_name]] = 1.0
        if ticker in ticker_indices:
            set_adj[call2nodeindex[call_name], ticker_indices[ticker]] = 1.0
            set_adj[ticker_indices[ticker], call2nodeindex[call_name]] = 1.0
            
    print('Final adjacency matrix shape: ', set_adj.shape)
    print('Number of connections: ', np.sum(set_adj))
    
    G = nx.from_numpy_matrix(set_adj)
    print("Graph Info:")
    print(nx.info(G))
    
    edges=list(G.edges())
    
    edges_np = np.array(edges)
    edges_transpose = np.transpose(edges_np)
    
    TOTAL_NODES = set_adj.shape[0]
    
    #collecting labels
    y3days = np.zeros(TOTAL_NODES, dtype=np.float32)
    y7days = np.zeros(TOTAL_NODES, dtype=np.float32)
    y15days = np.zeros(TOTAL_NODES, dtype=np.float32)
    y30days = np.zeros(TOTAL_NODES, dtype=np.float32)
    
    for index, row in df.iterrows():
        y3days[call2nodeindex[row['text_file_name']]]= row['future_3']
        y7days[call2nodeindex[row['text_file_name']]]= row['future_7']
        y15days[call2nodeindex[row['text_file_name']]]= row['future_15']
        y30days[call2nodeindex[row['text_file_name']]]= row['future_30']
    
    return edges_transpose,y3days, y7days, y15days, y30days

edges_transpose,y3days, y7days, y15days, y30days=convert_calls2graph(traindf)
class BiLSTMAudio(nn.Module):
    def __init__(self):
        super(BiLSTMAudio, self).__init__()
        self.dropout = nn.Dropout(0.2)
        self.lstm = nn.LSTM(input_size=23,
                            hidden_size=100,
                            num_layers=1, 
                            dropout = 0.5,
                            batch_first=True,
                            bidirectional=True).cuda()
#         self.time_dist = TimeDistributed(self.fc)
#         self.hidden2label = nn.Linear(100*2*2, 200).cuda()
    
    def forward(self, sents):
#         sents = torch.transpose(sents, 0, 1)
        lstm_out, (h_n, c_n) = self.lstm(sents)
#         y = self.hidden2label(self.dropout(torch.cat([c_n[i,:, :] for i in range(c_n.shape[0])], dim=1)))
#         y = self.dropout(torch.cat([c_n[i,:, :] for i in range(c_n.shape[0])], dim=1))
#         lstm_out = torch.transpose(lstm_out, 0, 1)
        return lstm_out


class BiLSTM(nn.Module):
    def __init__(self):
        super(BiLSTM, self).__init__()
        self.dropout = nn.Dropout(0.2)
        self.lstm = nn.LSTM(input_size=200,
                            hidden_size=100,
                            num_layers=1, 
                            dropout = 0.0,
                            batch_first=True,
                            bidirectional=True).cuda()
#         self.time_dist = TimeDistributed(self.fc)
        self.hidden2label = nn.Linear(100*2, 50).cuda()
        self.tanh = nn.Tanh()
    
    def forward(self, sents):
        lstm_out, (h_n, c_n) = self.lstm(sents)
        y = self.hidden2label(self.tanh(self.dropout(torch.cat([c_n[i,:, :] for i in range(c_n.shape[0])], dim=1))))
#         y = self.dropout(torch.cat([c_n[i,:, :] for i in range(c_n.shape[0])], dim=1))
        y=self.tanh(y)
        return y

class BiLSTMText(nn.Module):
    def __init__(self):
        super(BiLSTMText, self).__init__()
        self.dropout = nn.Dropout(0.2)
        self.lstm = nn.LSTM(input_size=768,
                            hidden_size=100,
                            num_layers=1, 
                            dropout = 0.8,
                            batch_first=True,
                            bidirectional=True).cuda()
        self.tanh = nn.Tanh()
#         self.time_dist = TimeDistributed(self.fc)
#         self.hidden2label = nn.Linear(100*2*2, 200).cuda()
    
    def forward(self, sents):
#         sents = torch.transpose(sents, 0, 1)
        lstm_out, (h_n, c_n) = self.lstm(sents)
#         y = self.hidden2label(self.dropout(torch.cat([c_n[i,:, :] for i in range(c_n.shape[0])], dim=1)))
#         y = self.dropout(torch.cat([c_n[i,:, :] for i in range(c_n.shape[0])], dim=1))
#         lstm_out = torch.transpose(lstm_out, 0, 1)
        lstm_out = self.tanh(lstm_out)
        return lstm_out
# blstm = BiLSTMText()
# inp = torch.randn(5, 3, 768).cuda()
# y = blstm(inp)
# print(y.size())
class MDRM_Pytorch(nn.Module):
    def __init__(self):
        super(MDRM_Pytorch, self).__init__()
        self.prev1 = BiLSTMAudio().cuda()
        self.prev2 = BiLSTMText().cuda()
        self.model = nn.Sequential(
#             nn.Linear(100, 50),
#                       nn.ReLU(),
                      nn.Linear(50, 1)).cuda()
        self.softmax = nn.Softmax()
        self.blstm = BiLSTM().cuda()
        
#         self.hidden2label = nn.Linear(100*2*2, 200).cuda()
    
    def forward(self, inp_audio, inp_text):
        y1 = self.prev1(inp_audio)
        y2 = self.prev2(inp_text)
        
        m1 = torch.matmul(y1, torch.transpose(y2, 1,2))
        n1 = self.softmax(m1)
        o1 = torch.matmul(n1, y2)
        a1 = torch.mul(o1, y1)
        
        m2 = torch.matmul(y2, torch.transpose(y1, 1,2))
        n2 = self.softmax(m2)
        o2 = torch.matmul(n2, y1)
        a2 = torch.mul(o2, y2)
#         print(a1.size(), a2.size())
        y = torch.cat((a1, a2), dim=1)
        
        y = self.blstm(y)
#         y = self.hidden2label(y)
        out = self.model(y)
        return y1, y2, y, out
inp = torch.randn(5, 3, 23)
inp1 = torch.randn(5, 3, 23)
x = torch.matmul(inp, torch.transpose(inp1, 1,2))
a = nn.Softmax()
x_ = a(x)
x = torch.matmul(x_, inp1)
x.size()
model = MDRM_Pytorch()
inp = torch.randn(5, 3, 23).cuda()
inp1 = torch.randn(5, 3, 768).cuda()
y1, y2, y, out= model(inp, inp1)
print(out.size())
# model.cuda()

# criterion= criterion.cuda()
USE_GPU=True
if USE_GPU and torch.cuda.is_available():
    X1 = torch.from_numpy(X_train_audio).cuda()
    X2 = torch.from_numpy(X_train_text).cuda()
    y3 = torch.from_numpy(y_train3days).view(-1,1).cuda()
    y7 = torch.from_numpy(y_train7days).view(-1,1).cuda()
    y15 = torch.from_numpy(y_train15days).view(-1,1).cuda()
    y30 = torch.from_numpy(y_train30days).view(-1,1).cuda()
    X1test = torch.from_numpy(X_test_audio).cuda()
    X2test = torch.from_numpy(X_test_text).cuda()
    y3test = torch.from_numpy(y_test3days).view(-1,1).cuda()
    y7test = torch.from_numpy(y_test7days).view(-1,1).cuda()
    y15test = torch.from_numpy(y_test15days).view(-1,1).cuda()
    y30test = torch.from_numpy(y_test30days).view(-1,1).cuda()
    dtype = torch.cuda.FloatTensor
#     model.cuda()
# else:
#     X = torch.from_numpy(X_train_audio)
#     y = torch.from_numpy(y_train3days).view(-1,1)
#     dtype = torch.FloatTensor

X1 = Variable(X1).type(dtype)
X2 = Variable(X2).type(dtype)
ys = [Variable(y3).type(dtype), Variable(y7).type(dtype), Variable(y15).type(dtype), Variable(y30).type(dtype)]

X1test = Variable(X1test).type(dtype)
X2test = Variable(X2test).type(dtype)
ystest = [Variable(y3test).type(dtype), Variable(y7test).type(dtype), Variable(y15test).type(dtype), Variable(y30test).type(dtype)]


print(X1.shape, ys[0].shape)
# torch.cuda.manual_seed_all(SEED)
epoches = 30
i=0
mses = [[],[],[],[]]
for y_i in ys:
    days = ['3 days', '7 days', '15 days', '30 days']
    for run in range(50):
        model = MDRM_Pytorch()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        for epoch in np.arange(epoches):
            model.train()
            optimizer.zero_grad()
            y1, y2, y, out= model(X1, X2)
            loss = criterion(out, y_i)
            loss.backward()
            optimizer.step()
#             if epoch % 4 == 0:
#                 print(f"Loss for {epoch} epoch : {loss.item()}" )
        with torch.no_grad():
            model.eval()
            _, __, ___, pred = model(X1test, X2test)
            mse = criterion(pred, ystest[i])
#             print("Loss : ", mse.item())
            mses[i].append(mse.item())
#     np.save('sia_bert_mdrm_pyt_concat_'+str(days[i])+'.npy',np.array(mses[i]) )
    print('Mean for ', days[i], ':', np.mean(np.array(mses[i])))
    print('Stdev for ', days[i], ':', np.std(np.array(mses[i])))
    i+=1
mse_df=pd.DataFrame(mses)
mse_df.to_csv('./MSE List.csv')
call_dict = {} ## the index of the call in the adjacency matrix
call_stock_ind = {}
i=277
start = i
for index, row in traindf.iterrows():
    if not row['text_file_name'] in call_dict.keys():
        call_dict[row['text_file_name']] = i
        if row['ticker'].upper() in ticker_indices.keys():
            call_stock_ind[row['text_file_name']] = ticker_indices[row['ticker'].upper()]
        else:
            call_stock_ind[row['text_file_name']] = -1
        i+=1
train_end = i
train_idx = range(start, train_end)

for index, row in testdf.iterrows():
    if not row['text_file_name'] in call_dict.keys():
        call_dict[row['text_file_name']] = i
        if row['ticker'].upper() in ticker_indices.keys():
            call_stock_ind[row['text_file_name']] = ticker_indices[row['ticker'].upper()]
        else:
            call_stock_ind[row['text_file_name']] = -1
        i+=1
test_end = i
test_idx = range(train_end, test_end)

for index, row in valdf.iterrows():
    if not row['text_file_name'] in call_dict.keys():
        call_dict[row['text_file_name']] = i
        if row['ticker'].upper() in ticker_indices.keys():
            call_stock_ind[row['text_file_name']] = ticker_indices[row['ticker'].upper()]
        else:
            call_stock_ind[row['text_file_name']] = -1
        i+=1
val_end = i
val_idx = range(test_end, val_end)
