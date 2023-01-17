import numpy as np 
import pandas as pd

import torch
from sklearn import preprocessing
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
from random import shuffle
import os
from torch.autograd import Variable
import torch.optim as optim
import warnings
warnings.filterwarnings("ignore")
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
    
class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        super(TCN, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size, dropout=dropout)
        self.linear = nn.Linear(num_channels[-1], output_size)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        # x needs to have dimension (N, C, L) in order to be passed into CNN
        output = self.tcn(x.transpose(1, 2)).transpose(1, 2)
        output = self.linear(output).double()
        return self.sig(output)
from kaggle.competitions import twosigmanews
env = twosigmanews.make_env()

(market_train_df, news_train_df) = env.get_training_data()

def prepare_data(market_data, news_data, asset_name):

    df_market = market_data[market_data['assetName']==asset_name]
    df_market['time'] = df_market['time'].apply(lambda x: x.date())
    
    df_news = news_data[news_data['assetName']==asset_name]
    df_news['time'] = df_news['time'].apply(lambda x: x.date())
    df_news =  df_news.groupby('time').mean()
    df_news ['time'] = df_news.index


    df_target = df_market['returnsOpenNextMktres10']

    
    market_columns = ['time', 'volume', 'close', 'open', 
                    'returnsClosePrevRaw1', 'returnsOpenPrevRaw1', 'returnsClosePrevMktres1', 
                    'returnsOpenPrevMktres1', 'returnsClosePrevRaw10', 'returnsOpenPrevRaw10', 
                    'returnsClosePrevMktres10', 'returnsOpenPrevMktres10']
    df_market = df_market[market_columns]

    
    news_columns = ['time', 'urgency', 'takeSequence', 'bodySize', 
                    'sentenceCount', 'wordCount', 'firstMentionSentence', 'relevance', 
                    'sentimentClass', 'sentimentNegative', 'sentimentNeutral', 'sentimentPositive', 
                    'sentimentWordCount']
    df_news = df_news[news_columns]
    
    if df_news.shape[0] == 0:
        df_news.ix[0] = [0] * df_news.shape[1]

    df_merged = pd.merge(df_market, df_news, how='left', on=['time']) #, validate='many_to_one')
    
    df_merged = df_merged.fillna(0)
    df_merged = df_merged.drop('time', axis=1)

    
    data_array = preprocessing.scale(np.array(df_merged, dtype="float32"))
    X = torch.tensor(data_array)
    
    Y = torch.tensor(np.array(df_target, dtype="float32"))

    return X, Y
args_cuda = True
args_dropout = 0.25
args_epochs = 5 #####
args_kernel_size = 4
args_levels = 5
args_nhid = 120
args_lr = 1e-3
args_optim = 'Adam'
args_log_interval = 50
args_cuda = torch.cuda.is_available()

input_size = 23
output_size = 1
n_channels = [args_nhid] * args_levels
kernel_size = args_kernel_size

model = TCN(input_size, output_size, n_channels, args_kernel_size, dropout=args_dropout)
if args_cuda:
    model.cuda()

optimizer = getattr(optim, args_optim)(model.parameters(), lr=args_lr)


def evaluate(market_data, news_data):  
    mse = torch.nn.MSELoss()
    sigm = torch.nn.Sigmoid()
    
    model.eval()
    
    total_loss = 0.0
    count = 0
    
    for asset_name in asset_names:
        x, y = prepare_data(market_data, news_data, asset_name)
        X, Y = Variable(torch.tensor(x)), Variable(torch.tensor(y))
        if args_cuda:
            X, Y = X.cuda(), Y.cuda()
                 
        output = model(X.unsqueeze(0)).squeeze(0)
              
        target = sigm(Y)
              
        loss = mse(output.float(), target)
        
        total_loss += loss.item()
        count += 1
        
    eval_loss = total_loss/count
    print("Evaluation loss: ", eval_loss)
    return eval_loss


def train(market_data, news_data, epoch_count):
    mse = torch.nn.MSELoss()
    sigm = torch.nn.Sigmoid()
    
    model.train()
    
    total_loss = 0
    
    cnt = 0

    for asset_name in asset_names:
        x, y = prepare_data(market_data, news_data, asset_name)
        X, Y = Variable(torch.tensor(x)), Variable(torch.tensor(y))
        if args_cuda:
            X, Y = X.cuda(), Y.cuda()
    
        optimizer.zero_grad() 
        output = model(X.unsqueeze(0)).squeeze(0)
             
        target = sigm(Y)
        
        loss = mse(output.float(), target)
        
        total_loss += loss.item()      
        
        loss.backward()
        optimizer.step()
        
        cnt += 1
        if cnt % args_log_interval == 0:
            
            curr_loss = total_loss/cnt
            print("Epoch", epoch_count,
                  " | lr ", args_lr,
                  " | loss ", curr_loss)
            total_loss = 0.0
 
"""Main Part"""

best_vloss = 1e8
vloss_list = []

input_path = "../input/tcn-twosigma-training/"
state_dict = "TwoSigma_ModelDict"
model_name = "TwoSigma_Model.pt"

if model_name in os.listdir(input_path) and state_dict in os.listdir(input_path):
    model = torch.load(input_path + model_name)
    checkpoint = torch.load(input_path + state_dict)
    model.load_state_dict(checkpoint['model_state'])
    optimizer.load_state_dict(checkpoint['optim_state'])
    args_lr = checkpoint['lr']
    print('Model loaded')
    

for epoch_cnt in range(1, args_epochs+1):
    
    asset_names = market_train_df['assetName'].unique()
    shuffle(asset_names)
    asset_names = asset_names[:500]
    
    train(market_train_df, news_train_df, epoch_cnt)
    vloss =evaluate(market_train_df, news_train_df)
    if vloss < best_vloss:
        save_dict = {'model_state':model.state_dict(),
                    'optim_state':optimizer.state_dict(),
                    "lr" : args_lr}
        torch.save(save_dict, state_dict)
        torch.save(model, model_name)
        best_vloss = vloss
        print('Model saved')
    if epoch_cnt>10 and vloss > max(vloss_list[-3]):
        args_lr /= 10
        for param_group in optimizer.param_groups:
            param_group['lr'] = args_lr
    vloss_list.append(vloss)        
    print('Epoch finished')
print('Done!')

