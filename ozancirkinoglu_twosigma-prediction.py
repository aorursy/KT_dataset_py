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
from kaggle.competitions import twosigmanews
env = twosigmanews.make_env()

(market_train_df, news_train_df) = env.get_training_data()
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
def prepare_prediction_data(market_data, news_data, asset_code):

    df_market = market_data[market_data['assetCode']==asset_code]
    
    if df_market.shape[0] == 0:
        return None
        
    asset_name = df_market.iloc[0]['assetName']
    df_market['time'] = df_market['time'].apply(lambda x: x.date())
    
    df_news = news_data[news_data['assetName']==asset_name]
    df_news['time'] = df_news['time'].apply(lambda x: x.date())
    df_news =  df_news.groupby('time').mean()
    df_news ['time'] = df_news.index

    
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

    return X
'''Main Part'''

state_dict = "../input/tcn-twosigma-training/TwoSigma_ModelDict"
model_name = "../input/tcn-twosigma-training/TwoSigma_Model.pt"


args_dropout = 0.25
args_kernel_size = 4
args_lr = 1e-3
args_levels = 5
args_nhid = 120
args_optim = 'Adam'
args_cuda = False

input_size = 23
output_size = 1
n_channels = [args_nhid] * args_levels
kernel_size = args_kernel_size
model = TCN(input_size, output_size, n_channels, args_kernel_size, dropout=args_dropout)
if args_cuda:
    model.cuda()
optimizer = getattr(optim, args_optim)(model.parameters(), lr=args_lr)
    
model = torch.load(model_name, map_location='cpu')
checkpoint = torch.load(state_dict, map_location='cpu')
model.load_state_dict(checkpoint['model_state'])
optimizer.load_state_dict(checkpoint['optim_state'])
lr = checkpoint['lr']
print('Model loaded')

if ("twosigma-prediction" in os.listdir("../input/") and "X_Dict" in os.listdir("../input/twosigma-prediction/")):
    X_dict = torch.load("../input/twosigma-prediction/X_Dict", map_location='cpu')
    torch.save(X_dict, "X_Dict")
    print('X Dict loaded')
else:
    X_dict = {}
    asset_codes = market_train_df['assetCode'].unique()
    for asset_code in asset_codes:
        X = prepare_prediction_data(market_train_df, news_train_df, asset_code)
        X_dict[asset_code] = X
    torch.save(X_dict, "X_Dict")
    print('X Dict saved')
input_path = "../input/twosigma-prediction/"

if  "predictions" in os.listdir(input_path):
    predictions_dict = torch.load(input_path+"predictions")
    torch.save(predictions_dict, "predictions")
    print('Predictions loaded...')
else:
    predictions_dict = {}
    print('Predictions initialized') 

days = env.get_prediction_days()
cnt=1
for (market_obs_df, news_obs_df, predictions_template_df) in days:
    print('Day ', cnt)    
    if cnt in predictions_dict.keys():
        predictions_df = predictions_dict[cnt]
    elif cnt > 3:
        predictions_df = predictions_template_df
    else:
        predictions_df = predictions_template_df
        asset_codes = market_obs_df['assetCode'].unique()

        for asset_code in asset_codes:
            X_new = prepare_prediction_data(market_obs_df, news_obs_df, asset_code)
            if asset_code in X_dict.keys():
                X_train = X_dict[asset_code] 
                X_prediction = torch.cat((X_train, X_new), 0)      
            else:
                X_prediction = X_new
            X_dict[asset_code] = X_prediction
            
            if args_cuda:
                X_prediction = X_prediction.cuda()
            output = model(X_prediction.unsqueeze(0)).squeeze(0)
            prediction = output[-1,:].item()
            predictions_df.loc[predictions_df['assetCode'] == asset_code,'confidenceValue'] = prediction
        predictions_dict[cnt] = predictions_df
        torch.save(predictions_dict, "predictions")
        
    env.predict(predictions_df)
    cnt += 1
    print('Done...')

#env.write_submission_file() 


