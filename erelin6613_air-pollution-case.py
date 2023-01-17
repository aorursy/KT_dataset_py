# Uncomment if you do not have folium installed
# !pip3 install folium
!pip3 install jovian --quiet
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import random_split
import torch.nn.functional as F
from torch.optim import Adam
import jovian
import geopandas as gpd
import folium
from folium.plugins import HeatMap
from tqdm import tqdm
#from category_encoders import Label
# jovian.commit(project=project_name)

%matplotlib inline
plt.rcParams['figure.figsize'] = (15, 5);
project_name = 'air-pollution-case'
author = 'Valentyna Fihurska'
pol_df = pd.read_csv(
    '../input/so2-emissions-daily-summary-data/daily_42401_2017/daily_42401_2017.csv')
pol_df.head(5)
pol_df
samp = pol_df.loc[pol_df['Latitude'].notnull()]
map_sample = folium.Map(location=[50,-85], zoom_start=3)
HeatMap(data=samp[['Latitude', 'Longitude']], radius=10).add_to(map_sample)
map_sample
p2018 = pd.read_csv('../input/so2-emissions-daily-summary-data/daily_42401_2018.csv')
p2019 = pd.read_csv('../input/so2-emissions-daily-summary-data/daily_42401_2019.csv')
pol_df = pol_df.loc[(pol_df['County Code']==101) & (pol_df['State Code']==42)]
p2018 = p2018.loc[(p2018['County Code']==101) & (p2018['State Code']==42)]
p2019 = p2019.loc[(p2019['County Code']==101) & (p2019['State Code']==42)]
pol_df = pd.concat([pol_df, p2018, p2019], axis=0)
del p2018, p2019, samp
pol_df.head()
samp = pol_df.loc[pol_df['Latitude'].notnull()]
map_sample = folium.Map(location=[40,-75], zoom_start=9)
HeatMap(data=samp[['Latitude', 'Longitude']], radius=10).add_to(map_sample)
map_sample
pol_df.isnull().sum()
pol_df.shape
to_drop = ['State Code', 'County Code', 'Site Num', 
           'Parameter Code', 'Latitude', 'Longitude',
          'Parameter Name', 'Sample Duration', 
           'Pollutant Standard']
df = pol_df.drop(to_drop, axis=1)
df.tail()
def get_uniques(df):
    
    for col in df.columns:
        if df[col].dtype=='object' and col != 'Date Local':
            print(f'Unique values for {col}: {df[col].unique()}')

get_uniques(df)
df.drop(['Datum', 'Units of Measure', 'Event Type', 
         'State Name', 'County Name', 'City Name',
        'CBSA Name'], axis=1, inplace=True)
df.loc[:, 'Date Local'] = pd.to_datetime(df['Date Local'])
df.set_index('Date Local', inplace=True)
df
t11 = df.loc[(df['Address']==df['Address'].unique()[0]) & (
    df['Method Name']==df['Method Name'].unique()[0])]
t12 = df.loc[(df['Address']==df['Address'].unique()[0]) & (
    df['Method Name']==df['Method Name'].unique()[1])]
t21 = df.loc[(df['Address']==df['Address'].unique()[1]) & (
    df['Method Name']==df['Method Name'].unique()[0])]
t22 = df.loc[(df['Address']==df['Address'].unique()[1]) & (
    df['Method Name']==df['Method Name'].unique()[1])]
plt.plot(t11['Arithmetic Mean'], label='parts per billion {}'.format(t11['Method Name'][0]))
plt.plot(t12['Arithmetic Mean'], label='parts per billion {}'.format(t12['Method Name'][0]))
plt.title('S02 particles concentraction at {}'.format(t11['Address'][0]))
plt.legend()
#plt.plot(t21['Observation Count'])
#plt.plot(t22['Observation Count'])
plt.plot(t21['Arithmetic Mean'], label='parts per billion {}'.format(t21['Method Name'][0]))
plt.plot(t22['Arithmetic Mean'], label='parts per billion {}'.format(t22['Method Name'][0]))
plt.title('S02 particles concentraction at {}'.format(t21['Address'][0]))
plt.legend()
plt.plot(t11['AQI'], label='method {}'.format(t11['Method Name'][0]))
plt.plot(t12['AQI'], label='method {}'.format(t12['Method Name'][0]))
plt.title('AQI at {}'.format(t11['Address'][0]))
plt.legend()
plt.plot(t21['AQI'], label='method: {}'.format(t21['Method Name'][0]))
plt.plot(t22['AQI'], label='method: {}'.format(t22['Method Name'][0]))
plt.title('AQI at {}'.format(t21['Address'][0]))
plt.legend()
# in case we will come back later
df = df.loc[(df['Address']==df['Address'].unique()[0]) & (
    df['Method Name']==df['Method Name'].unique()[0])]
df
df.drop(['Address', 'Method Name', 'Method Code', 'Local Site Name', 
         'Date of Last Change', 'POC', 'Observation Count', 
         'Observation Percent', '1st Max Hour'], axis=1, inplace=True)
df
df.isnull().sum()
# inputer_data = df.loc[df['AQI'].notnull()]
plt.plot(df.dropna()['Arithmetic Mean'][-100:], label='parts per billion')
plt.plot(df.dropna()['1st Max Value'][-100:], label='max value')
plt.plot(df.dropna()['AQI'][-100:], label='AQI')
plt.legend(loc = 'upper right')
# jovian.commit(project=project_name)
df #.loc[df.index > np.datetime64('2019-01-01')] 
def inpute_aqi(aqi, max_values, mv_pred, epochs=50):
    linreg = nn.Linear(1, 1)
    loss_f = F.mse_loss #.l1_loss
    opt = torch.optim.Adam(linreg.parameters(), lr=0.0001)
    if isinstance(aqi, np.ndarray):
        aqi_max = np.max(aqi)
        aqi = torch.from_numpy(aqi/aqi_max)
    if isinstance(max_values, np.ndarray):
        max_values_max = np.max(max_values)
        max_values = torch.from_numpy(max_values/max_values_max)
    if isinstance(mv_pred, np.ndarray):
        mv_pred_max = np.max(mv_pred)
        mv_pred = torch.from_numpy(mv_pred/max_values_max)
    for epoch in range(0, epochs):
        
        # Train with batches of data
        for x, y in zip(aqi, max_values):
            #print(x, y)
            out = F.relu(linreg(x))
            loss = loss_f(out, y)
            loss.backward()
            opt.step()
            opt.zero_grad()
        if (epoch+1) % 10 == 0:
            print('Epoch [{}/{}], Loss: {:.4f}'.format(
                epoch+1, epochs, loss.item()))
    l = [round(linreg(x).item()*max_values_max) for x in mv_pred]
    return l

# if you have NaN values for AQI use similar procedure for 
# value imputation
'''
aqi = np.array([[x] for x in df.dropna()['AQI'].values], dtype=np.float32)
max_values = np.array([[x] for x in df.dropna()['1st Max Value'].values], dtype=np.float32)
mv_pred = np.array([[x] for x in df.loc[df['AQI'].isnull()]['1st Max Value'].values], dtype=np.float32)
#print(aqi[0], max_values[0], mv_pred[0])
l = inpute_aqi(aqi, max_values, mv_pred)
df.loc[df['AQI'].isnull(), 'AQI'] = l
df
'''
def rescale(df):
    for col in df.columns:
        scalers = []
        if np.max(df[col].values)>1:
            scaler = MinMaxScaler()
            scaler.fit(df[col].values.reshape(-1, 1))
            df.loc[:, col] = scaler.transform(df[col].values.reshape(-1, 1))
            scalers.append(scaler)
    return df, scalers

df, scalers = rescale(df)
df
temp_df = df.copy()
temp_df = temp_df.reset_index().drop('Date Local', axis=1)
temp_df
class PollutionDataset(Dataset):
    
    def __init__(self, frame):
        super().__init__()
        self.frame = frame
        
    def __len__(self):
        return self.frame.shape[0]
    
    def __getitem__(self, ind):
        x = torch.tensor(ind).float()
        y = self.frame.loc[ind, :].values.astype(np.float32)
        return x, y
test_date = temp_df.index[int(len(temp_df)*0.85)]
batch_size = 16
train_df = PollutionDataset(temp_df[:test_date])
val_df = PollutionDataset(temp_df[test_date:])
train_loader = DataLoader(train_df,
                         shuffle=False)
val_loader = DataLoader(val_df,
                         shuffle=False)
temp_df[:test_date]
class PollutionModel(nn.Module):
    
    def __init__(self, out_features=3):
        super().__init__()
        self.input = nn.Linear(1, 32)
        self.lin1 = nn.Linear(32, 64)
        self.norm1 = nn.BatchNorm1d(64)
        self.lin2 = nn.Linear(64, 128)
        self.norm2 = nn.BatchNorm1d(128)
        self.lin3 = nn.Linear(128, 32)
        self.out = nn.Linear(32, 3)
    
    def forward(self, x):
        x = F.relu(self.input(x)) #.view(1, -1)
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = F.relu(self.lin3(x))
        return F.relu(self.out(x))
model = PollutionModel()
epochs = 50
lr = 0.001

loss_f = F.mse_loss
train_loader.dataset[0]
def fit(model, epochs=50, lr=0.001, loss_f=F.mse_loss):

    optimizer = Adam(model.parameters(), lr=lr)
    total_loss = []
    for epoch in tqdm(range(epochs)):
        model.train()
        for sample in train_loader:
            # print(sample)
            x, y = sample
            model.zero_grad()
            out = model(x)
            loss = loss_f(out, y)
            loss.backward()
            optimizer.step()
            total_loss.append(loss.item())
            optimizer.zero_grad()
    print(np.array(total_loss).mean())
    return model

model = fit(model)
def predict(model, val_loader):
    
    preds = []    
    model.eval()
    for sample in train_loader:
        x, y = sample
        model.zero_grad()
        out = model(x)
        preds.append(out.detach().numpy())
    return np.array(preds)

preds = predict(model, val_loader)
temp_df[test_date:]['AQI'], [x[-1] for x in preds]
def get_sequences(df, seq_len):
    xs = []
    ys = []
    for i in range(len(df)-seq_len-1):
        x = df[i:(i+seq_len)]
        y = df[i+seq_len]
        xs.append(x)
        ys.append(y)

    return np.array(xs), np.array(ys)

s_len = 7
test_size = int(len(df)*0.85)
train_seq = get_sequences(df.loc[:df.index[test_size], 'AQI'], s_len)
val_seq = get_sequences(df.loc[df.index[test_size]:, 'AQI'], s_len)
train_seq[0].shape, train_seq[1].shape
class StatelessModel(nn.Module):
    
    def __init__(self, n_features, n_hidden, seq_len=7, 
                 n_layers=2, out_features=1):
        super().__init__()
        self.n_hidden = n_hidden
        self.seq_len = seq_len
        self.n_layers = n_layers
        self.lstm = nn.LSTM(n_features, 
                            n_hidden, 
                            n_layers,
                            dropout=0.5)
        self.linear = nn.Linear(n_hidden, out_features)
    
    def reset_hidden_state(self):
        self.hidden = (torch.zeros(self.n_layers, self.seq_len, self.n_hidden),
                       torch.zeros(self.n_layers, self.seq_len, self.n_hidden))
    
    def forward(self, x):
        #print(type(x))
        x = torch.tensor(x).view(1, self.seq_len, -1)
        lstm_out, _ = self.lstm(x.float(), self.hidden)
        last_time_step = lstm_out.view(
            self.seq_len, len(x), self.n_hidden)[-1]
        x = self.linear(last_time_step)
        return torch.flatten(x)
def fit_lstm(model, data, epochs=50, 
             lr=0.0001, loss_f=F.mse_loss):

    optimizer = Adam(model.parameters(), lr=lr)
    x, y = data
    model.train()
    for epoch in range(epochs):
        total_loss = []
        for x_seq, y_seq in tqdm(zip(x, y)):
            model.zero_grad()
            model.reset_hidden_state()
            out = model(np.array([[i] for i in x_seq]))
            #print(out, y_seq)
            loss = loss_f(out, torch.tensor([y_seq]).float())#.float()
            #print(loss)
            loss.backward()
            optimizer.step()
            total_loss.append(loss.item())
            optimizer.zero_grad()
        if epoch+1 % 10 ==0:
            print(np.array(total_loss).mean())
    return model

model = StatelessModel(1, 256)
# batch_size = 14
# train_loader = DataLoader(train_seq, shuffle=False, batch_size=batch_size)
# val_loader = DataLoader(val_seq, shuffle=False, batch_size=batch_size)
model = fit_lstm(model, train_seq)
def evaluate(model, val_set):
    x, y = val_set
    diviance = []
    model.eval()
    total_loss = []
    for x_seq, y_seq in tqdm(zip(x, y)):
        #model.zero_grad()
        model.reset_hidden_state()
        out = model(np.array([[i] for i in x_seq]))
        #print(out, y_seq)
        loss = loss_f(out, torch.tensor([y_seq]).float())
        diviance.append(np.abs(out.detach().item()-y_seq))
        total_loss.append(loss.item())
        #optimizer.zero_grad()
    mean_loss = np.array(total_loss).mean()
    mean_div = np.array(diviance).mean()
    return mean_loss, mean_div

evaluate(model, val_seq)
import jovian
jovian.commit(project=project_name)
