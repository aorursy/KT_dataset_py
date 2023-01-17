# !pip3 install git+https://github.com/Kaggle/kaggle-api.git --upgrade
!pip3 install jovian --quiet
def setup_local(kaggle_creds='kaggle.json'):
    import os
    import json
    with open(kaggle_creds, 'r') as f:
        creds = json.loads(f.read())
    os.environ['KAGGLE_USERNAME']=creds['username']
    os.environ['KAGGLE_KEY']=creds['key']
    os.system('kaggle datasets download -d so2-emissions-daily-summary-data')
    os.system('unzip so2-emissions-daily-summary-data.zip')
    return ''

# provide a kaggle.json credentials and
# uncomment the line if you run localy
#
# data_dir = setup_local()
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler
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
project_name = 'air-pollution-case-p2'
author = 'Valentyna Fihurska'
try:
    data_dir
except Exception:
    data_dir = '../input/so2-emissions-daily-summary-data/'
pol_df = pd.read_csv(os.path.join(data_dir, 'daily_42401_2017/daily_42401_2017.csv'))
p2018 = pd.read_csv(os.path.join(data_dir, 'daily_42401_2018.csv'))
p2019 = pd.read_csv(os.path.join(data_dir, 'daily_42401_2019.csv'))
pol_df = pol_df.loc[(pol_df['County Code']==101) & (pol_df['State Code']==42)]
p2018 = p2018.loc[(p2018['County Code']==101) & (p2018['State Code']==42)]
p2019 = p2019.loc[(p2019['County Code']==101) & (p2019['State Code']==42)]
pol_df = pd.concat([pol_df, p2018, p2019], axis=0)
del p2018, p2019
to_drop = ['State Code', 'County Code', 'Site Num', 
           'Parameter Code', 'Latitude', 'Longitude',
          'Parameter Name', 'Sample Duration', 
           'Pollutant Standard']
df = pol_df.drop(to_drop, axis=1)
df.drop(['Datum', 'Units of Measure', 'Event Type', 
         'State Name', 'County Name', 'City Name',
        'CBSA Name'], axis=1, inplace=True)
df.loc[:, 'Date Local'] = pd.to_datetime(df['Date Local'])
df.set_index('Date Local', inplace=True)
df = df.loc[(df['Address']==df['Address'].unique()[0]) & (
    df['Method Name']==df['Method Name'].unique()[0])]
df.drop(['Address', 'Method Name', 'Method Code', 'Local Site Name', 
         'Date of Last Change', 'POC', 'Observation Count', 
         'Observation Percent', '1st Max Hour'], axis=1, inplace=True)
df
def save_n_commit(model, metrics=None):
    
    kw = ['arch', 'epochs', 'lr', 
          'scheduler', 'weight_decay', 
          'grad_clip', 'opt', 'val_loss',
          'val_score', 'train_loss', 
          'train_time', 'loss_func']
    
    if metrics:
        for k in kw:
            try:
                metrics[k]
            except Exception:
                metrics[k] = None
    else:
        metrics = {k: None for k in kw}
    
    weights_fname = '{}-{}ep-{}lr.pth'.format(metrics['arch'],
                                              metrics['epochs'],
                                              metrics['lr'])
    torch.save(model.state_dict(), weights_fname)
    jovian.reset()

    jovian.log_hyperparams(arch=metrics['arch'], 
                           epochs=metrics['epochs'], 
                           lr=metrics['lr'], 
                           scheduler=metrics['scheduler'], 
                           weight_decay=metrics['weight_decay'], 
                           grad_clip=metrics['grad_clip'],
                           opt=metrics['opt'],
                           val_loss=metrics['val_loss'],
                           val_score=metrics['val_score'],
                           train_loss=metrics['train_loss'],
                           train_time=metrics['train_time'],
                          loss_func=metrics['loss_func'])
    
    jovian.commit(project=project_name, 
                  environment=None, 
                  outputs=[weights_fname])
    return True
def get_sequences(df, seq_len):
    xs = []
    ys = []
    for i in range(len(df)-seq_len-1):
        x = df[i:(i+seq_len)].values
        y = df.loc[df.index[i+seq_len]]#.values
        xs.append(x)
        ys.append(y)

    return np.array(xs), np.array(ys)
def plot_dist(df):
    fig, ax = plt.subplots(1, len(df.columns))
    for i, col in zip(range(len(df.columns)), 
                      df.columns):
        sns.distplot(df[col], ax=ax[i], axlabel=col)

plot_dist(df)
def log_scale(df, invert=False):
    for col in df.columns:
        if invert:
            df.loc[:, col] = np.exp(df[col]) #- 1
            continue
        df.loc[:, col] = np.log1p(df[col])
    return df
class PollutionDataset(Dataset):
    
    def __init__(self, frame, window=14):
        super().__init__()
        self.frame = frame
        self.window = window
        self.x, self.y = get_sequences(self.frame,
                            seq_len=self.window)
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, ind):
        x = np.array(self.x[ind])
        y = np.array(self.y[ind])
        return torch.tensor(x), torch.tensor(y)
    
    def stationary_conversion(self):
        self.frame = self.frame.diff()
        print(self.frame)
    
    def scale_features(self, scaler='custom'):
        if scaler == 'custom':
            scalers = self.get_scalers()
            for i in range(len(self.frame.columns)):
                col = self.frame.columns[i]
                self.frame.loc[:, col] = self.frame[col]/scalers[i]
            return
        scalers = []
        for col in self.frame.columns:
            if scaler=='standart':
                sc = StandardScaler()
            if scaler=='minmax':
                sc = MinMaxScaler()
            sc.fit(self.frame[col].values.reshape(-1, 1))
            self.frame.loc[:, col] = sc.transform(
                self.frame[col].values.reshape(-1, 1))
            scalers.append(sc)
        self.scalers = scalers
        
    def get_scalers(self):
        scalers = list()
        scalers.append(self.frame[self.frame.columns[0]].values.max())
        scalers.append(self.frame[self.frame.columns[1]].values.max())
        scalers.append(self.frame[self.frame.columns[2]].values.max())
        self.scalers = scalers
        return self.scalers
        
train_size = int(0.85*len(df))
train_set = PollutionDataset(df[:train_size])
val_set = PollutionDataset(df[train_size:])
class BaseModel(nn.Module):
    
    def training_step(self, x, y, loss_f=F.mse_loss):
        x, y = x.float(), y.float()
        out = self(x)
        #print(out)
        loss = loss_f(out, y)      
        return loss
    
    def validation_step(self, x, y, loss_f=F.mse_loss):
        x, y = x.float(), y.float()
        out = self(x)
        loss = loss_f(out, y)
        return {'val_loss': loss.detach() }
        
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()
        return {'val_loss': epoch_loss.item()}
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], last_lr: {}, train_loss: {:.4f}, val_loss: {:.4f}".format(
            epoch, result['lrs'][-1], result['train_loss'], result['val_loss']))
def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
    
def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        
    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl: 
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)
class StatefulModel(BaseModel):
    
    def __init__(self, n_features, n_hidden, seq_len=14, 
                 n_layers=2, out_features=1):
        super().__init__()
        self.n_features = n_features
        self.n_hidden = n_hidden
        self.seq_len = seq_len
        self.n_layers = n_layers
        self.init_hidden()
        self.lstm1 = nn.LSTM(n_features, 
                            n_hidden, 
                            n_layers,
                            dropout=0.5)
        self.lstm2 = nn.LSTM(n_hidden, 
                            n_hidden, 
                            n_layers,
                            dropout=0.5)
        self.linear = nn.Linear(n_hidden, out_features)
    
    def init_hidden(self):
        self.hidden = (torch.zeros(self.n_layers, self.seq_len, self.n_hidden),
                       torch.zeros(self.n_layers, self.seq_len, self.n_hidden))
    
    def forward(self, x):
        x = x.view(1, self.seq_len, -1).float()
        #print(x.view(1, self.seq_len, -1))
        lstm_out, self.hidden = self.lstm1(x, self.hidden)
        lstm_out, self.hidden = self.lstm2(lstm_out, 
                                           self.hidden)
        last_time_step = lstm_out.view(
            self.seq_len, len(x), self.n_hidden)[-1]
        #last_time_step = torch.flatten(last_time_step)
        x = self.linear(last_time_step)
        #print(x)
        self.hidden = self.hidden[0].detach(), self.hidden[1].detach()
        return torch.flatten(x)
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def fit(model, train_set, val_set, subset, epochs=10, 
        lr=1e-4, loss_f=F.mse_loss, save=False):
    
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-06)
    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, 
                                                 base_lr=lr, 
                                                 max_lr=1e-6, 
                                                 step_size_up=int(len(train_set)/2),
                                                 step_size_down=int(len(train_set)/2),
                                                 cycle_momentum=False)
    model.train()
    outputs = []
    for epoch in range(epochs):
        total_loss = []
        lrs = []
        for each in tqdm(train_set):
            x, y = each
            x = x.T[subset]
            #x = x.view(1, 30, -1)
            y = y[subset].view(1, )
            #print(x, y)
            model.zero_grad()
            loss = model.training_step(x, y)
            loss.backward()
            optimizer.step()
            scheduler.step()
            lrs.append(get_lr(optimizer))
            total_loss.append(loss.item())
            optimizer.zero_grad()
            
        t_loss = np.array(total_loss).mean()
        print('Epoch: {}, loss: {}, last lr: {}'.format(
            epoch, t_loss, lrs[-1]))
    if save:
        save_n_commit(model, metrics={'arch': str(model),
                                     'epochs': epochs,
                                      'last_lr': lrs[-1],
                                     'loss_func': loss_f.__name__,
                                     'opt': str(optimizer),
                                     'train_loss': t_loss})
    return model
model_am = StatefulModel(1, 512)
model_mv = StatefulModel(1, 512)
model_aqi = StatefulModel(1, 512)
fit(model_am, train_set, val_set, 0, save=True)
fit(model_mv, train_set, val_set, 1, save=True)
fit(model_aqi, train_set, val_set, 2, save=True)
def evaluate(model, val_set, subset, return_preds=True):
    #x, y = val_set
    diviance = []
    model.eval()
    if return_preds:
        preds = []
    for each in tqdm(val_set):
        x, y = each
        x = x.T[subset].float()
        y = torch.tensor(y[subset]).view(1, )
        out = model(x).detach().item()
        if return_preds:
            preds.append(out)
        diviance.append(np.abs(out-y))
    mean_div = np.array(diviance).mean()

    if return_preds:
        return mean_div, np.array(preds)
    return mean_div
mean_div_am, preds_am = evaluate(
    model_am, val_set, 0)
mean_div_mv, preds_mv = evaluate(
    model_mv, val_set, 1)
mean_div_aqi, preds_aqi = evaluate(
    model_aqi, val_set, 2)
preds_am
true_am = np.array([val_set[i][1][0] for i in range(len(val_set))])
true_mv = np.array([val_set[i][1][1] for i in range(len(val_set))])
true_aqi = np.array([val_set[i][1][2] for i in range(len(val_set))])
for i in range(1, len(preds_am)):
    print('true:', true_am[i-1], '\tpredicted:', preds_am[i])
    if i > 10:
        break
plt.plot(true_am, label='True parts per billion')
plt.plot(preds_am, label='Predicted parts per billion')
plt.legend()
plt.xticks([]);
for i in range(1, len(preds_mv)):
    print('true:', true_mv[i-1], '\tpredicted:', preds_mv[i])
    if i > 10:
        break
plt.plot(true_mv, label='True maximum parts per billion')
plt.plot(preds_mv, label='Predicted maximum parts per billion')
plt.legend()
plt.xticks([]);
for i in range(1, len(preds_am)):
    print('true:', true_aqi[i-1], '\tpredicted:', preds_aqi[i])
    if i > 10:
        break
plt.plot(true_aqi, label='True AQI')
plt.plot(preds_aqi, label='Predicted AQI')
plt.legend()
plt.xticks([]);
import jovian
jovian.commit(project=project_name, environment=None)
