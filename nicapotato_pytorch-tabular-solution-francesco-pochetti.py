%matplotlib inline
import pathlib
import matplotlib.pyplot as plt
from pylab import rcParams
rcParams['figure.figsize'] = 8, 6
import pandas as pd
import numpy as np
import seaborn as sns
pd.set_option('display.max_columns', 500)
from collections import defaultdict

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity='all'

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error

pd.options.mode.chained_assignment = None

from torch.nn import init
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils import data
from torch.optim import lr_scheduler

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device

from tqdm import tqdm, tqdm_notebook, tnrange
tqdm.pandas(desc='Progress')
def haversine_distance(df, start_lat, end_lat, start_lng, end_lng, prefix):
    """
    calculates haversine distance between 2 sets of GPS coordinates in df
    """
    R = 6371  #radius of earth in kilometers
       
    phi1 = np.radians(df[start_lat])
    phi2 = np.radians(df[end_lat])
    
    delta_phi = np.radians(df[end_lat]-df[start_lat])
    delta_lambda = np.radians(df[end_lng]-df[start_lng])
    
        
    a = np.sin(delta_phi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2.0) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    d = (R * c) #in kilometers
    df[prefix+'distance_km'] = d

def add_datepart(df, col, prefix):
    attr = ['Year', 'Month', 'Week', 'Day', 'Dayofweek', 'Dayofyear',
            'Is_month_end', 'Is_month_start', 'Is_quarter_end', 'Is_quarter_start', 'Is_year_end', 'Is_year_start']
    attr = attr + ['Hour', 'Minute', 'Second']
    for n in attr: df[prefix + n] = getattr(df[col].dt, n.lower())
    df[prefix + 'Elapsed'] = df[col].astype(np.int64) // 10 ** 9
    df.drop(col, axis=1, inplace=True)
    
def reject_outliers(data, m = 2.):
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d/(mdev if mdev else 1.)
    return s<m

def parse_gps(df, prefix):
    lat = prefix + '_latitude'
    lon = prefix + '_longitude'
    df[prefix + '_x'] = np.cos(df[lat]) * np.cos(df[lon])
    df[prefix + '_y'] = np.cos(df[lat]) * np.sin(df[lon]) 
    df[prefix + '_z'] = np.sin(df[lat])
    df.drop([lat, lon], axis=1, inplace=True)
    
def prepare_dataset(df):
    df['pickup_datetime'] = pd.to_datetime(df.pickup_datetime, infer_datetime_format=True)
    add_datepart(df, 'pickup_datetime', 'pickup')
    haversine_distance(df, 'pickup_latitude', 'dropoff_latitude', 'pickup_longitude', 'dropoff_longitude', '')
    parse_gps(df, 'pickup')
    parse_gps(df, 'dropoff')
    df.dropna(inplace=True)
    y = np.log(df.fare_amount)
    df.drop(['key', 'fare_amount'], axis=1, inplace=True)
    
    return df, y

def split_features(df):
    catf = ['pickupYear', 'pickupMonth', 'pickupWeek', 'pickupDay', 'pickupDayofweek', 
            'pickupDayofyear', 'pickupHour', 'pickupMinute', 'pickupSecond', 'pickupIs_month_end',
            'pickupIs_month_start', 'pickupIs_quarter_end', 'pickupIs_quarter_start',
            'pickupIs_year_end', 'pickupIs_year_start']

    numf = [col for col in df.columns if col not in catf]
    for c in catf: 
        df[c] = df[c].astype('category').cat.as_ordered()
        df[c] = df[c].cat.codes+1
    
    return catf, numf

def numericalize(df):
    df[name] = col.cat.codes+1

def split_dataset(df, y): return train_test_split(df, y, test_size=0.25, random_state=42)

def inv_y(y): return np.exp(y)

def get_numf_scaler(train): return preprocessing.StandardScaler().fit(train)

def scale_numf(df, num, scaler):
    cols = numf
    index = df.index
    scaled = scaler.transform(df[numf])
    scaled = pd.DataFrame(scaled, columns=cols, index=index)
    return pd.concat([scaled, df.drop(numf, axis=1)], axis=1)

class RegressionColumnarDataset(data.Dataset):
    def __init__(self, df, cats, y):
        self.dfcats = df[cats]
        self.dfconts = df.drop(cats, axis=1)
        
        self.cats = np.stack([c.values for n, c in self.dfcats.items()], axis=1).astype(np.int64)
        self.conts = np.stack([c.values for n, c in self.dfconts.items()], axis=1).astype(np.float32)
        self.y = y.values.astype(np.float32)
        
    def __len__(self): return len(self.y)

    def __getitem__(self, idx):
        return [self.cats[idx], self.conts[idx], self.y[idx]]
    
def rmse(targ, y_pred):
    return np.sqrt(mean_squared_error(inv_y(y_pred), inv_y(targ))) #.detach().numpy()

def emb_init(x):
    x = x.weight.data
    sc = 2/(x.size(1)+1)
    x.uniform_(-sc,sc)

class MixedInputModel(nn.Module):
    def __init__(self, emb_szs, n_cont, emb_drop, out_sz, szs, drops, y_range, use_bn=True):
        super().__init__()
        for i,(c,s) in enumerate(emb_szs): assert c > 1, f"cardinality must be >=2, got emb_szs[{i}]: ({c},{s})"
        self.embs = nn.ModuleList([nn.Embedding(c, s) for c,s in emb_szs])
        for emb in self.embs: emb_init(emb)
        n_emb = sum(e.embedding_dim for e in self.embs)
        self.n_emb, self.n_cont=n_emb, n_cont
        
        szs = [n_emb+n_cont] + szs
        self.lins = nn.ModuleList([nn.Linear(szs[i], szs[i+1]) for i in range(len(szs)-1)])
        self.bns = nn.ModuleList([nn.BatchNorm1d(sz) for sz in szs[1:]])
        for o in self.lins: nn.init.kaiming_normal_(o.weight.data)
        self.outp = nn.Linear(szs[-1], out_sz)
        nn.init.kaiming_normal_(self.outp.weight.data)

        self.emb_drop = nn.Dropout(emb_drop)
        self.drops = nn.ModuleList([nn.Dropout(drop) for drop in drops])
        self.bn = nn.BatchNorm1d(n_cont)
        self.use_bn,self.y_range = use_bn,y_range

    def forward(self, x_cat, x_cont):
        if self.n_emb != 0:
            x = [e(x_cat[:,i]) for i,e in enumerate(self.embs)]
            x = torch.cat(x, 1)
            x = self.emb_drop(x)
        if self.n_cont != 0:
            x2 = self.bn(x_cont)
            x = torch.cat([x, x2], 1) if self.n_emb != 0 else x2
        for l,d,b in zip(self.lins, self.drops, self.bns):
            x = F.relu(l(x))
            if self.use_bn: x = b(x)
            x = d(x)
        x = self.outp(x)
        if self.y_range:
            x = torch.sigmoid(x)
            x = x*(self.y_range[1] - self.y_range[0])
            x = x+self.y_range[0]
        return x.squeeze()

def fit(model, train_dl, val_dl, loss_fn, opt, scheduler, epochs=3):
    num_batch = len(train_dl)
    for epoch in tnrange(epochs):      
        y_true_train = list()
        y_pred_train = list()
        total_loss_train = 0          
        
        t = tqdm_notebook(iter(train_dl), leave=False, total=num_batch)
        for cat, cont, y in t:
            cat = cat.cuda()
            cont = cont.cuda()
            y = y.cuda()
            
            t.set_description(f'Epoch {epoch}')
            
            opt.zero_grad()
            pred = model(cat, cont)
            loss = loss_fn(pred, y)
            loss.backward()
            lr[epoch].append(opt.param_groups[0]['lr'])
            tloss[epoch].append(loss.item())
            scheduler.step()
            opt.step()
            
            t.set_postfix(loss=loss.item())
            
            y_true_train += list(y.cpu().data.numpy())
            y_pred_train += list(pred.cpu().data.numpy())
            total_loss_train += loss.item()
            
        train_acc = rmse(y_true_train, y_pred_train)
        train_loss = total_loss_train/len(train_dl)
        
        if val_dl:
            y_true_val = list()
            y_pred_val = list()
            total_loss_val = 0
            for cat, cont, y in tqdm_notebook(val_dl, leave=False):
                cat = cat.cuda()
                cont = cont.cuda()
                y = y.cuda()
                pred = model(cat, cont)
                loss = loss_fn(pred, y)
                
                y_true_val += list(y.cpu().data.numpy())
                y_pred_val += list(pred.cpu().data.numpy())
                total_loss_val += loss.item()
                vloss[epoch].append(loss.item())
            valacc = rmse(y_true_val, y_pred_val)
            valloss = total_loss_val/len(valdl)
            print(f'Epoch {epoch}: train_loss: {train_loss:.4f} train_rmse: {train_acc:.4f} | val_loss: {valloss:.4f} val_rmse: {valacc:.4f}')
        else:
            print(f'Epoch {epoch}: train_loss: {train_loss:.4f} train_rmse: {train_acc:.4f}')
    
    return lr, tloss, vloss
PATH='../input/'
names = ['key','fare_amount','pickup_datetime','pickup_longitude',
         'pickup_latitude','dropoff_longitude','dropoff_latitude','passenger_count']
df = pd.read_csv(f'{PATH}train.csv', nrows=6000000)

print(df.shape)
print(df.head())
print(df.passenger_count.describe())
print(df.passenger_count.quantile([.85, .99]))
print(df.fare_amount.describe())
print(df.fare_amount.quantile([.85, .99]))
df = df.loc[(df.fare_amount > 0) & (df.passenger_count < 6) & (df.fare_amount < 53),:]
df, y = prepare_dataset(df)

print(df.shape)
print(df.head())
ax = y.hist(bins=20, figsize=(8,6))
_ = ax.set_xlabel("Ride Value (EUR)")
_ = ax.set_ylabel("# Rides")
_ = ax.set_title('Ditribution of Ride Values (USD)')
catf, numf = split_features(df)

len(catf)
catf

len(numf)
numf
df.head()
y_range = (0, y.max()*1.2)
y_range

y = y.clip(y_range[0], y_range[1])
X_train, X_test, y_train, y_test = split_dataset(df, y)

X_train.shape
X_test.shape
scaler = get_numf_scaler(X_train[numf])

X_train_sc = scale_numf(X_train, numf, scaler)
X_train_sc.std(axis=0)
X_test_sc = scale_numf(X_test, numf, scaler)

X_train_sc.shape
X_test_sc.shape
X_test_sc.std(axis=0)
trainds = RegressionColumnarDataset(X_train_sc, catf, y_train)
valds = RegressionColumnarDataset(X_test_sc, catf, y_test)
class RegressionColumnarDataset(data.Dataset):
    def __init__(self, df, cats, y):
        self.dfcats = df[cats]
        self.dfconts = df.drop(cats, axis=1)
        
        self.cats = np.stack([c.values for n, c in self.dfcats.items()], axis=1).astype(np.int64)
        self.conts = np.stack([c.values for n, c in self.dfconts.items()], axis=1).astype(np.float32)
        self.y = y.values.astype(np.float32)
        
    def __len__(self): return len(self.y)

    def __getitem__(self, idx):
        return [self.cats[idx], self.conts[idx], self.y[idx]]
params = {'batch_size': 128,
          'shuffle': True,
          'num_workers': 8}

traindl = data.DataLoader(trainds, **params)
valdl = data.DataLoader(valds, **params)
cat_sz = [(c, df[c].max()+1) for c in catf]
cat_sz

emb_szs = [(c, min(50, (c+1)//2)) for _,c in cat_sz]
emb_szs
m = MixedInputModel(emb_szs=emb_szs, 
                    n_cont=len(df.columns)-len(catf), 
                    emb_drop=0.04, 
                    out_sz=1, 
                    szs=[1000,500,250], 
                    drops=[0.001,0.01,0.01], 
                    y_range=y_range).to(device)

opt = optim.Adam(m.parameters(), 1e-2)
lr_cosine = lr_scheduler.CosineAnnealingLR(opt, 1000)

lr = defaultdict(list)
tloss = defaultdict(list)
vloss = defaultdict(list)
m
epoch_n = 12

lr, tloss, vloss = fit(model=m, train_dl=traindl, val_dl=valdl, loss_fn=F.mse_loss, opt=opt, scheduler=lr_cosine, epochs=epoch_n)
_ = plt.plot(lr[0])
_ = plt.title('Learning Rate Cosine Annealing over Train Batches Iterations (Epoch 0)')
t = [np.mean(tloss[el]) for el in tloss]
v = [np.mean(vloss[el]) for el in vloss]
p = pd.DataFrame({'Train Loss': t, 'Validation Loss': v, 'Epochs': range(1, epoch_n+1)})

_ = p.plot(x='Epochs', y=['Train Loss', 'Validation Loss'], 
           title='Train and Validation Loss over Epochs')
