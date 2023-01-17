import numpy as np

import dask.dataframe as dd

import pandas as pd

import pandas_profiling

from tqdm import tqdm_notebook as tqdm

from statsmodels.tsa.stattools import adfuller

from scipy import stats

from IPython.display import display

from sklearn.preprocessing import MinMaxScaler, StandardScaler

from sklearn.metrics import mean_squared_log_error

import os, re, pickle, datetime, gc, shutil

import seaborn as sns



import torch

import torch.nn as nn

import torch.nn.functional as F

from torch.nn.utils.rnn import pad_sequence

from torch.utils.data import Dataset, DataLoader



gc.enable()



building_df   = pd.read_csv("../input/ashrae-energy-prediction/building_metadata.csv")

weather_train = pd.read_csv("../input/ashrae-energy-prediction/weather_train.csv")

train         = pd.read_csv("../input/ashrae-energy-prediction/train.csv")



weather_train.sort_values(['site_id','timestamp'], inplace=True)

weather_train.fillna(method='ffill', inplace=True)

weather_train.fillna(method='bfill', inplace=True)

weather_train.isnull().sum().sum()
time_cols = [col for col in weather_train.columns if col not in ['site_id', 'timestamp', 'cloud_coverage']]

weather_scalers = {}



for col in tqdm(time_cols):

    Scaler = StandardScaler()

    weather_train[col] = Scaler.fit_transform(weather_train[col].values.reshape(-1, 1))

    assert Scaler.n_samples_seen_ == len(weather_train[col])

    weather_scalers[col] = Scaler

    

pickle.dump(weather_scalers, open('weather_scalers.pkl','wb'))
building_df['square_feet'] = np.log(building_df['square_feet'])

mean_per_site = pd.DataFrame(building_df.groupby('site_id').mean()).reset_index()

mean_entire_col = pd.DataFrame(building_df.mean()).reset_index()

mean_entire_col.columns = ['column','mean']

cols_with_nulls = building_df.columns[building_df.isna().any()].tolist()
for site_id in tqdm(pd.unique(building_df['site_id'])):

    for col in tqdm(cols_with_nulls, leave=False):  

        mean = mean_per_site.loc[mean_per_site['site_id']==site_id, col].values

        if np.isnan(mean)[0]:

            mean = mean_entire_col.loc[mean_entire_col['column']==col, 'mean'].values

            

        building_df[col] = building_df[col].mask((building_df['site_id']==site_id) & (np.isnan(building_df[col])), mean[0])

        

building_df.isnull().sum().sum()



del mean_per_site, mean_entire_col, cols_with_nulls

gc.collect()
pickle.dump(building_df, open('building_df.pkl','wb'))
train = train.merge(building_df, left_on = "building_id", right_on = "building_id", how = "left")

train = train.merge(weather_train, left_on = ["site_id", "timestamp"], right_on = ["site_id", "timestamp"])
del building_df, weather_train

gc.collect()

train.isnull().sum().sum()
#Based on this great kernel https://www.kaggle.com/arjanso/reducing-dataframe-memory-size-by-65

def reduce_mem_usage(df):

    start_mem_usg = df.memory_usage().sum() / 1024**2 

    print("Memory usage of properties dataframe is :",start_mem_usg," MB")

    NAlist = [] # Keeps track of columns that have missing values filled in. 

    for col in df.columns:

        if df[col].dtype != object:  # Exclude strings            

            # Print current column type

            print("******************************")

            print("Column: ",col)

            print("dtype before: ",df[col].dtype)            

            # make variables for Int, max and min

            IsInt = False

            mx = df[col].max()

            mn = df[col].min()

            print("min for this col: ",mn)

            print("max for this col: ",mx)

            # Integer does not support NA, therefore, NA needs to be filled

            if not np.isfinite(df[col]).all(): 

                NAlist.append(col)

                df[col].fillna(mn-1,inplace=True)  

                   

            # test if column can be converted to an integer

            asint = df[col].fillna(0).astype(np.int64)

            result = (df[col] - asint)

            result = result.sum()

            if result > -0.01 and result < 0.01:

                IsInt = True            

            # Make Integer/unsigned Integer datatypes

            if IsInt:

                if mn >= 0:

                    if mx < 255:

                        df[col] = df[col].astype(np.int8)

                    elif mx < 65535:

                        df[col] = df[col].astype(np.int16)

                    elif mx < 4294967295:

                        df[col] = df[col].astype(np.int32)

                    else:

                        df[col] = df[col].astype(np.int64)

                else:

                    if mn > np.iinfo(np.int8).min and mx < np.iinfo(np.int8).max:

                        df[col] = df[col].astype(np.int8)

                    elif mn > np.iinfo(np.int16).min and mx < np.iinfo(np.int16).max:

                        df[col] = df[col].astype(np.int16)

                    elif mn > np.iinfo(np.int32).min and mx < np.iinfo(np.int32).max:

                        df[col] = df[col].astype(np.int32)

                    elif mn > np.iinfo(np.int64).min and mx < np.iinfo(np.int64).max:

                        df[col] = df[col].astype(np.int64)    

            # Make float datatypes 32 bit

            else:

                df[col] = df[col].astype(np.float32)

            

            # Print new column type

            print("dtype after: ",df[col].dtype)

            print("******************************")

    # Print final result

    print("___MEMORY USAGE AFTER COMPLETION:___")

    mem_usg = df.memory_usage().sum() / 1024**2 

    print("Memory usage is: ",mem_usg," MB")

    print("This is ",100*mem_usg/start_mem_usg,"% of the initial size")

    return df, NAlist
train, NAlist = reduce_mem_usage(train)

NAlist
train.head()
train["timestamp"] = pd.to_datetime(train["timestamp"])

train["hour"]      = train["timestamp"].dt.hour.astype(np.int8)

train["day"]       = train["timestamp"].dt.day.astype(np.int8)

train["weekday"]   = train["timestamp"].dt.weekday.astype(np.int8)

train["month"]     = train["timestamp"].dt.month.astype(np.int8)
categoricals = ["site_id", 

                "building_id",

                "cloud_coverage", 

                "primary_use", 

                "hour", 

                "day", 

                "weekday", 

                "month", 

                "meter"]

numericals = ['meter_reading',

              'square_feet',

              'year_built',

              'floor_count',

              'air_temperature',

              'dew_temperature',

              'precip_depth_1_hr',

              'sea_level_pressure',

              'wind_direction',

              'wind_speed']



feat_cols = categoricals + numericals
train = train.astype({cat:'category' for cat in categoricals})



def Cat2IDs(data, categoricals_cols, IX_start):

    encode = {}

    for col in categoricals_cols:

        encode[col] = dict(enumerate(data[col].cat.categories, start=IX_start)) 



    Cat2Int = {}

    for cat in categoricals_cols:

        ValueKey = {}

        for key, value in encode[cat].items():

            ValueKey[value] = key

        Cat2Int[cat] = ValueKey

    return Cat2Int

    

Cat2Int = Cat2IDs(data = train, 

                  categoricals_cols = categoricals, 

                  IX_start = 2) #index 0 will be for padding and index 1 will be for OOV 

categorical_sizes = [train[c].nunique() + 2 for c in categoricals]



pickle.dump(Cat2Int, open('Cat2Int.pkl','wb'))
for col in categoricals:

    train[col] = train[col].map(Cat2Int[col])
for col in numericals:

    print(col,':')

    stat, p = stats.normaltest(train[col])

    print('normal test: Statistics=%.3f, p=%.3f' % (stat, p))

    alpha = 0.05

    if p > alpha:

        print('Gaussian: True')  #fail to reject H0

    else:

        print('Gaussian: False') #reject H0

    

    print( 'Kurtosis of normal distribution: {}'.format(stats.kurtosis(train[col])))

    print( 'Skewness of normal distribution: {}'.format(stats.skew(train[col])))

    print()
len(train[train['meter_reading']<0])
MMScaler = MinMaxScaler()

train['meter_reading'] = MMScaler.fit_transform(train['meter_reading'].values.reshape(-1,1))

pickle.dump(MMScaler, open('target_Scaler.pkl', 'wb'))

del MMScaler

gc.collect()
train.building_id = train.building_id.astype('int16')

train.reset_index(inplace=True)

train.set_index(['building_id','timestamp'], drop=False, inplace=True)

train.sort_index(inplace=True)

train.head()
numericals.remove('meter_reading')
%%time

X_train_cat = train[categoricals]

X_train_num = train[numericals]

y_train     = train[['meter_reading']]
del train

gc.collect()
print(X_train_cat.shape)

print(X_train_num.shape)

print(y_train.shape)   
train_counts = X_train_cat.building_id.value_counts().to_frame('counts').sort_index()

train_counts = train_counts[train_counts.counts >0]

train_counts['cumsum_counts'] = train_counts.counts.cumsum()

train_counts.head()
print('average number of samples per building train:', train_counts.counts.mean())

print('maximum number of samples per building train:', train_counts.counts.max())
%%time

X_train_cat = X_train_cat.values

X_train_num = X_train_num.values

y_train     = y_train.values
%%time

train_splits = train_counts.drop(train_counts.tail(1).index).cumsum_counts



X_train_cat  = np.split(X_train_cat, train_splits)

X_train_num  = np.split(X_train_num, train_splits)

y_train      = np.split(y_train,     train_splits)

 

del train_splits

gc.collect()
def create_path(path_list):

    for directory in path_list:

        if not os.path.isdir(directory):

            os.makedirs(directory)

            os.makedirs(f'{directory}/categorical')

            os.makedirs(f'{directory}/numerical')

            os.makedirs(f'{directory}/Ys')

        else:

            shutil.rmtree(directory)

            os.makedirs(directory)

            os.makedirs(f'{directory}/categorical')

            os.makedirs(f'{directory}/numerical')

            os.makedirs(f'{directory}/Ys')

            

create_path(path_list=['train',

                       'validation'])
def save_data(train_size, TBPTT, Building_IDs, X_Cat, X_Num, Ys):     

        

    for cats_values, nums_values, y_values, building_id in tqdm(zip(X_Cat, 

                                                                    X_Num, 

                                                                    Ys, 

                                                                    Building_IDs), 

                                                                total=len(X_Cat)):           

        

        X_cat_loop, X_num_loop, y_loop = [], [], []

        for i in range(TBPTT, cats_values.shape[0]+TBPTT, TBPTT):



            cat = cats_values[i-TBPTT:i]

            num = nums_values[i-TBPTT:i]

            y_v = y_values[i-TBPTT:i]



            # most of the time the number of data point in each building did not divide to pices of 100 so I need to pad them

            # the last batch in the most of the time did not equal to TBPTT to I pad them:

            zero_shape = cat.shape[0]

            if zero_shape != TBPTT:



                cat_pad = np.zeros((TBPTT, cat.shape[1]))

                num_pad = np.zeros((TBPTT, num.shape[1]))

                y_v_pad = np.zeros((TBPTT, y_v.shape[1]))



                cat_pad[:zero_shape,:] = cat

                num_pad[:zero_shape,:] = num

                y_v_pad[:zero_shape,:] = y_v



                cat = cat_pad

                num = num_pad

                y_v = y_v_pad



            X_cat_loop.append(cat)

            X_num_loop.append(num)

            y_loop.append(    y_v)



        X_cat_loop = np.array(X_cat_loop)

        X_num_loop = np.array(X_num_loop)

        y_loop     = np.array(y_loop)



        train_cat = X_cat_loop[:-train_size]

        val_cat   = X_cat_loop[-train_size:]

        train_num = X_num_loop[:-train_size]

        val_num   = X_num_loop[-train_size:]

        train_Y   = y_loop[:-train_size]

        val_Y     = y_loop[-train_size:]



        np.save(f'train/categorical/'     +str(building_id)+'.npy', train_cat)

        np.save(f'train/numerical/'       +str(building_id)+'.npy', train_num)

        np.save(f'train/Ys/'              +str(building_id)+'.npy', train_Y)



        np.save(f'validation/categorical/'+str(building_id)+'.npy', val_cat)

        np.save(f'validation/numerical/'  +str(building_id)+'.npy', val_num)

        np.save(f'validation/Ys/'         +str(building_id)+'.npy', val_Y)
TBPTT = 48



save_data(train_size  = 4,

          TBPTT       = TBPTT, 

          Building_IDs= list(train_counts.index), 

          X_Cat       = X_train_cat, 

          X_Num       = X_train_num, 

          Ys          = y_train)
del X_train_cat, X_train_num, y_train

gc.collect()
train_counts.index.name = 'building_id'

train_counts.reset_index(inplace=True)

train_counts.sort_values('counts',inplace=True)

train_counts.head()
class ASHRAE_Dataset(Dataset):

    def __init__(self, 

                 data, 

                 cat_path, 

                 num_path, 

                 label_path, 

                 Reduce_Data=False, 

                 pct_reduce=1, 

                 is_val=False):

        

        self.data        = data        

        self.cat_path    = cat_path

        self.num_path    = num_path

        self.label_path  = label_path        

        self.len         = self.data.shape[0]

        self.Reduce_Data = Reduce_Data

        self.pct_reduce  = pct_reduce

        self.is_val      = is_val

        

    def __getitem__(self,IXs):



        B_ID = self.data.loc[IXs, 'building_id']



        cats  = np.load(self.cat_path  +str(B_ID)+'.npy', allow_pickle=True)

        nums  = np.load(self.num_path  +str(B_ID)+'.npy', allow_pickle=True)

        label = np.load(self.label_path+str(B_ID)+'.npy', allow_pickle=True)



        # if is not validation set - reduce the data

        if self.Reduce_Data and not self.is_val:   

            

            seq_len     = cats.shape[0]

            new_seq_len = int(seq_len * self.pct_reduce) 

            start_at = np.random.randint(low  = 0, 

                                         high = seq_len - new_seq_len)

            

            cats  =  cats[start_at : start_at + new_seq_len]

            nums  =  nums[start_at : start_at + new_seq_len]

            label = label[start_at : start_at + new_seq_len]

        

        cats = cats.astype(np.int16)



        return [cats, nums, label]

    

    def __len__(self):

        return self.len
class MyCollator(object):

    def __init__(self):

        pass

    

    def __call__(self, batch):

        

        cat   = [torch.tensor(item[0]) for item in batch]

        num   = [torch.tensor(item[1]) for item in batch]

        label = [torch.tensor(item[2]) for item in batch]



        cat   = pad_sequence(cat,   batch_first=True, padding_value=0)

        num   = pad_sequence(num,   batch_first=True, padding_value=0)

        label = pad_sequence(label, batch_first=True, padding_value=0)

        

        cat   =   cat.type(torch.long)

        num   =   num.type(torch.float)

        label = label.type(torch.float)

        

        return [cat, num, label]
categorical_sizes = [c+1 for c in categorical_sizes] 
class ASHRAE_LSTM(nn.Module):



    def __init__(self, 

                 hidden_dim, 

                 dropout_proba, 

                 categorical_sizes, 

                 LSTM_layers   = 1,

                 bidirectional = False,

                 Numeric_Feat  = None):

        super(ASHRAE_LSTM, self).__init__()



        self.hidden_dim    = hidden_dim

        self.LSTM_layers   = LSTM_layers

        self.bidirectional = bidirectional

        

        emb_dims           = [(c, min(50, (c+1)//2)) for c in categorical_sizes]        

        self.emb_layers    = nn.ModuleList([nn.Embedding(x, y, padding_idx=0) 

                                            for x, y in emb_dims])        

        total_embs_size    = sum([y for x, y in emb_dims])        

        total_nums_size    = len(Numeric_Feat) if Numeric_Feat else 0

        total_size         = total_embs_size + total_nums_size

                

        self.lstm    = nn.LSTM(input_size    = total_size, 

                               hidden_size   = hidden_dim,

                               batch_first   = True,

                               bidirectional = bidirectional,

                               num_layers    = LSTM_layers)  

    

        self.dropout = nn.Dropout(p=dropout_proba)

        

        self.fc1     = nn.Linear(hidden_dim, 64)

        self.fc2     = nn.Linear(64, 32)

        self.fc3     = nn.Linear(32, 1)

        

        

    def forward(self, cat_data, numeric_data, hidden_states, cell_states, seq_len, batch_size):



        cat_embs = [emb_layer(cat_data[:,:, i]) 

                    for i, emb_layer in enumerate(self.emb_layers)]        

        cat_embs = torch.cat(cat_embs, 2)        



        x = torch.cat([cat_embs, numeric_data], 2)        

        

        output, (hidden_states, cell_states) = self.lstm(x, (hidden_states, cell_states))

        

        output = self.dropout(output)

        output = F.relu(self.fc1(output))

        output = self.dropout(output)

        output = F.relu(self.fc2(output))

        output = self.dropout(output)

        output = self.fc3(output) 

        

        return output, hidden_states, cell_states



    def init_hidden(self, batch_size): # initialize the hidden state and the cell state to zeros

        if self.bidirectional:

            nb_directions = 2

        else:

            nb_directions = 1

            

        return (torch.zeros(nb_directions * self.LSTM_layers, batch_size, self.hidden_dim),

                torch.zeros(nb_directions * self.LSTM_layers, batch_size, self.hidden_dim))
def count_parameters(model):

    print('Number of parameters:')

    print('{:0,.0f}'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))



m = ASHRAE_LSTM(hidden_dim        =80,  

                dropout_proba     =0.45, 

                categorical_sizes =categorical_sizes, 

                Numeric_Feat      = numericals)

count_parameters(m)

m
del m

gc.collect()



def RMSLE_LOSS(y_true, y_pred):

    return np.sqrt(mean_squared_log_error(y_true, y_pred))



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

device
epochs        = 5

BATCH_SIZE    = 16

hidden_dim    = 100

dropout_proba = 0.3

Reduce_Data   = True

pct_reduce    = 0.30



TrainSet = ASHRAE_Dataset(data       =  train_counts.sort_values('counts'),

                          cat_path   = 'train/categorical/', 

                          num_path   = 'train/numerical/', 

                          label_path = 'train/Ys/',

                          Reduce_Data= Reduce_Data,

                          pct_reduce = pct_reduce,

                          is_val     = False)



ValSet = ASHRAE_Dataset(data        = train_counts.sort_values('counts'),

                        cat_path    = 'validation/categorical/', 

                        num_path    = 'validation/numerical/', 

                        label_path  = 'validation/Ys/',

                        is_val      = True)



collate = MyCollator()



train_loader = DataLoader(TrainSet, 

                          batch_size=BATCH_SIZE, 

                          shuffle=False,

                          collate_fn=collate)



val_loader = DataLoader(ValSet, 

                        batch_size=BATCH_SIZE, 

                        shuffle=False,

                        collate_fn=collate)





model = ASHRAE_LSTM(hidden_dim        = hidden_dim, 

                    dropout_proba     = dropout_proba, 

                    categorical_sizes = categorical_sizes, 

                    Numeric_Feat      = numericals)

model.to(device)



criterion = nn.MSELoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)



Best_Test_loss = 10**10.0

total_loss = 0



for epochs in tqdm(range(epochs)):

    model.train()

    

    counter = 0

    total_loss = 0

    total_RMSLE_loss = 0

    for cat, num, label in tqdm(train_loader, leave=False):



        BS = cat.shape[0]

        hidden_states, cell_states = model.init_hidden(BS)

        hidden_states, cell_states = hidden_states.to(device), cell_states.to(device)

        

        for i, Batch_IX in enumerate(range(cat.shape[1]),start=1):

            

            cat_batch   = cat[:,Batch_IX,:,:].clone()

            num_batch   = num[:,Batch_IX,:,:].clone()

            label_batch = label[:,Batch_IX,:].clone()

            

            cat_batch, num_batch, label_batch = cat_batch.to(device), num_batch.to(device), label_batch.to(device)

            

            predictions, hidden_states, cell_states = model.forward(cat_batch, 

                                                                    num_batch, 

                                                                    hidden_states, 

                                                                    cell_states,

                                                                    TBPTT,

                                                                    BS)

        

            optimizer.zero_grad()

            loss = criterion(predictions, label_batch)

            total_loss+=loss.item()

            counter+=1

            loss.backward(retain_graph=True)

            optimizer.step()

            RMSLE_loss = RMSLE_LOSS(torch.clamp(label_batch.reshape(-1), min=0).cpu().detach().numpy(), 

                                    torch.clamp(predictions.reshape(-1), min=0).cpu().detach().numpy())

            total_RMSLE_loss += RMSLE_loss

            print(f'train - batch loss: {loss} avg loss: {total_loss/counter} RMSLE_loss: {RMSLE_loss} avg RMSLE loss: {total_RMSLE_loss/counter}' ,end='\r')

            torch.cuda.empty_cache()

            

            

    print('validation time!')



    y_preds = []

    y_trues = []

    with torch.no_grad():

        for cat, num, label in tqdm(val_loader, leave=False):



            BS=cat.shape[0]

            hidden_states, cell_states = model.init_hidden(BS)

            hidden_states, cell_states = hidden_states.to(device), cell_states.to(device)



            for Batch_IX in tqdm(range(cat.shape[1]),leave=False):



                cat_batch   = cat[:,Batch_IX,:,:].clone()

                num_batch   = num[:,Batch_IX,:,:].clone()

                label_batch = label[:,Batch_IX,:].clone()



                cat_batch, num_batch = cat_batch.to(device), num_batch.to(device)



                predictions, hidden_states, cell_states = model.forward(cat_batch, 

                                                                        num_batch, 

                                                                        hidden_states, 

                                                                        cell_states, 

                                                                        TBPTT,

                                                                        BS)



                y_preds.append(torch.clamp(predictions.reshape(-1), min=0).cpu().detach().numpy())

                y_trues.append(torch.clamp(label_batch.reshape(-1), min=0).detach().numpy())

                torch.cuda.empty_cache()

                

        y_preds = np.concatenate(y_preds)

        y_trues = np.concatenate(y_trues)



        RMSLE_loss = RMSLE_LOSS(y_trues, y_preds)



        print('val RMSLE loss:', RMSLE_loss)



        is_best = RMSLE_loss < Best_Test_loss

        Best_Test_loss = min(RMSLE_loss, Best_Test_loss)



        if is_best:

            print('best val score so far!')

            torch.save(model.state_dict(), 'Best_Model.pt')
del train_counts

gc.collect()



test        = pd.read_csv("../input/ashrae-energy-prediction/test.csv")

building_df = pd.read_pickle("building_df.pkl")

test = test.merge(building_df, left_on = "building_id", right_on = "building_id", how = "left")



del building_df

gc.collect()
test, NAlist = reduce_mem_usage(test)
test.shape
weather_test = pd.read_csv("../input/ashrae-energy-prediction/weather_test.csv")

weather_test.sort_values(['site_id','timestamp'], inplace=True)

weather_test.fillna(method='ffill', inplace=True)

weather_test.fillna(method='bfill', inplace=True)

weather_test.isnull().sum().sum()
with open("weather_scalers.pkl", "rb") as ws:

    weather_scalers = pickle.load(ws)



for col in tqdm(time_cols):

    Scaler = weather_scalers[col]

    weather_test[col] = Scaler.transform(weather_test[col].values.reshape(-1, 1))
test = test.merge(weather_test, 

                  left_on = ["site_id", "timestamp"], 

                  right_on = ["site_id", "timestamp"], how = "left")

del weather_test

gc.collect()
test, NAlist = reduce_mem_usage(test)
test["timestamp"] = pd.to_datetime(test["timestamp"])

test["hour"]      = test["timestamp"].dt.hour.astype(np.int8)

test["day"]       = test["timestamp"].dt.day.astype(np.int8)

test["weekday"]   = test["timestamp"].dt.weekday.astype(np.int8)

test["month"]     = test["timestamp"].dt.month.astype(np.int8)
with open("Cat2Int.pkl", "rb") as s:

    Cat2Int = pickle.load(s)

    

for col in categoricals:

    if col!='building_id':

        test[col] = test[col].map(Cat2Int[col]).fillna(1).astype(np.int8) 

    else:

        test[col] = test[col].map(Cat2Int[col]).fillna(1).astype(np.int16)

        

test.head(3)
test.sort_values(['building_id','timestamp'], inplace=True)

unorder_row_id = test['row_id'].values



X_test_cat = test[categoricals]

X_test_num = test[numericals]



del test

gc.collect()



test_counts = X_test_cat.building_id.value_counts().to_frame('counts').sort_index()

test_counts = test_counts[test_counts.counts >0]

test_counts['cumsum_counts'] = test_counts.counts.cumsum()

test_counts.head()
print('average number of samples per building train:', test_counts.counts.mean())

print('maximum number of samples per building train:', test_counts.counts.max())
%%time

X_test_cat = X_test_cat.values

X_test_num = X_test_num.values



test_splits = test_counts.drop(test_counts.tail(1).index).cumsum_counts



X_test_cat  = np.split(X_test_cat, test_splits)

X_test_num  = np.split(X_test_num, test_splits)



del test_splits, test_counts

gc.collect()
BATCH_SIZE = 1



model = ASHRAE_LSTM(hidden_dim        = hidden_dim, 

                    dropout_proba     = dropout_proba, 

                    categorical_sizes = categorical_sizes, 

                    Numeric_Feat      = numericals)

model.load_state_dict(torch.load('Best_Model.pt'))

model.eval()

model.to(device)



y_preds = []

with torch.no_grad():

    for cat, num in tqdm(zip(X_test_cat, X_test_num), total=len(X_test_cat)):

        

        hidden_states, cell_states = model.init_hidden(BATCH_SIZE)

        hidden_states, cell_states = hidden_states.to(device), cell_states.to(device)

        

        cat_batch, num_batch = (torch.tensor(cat, dtype=torch.long).to(device), 

                                torch.tensor(num, dtype=torch.float).to(device))



        cat_batch, num_batch = cat_batch.unsqueeze(0), num_batch.unsqueeze(0)

        

        predictions, hidden_states, cell_states = model.forward(cat_batch, 

                                                                num_batch, 

                                                                hidden_states, 

                                                                cell_states, 

                                                                cat_batch.shape[1],

                                                                BATCH_SIZE)



        y_preds.append(torch.clamp(predictions.reshape(-1), min=0).cpu().detach().numpy())

        torch.cuda.empty_cache()

        

    y_preds = np.concatenate(y_preds)           
del X_test_cat, X_test_num

gc.collect()
with open("target_Scaler.pkl", "rb") as f:

    MMScaler = pickle.load(f)

    

result = MMScaler.inverse_transform(y_preds.reshape(-1, 1))

result.shape
result_df = pd.DataFrame({'row_id':unorder_row_id,

                          'meter_reading':result.reshape(-1)})

result_df.sort_values('row_id',inplace=True)

result_df.to_csv('submission.csv', index=False)

create_path(path_list=['train','validation'])

result_df.head()