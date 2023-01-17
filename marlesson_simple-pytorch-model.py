import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import missingno as msno

import os

import torch

import gc

# determine the supported device

def get_device():

    if torch.cuda.is_available():

        device = torch.device('cuda:0')

    else:

        device = torch.device('cpu') # don't have GPU 

    return device



device = get_device()
!pip install torchbearer

!pip install livelossplot
import glob

import pandas as pd

from typing import List



def read_csv(path, re, dtype = None):

    all_files = glob.glob(os.path.join(path, re))     # advisable to use os.path.join as this makes concatenation OS independent



    df = (pd.read_csv(f, dtype=dtype) for f in all_files)

    df = pd.concat(df, ignore_index=True)

    return df



df = read_csv('/kaggle/input/kddbr-2020/', "2018*.csv")

df.head()
df.info()
df.shape
input_columns  = df.columns[df.columns.str.contains("input")]

output_columns = df.columns[df.columns.str.contains("output")]



print(input_columns[:10], output_columns[:10])
## Extract date information



def get_date_features(df, column):

    df['input_dt_sin_quarter']     = np.sin(2*np.pi*df[column].dt.quarter/4)

    df['input_dt_sin_day_of_week'] = np.sin(2*np.pi*df[column].dt.dayofweek/6)

    df['input_dt_sin_day_of_year'] = np.sin(2*np.pi*df[column].dt.dayofyear/365)

    df['input_dt_sin_day']         = np.sin(2*np.pi*df[column].dt.day/30)

    df['input_dt_sin_month']       = np.sin(2*np.pi*df[column].dt.month/12)

    

    

df['date']  = pd.to_datetime(df['date'])

get_date_features(df, 'date')

df.head()
input_columns  = df.columns[df.columns.str.contains("input")]

input_columns
# Filter onlu nissing values

null_columns=df.columns[df.isnull().any()]

msno.bar(df.sample(1000)[null_columns])
null_columns
df = df.fillna(0)
df[input_columns].shape
gc.collect()
import torch

from torch.utils.data import DataLoader

from torch.utils.data.dataset import Dataset



class CustomDataset(Dataset):

    def __init__(self, data, targets):

        self.data  = data

        self.targets = targets



    def __len__(self):

        return len(self.data)

    

    def __getitem__(self, idx):

        if torch.is_tensor(idx):

            idx = idx.tolist()

            

        input  = self.data[idx]

        target = self.targets[idx]

        

        return torch.Tensor(input.astype(float)).to(device), torch.Tensor(target.astype(float)).to(device)

    

input, output = CustomDataset(df[input_columns].values, df[output_columns].values).__getitem__(1)

[input, output]
import torch

import torch.nn as nn

import torch.nn.functional as F



class EncoderDecoder(nn.Module):

    def __init__(self, input_size, n_factors, output_size):

        super(EncoderDecoder,self).__init__()

        

        self.encoder = nn.Sequential(

            nn.Linear(input_size, int(input_size/2)),

            nn.ReLU(True),

            nn.Linear(int(input_size/2), int(input_size/4)),

            nn.ReLU(True),

            nn.Linear(int(input_size/4), n_factors),

            nn.ReLU(True))

        

        self.decoder = nn.Sequential(

            nn.Linear(n_factors, int(output_size/2)),

            nn.ReLU(True),

            nn.Linear(int(output_size/2), output_size),

            nn.ReLU(True),            

            nn.Linear(output_size, output_size))

        

        self.dropout = torch.nn.Dropout(0.2)

        

    def forward(self,x):

        x = self.encoder(x)

        x = self.dropout(x)

        x = self.decoder(x)

        return x
# Params

input_size  = len(input_columns) 

output_size = len(output_columns)

n_factors   = 50





# Model

model       = EncoderDecoder(input_size, n_factors, output_size).to(device)

model
#device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# Train Params

batch_size  = 128

num_epochs  = 30





# Loss and optimizer



# create a nn class (just-for-fun choice :-) 

class RMSELoss(nn.Module):

    def __init__(self):

        super().__init__()

        self.mse = nn.MSELoss()

        

    def forward(self,yhat,y):

        return torch.sqrt(self.mse(yhat,y))



criterion   = RMSELoss()

optimizer   = torch.optim.Adam(model.parameters(),weight_decay=1e-5)



model
from sklearn.model_selection import train_test_split

# Data Loader

class NoAutoCollationDataLoader(DataLoader):

    @property

    def _auto_collation(self):

        return False



    @property

    def _index_sampler(self):

        return self.batch_sampler

    

# Split dataset

#input_mean[np.isnan(input_mean)] = 0

X = df[input_columns].values

Y = df[output_columns].fillna(0).values



x_train, x_val, y_train, y_val = train_test_split(X, Y, test_size=0.20, random_state=42)

len(x_train), len(x_val)
# Pytorch Data Loader

train_loader = NoAutoCollationDataLoader(CustomDataset(x_train, y_train), batch_size=batch_size, shuffle=True)

val_loader   = NoAutoCollationDataLoader(CustomDataset(x_val, y_val), batch_size=batch_size, shuffle=False)
import torchbearer

from torchbearer import Trial

from torchbearer.callbacks import LiveLossPlot

from torchbearer.callbacks.checkpointers import ModelCheckpoint

from torchbearer.callbacks.early_stopping import EarlyStopping



weigth_path = "/kaggle/working/weights.pt"

callbacks = [

                LiveLossPlot(), 

                EarlyStopping(min_delta=1e-3, patience=10, monitor='val_loss', mode='min'),

                ModelCheckpoint(weigth_path, save_best_only=True, monitor='val_loss', mode='min')

            ]
%matplotlib inline



trial = Trial(model, optimizer, criterion, metrics=['loss'], callbacks=callbacks).to(device)

trial.with_generators(train_generator=train_loader, val_generator=val_loader)

hist = trial.run(epochs=num_epochs)
!ls /kaggle/working/
## Load Model



state_dict = torch.load(weigth_path, map_location=device)

model.load_state_dict(state_dict["model"])

#model.eval()
# Predict

val_predictions = model(torch.Tensor(x_val).to(device)).detach().cpu().numpy()
# Load trained model



#val_loader   = NoAutoCollationDataLoader(CustomDataset(x_val, y_val), batch_size=batch_size, shuffle=False)

trial           = Trial(model, optimizer, criterion).with_test_generator(val_loader).to(device)

val_predictions = trial.predict().detach().cpu().numpy()#.reshape(-1)



val_predictions.shape, y_val.shape
#  Weighted Root Mean Squared Error (WRMSE),

# 0	1.00

# 1	0.75

# 2	0.60

# 3	0.50

# 4	0.43

# 5	0.38

# 6	0.33



weigths = [1]*16 + [0.75]*16 + [0.6]*16 + [0.5]*16 + [0.43]*16 + [0.38]*16 + [0.33]*16



def wrmse(predictions, targets, weigths):

    #Is it???

    return np.sqrt((((predictions - targets) ** 2)*weigths).mean())



def rmse(predictions, targets):

    return np.sqrt(((predictions - targets) ** 2).mean())



error_wrmse = wrmse(val_predictions, y_val, weigths)

error_rmse  = rmse(val_predictions, y_val)



print("WRMSE:", error_wrmse)

print("RMSE:", error_rmse)
# Clean Memory

del x_train

del x_val

del df



del train_loader

del val_loader



gc.collect()
## Load Model



state_dict = torch.load(weigth_path, map_location=device)

model.load_state_dict(state_dict["model"])

model.eval()
# Load and Prepare data

df_pred = read_csv('/kaggle/input/kddbr-2020/', "public2019*.csv").fillna(0)



# transform

df_pred['date']  = pd.to_datetime(df_pred['date'])

get_date_features(df_pred, 'date')



df_pred.head()
# Predict

inputs = torch.Tensor(df_pred[input_columns].values).to(device)

pred   = model(inputs).cpu().detach().numpy()

pred
df_pred_sub = pd.DataFrame(pred)

df_pred_sub.columns = output_columns

df_pred_sub['id']   = df_pred['id']

df_pred_sub.head()
###
## submission

df_sub = []

for i, row in df_pred_sub.iterrows():

    for column, value in zip(output_columns, row.values):

        id = "{}_{}".format(int(row.id), column)

        df_sub.append([id, value])



df_sub = pd.DataFrame(df_sub)

df_sub.columns = ['id', 'value']

df_sub.to_csv('/kaggle/working/submission.csv', index=False)

df_sub