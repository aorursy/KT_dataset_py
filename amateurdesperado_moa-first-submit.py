import numpy as np

import pandas as pd

from matplotlib import pyplot as plt

%matplotlib inline

import seaborn as sns

sns.set_style('ticks')

sns.set_context("poster")

sns.set_palette('colorblind')

import warnings

warnings.filterwarnings('ignore')

import torch

from torch.utils.data import Dataset, DataLoader, TensorDataset

from torch import nn

import torch.nn.functional as F

import torchvision

import time

from sklearn.model_selection import train_test_split

import random
plt.rcParams['figure.figsize'] = (20.0, 10.0)
train_features = pd.read_csv('/kaggle/input/lish-moa/train_features.csv')
train_features.info()
train_features.head()
train_features['cp_type'].value_counts()
train_features[train_features['cp_type'] == 'ctl_vehicle']
train_mask = train_features['cp_type'] != 'ctl_vehicle'

train_sig_ids = train_features.loc[train_mask]['sig_id']

train = train_features.loc[train_mask]
train['cp_time'].value_counts()
train['cp_dose'].value_counts()
train['sig_id'].nunique()
train.shape[0]
d1_times = train[train['cp_dose'] == 'D1']['cp_time'].value_counts()
d2_times = train[train['cp_dose'] == 'D2']['cp_time'].value_counts()
plt.bar(d1_times.index-5, d1_times.values,  width=5, label='d1', align='center');

plt.bar(d2_times.index, d2_times.values, width=5, label='d2', align='center');

plt.legend();

plt.xlabel('cp_time');

plt.ylabel('cp_dose_counts');
g_features = [cols for cols in train.columns if cols.startswith('g-')]

c_features = [cols for cols in train.columns if cols.startswith('c-')]
train[g_features].describe()
train[c_features].describe()
g_sample = random.sample(g_features, 3)

c_sample = random.sample(c_features, 3)
colors = ['navy', 'r', 'g']

for col, color in zip(g_sample, colors):

    plt.hist(train[col], bins=50, alpha=0.5, label=col)

    plt.axvline(np.median(train[col]), linewidth=3, color=color, label='median_{}'.format(col))

plt.xlim(-7, 7)

plt.legend();
colors = ['navy', 'r', 'g']

for col, color in zip(c_sample, colors):

    plt.hist(train[col], bins=50, alpha=0.5, label=col)

    plt.axvline(np.median(train[col]), linewidth=3, color=color, label='median_{}'.format(col))

plt.legend();
test_features = pd.read_csv('/kaggle/input/lish-moa/test_features.csv')
test_features.info()
test_features.head()
test_features['cp_type'].value_counts()
test_mask = test_features['cp_type'] != 'ctl_vehicle'

test_sig_ids = test_features.loc[test_mask]['sig_id']

test = test_features.loc[test_mask]
test['cp_time'].value_counts()
test['cp_dose'].value_counts()
test['sig_id'].nunique()
test.shape[0]
d1_times_test = test[test['cp_dose'] == 'D1']['cp_time'].value_counts()
d2_times_test = test[test['cp_dose'] == 'D2']['cp_time'].value_counts()
plt.bar(d1_times_test.index-5, d1_times.values,  width=5, label='d1', align='center');

plt.bar(d2_times_test.index, d2_times.values, width=5, label='d2', align='center');

plt.legend();

plt.xlabel('cp_time');

plt.ylabel('cp_dose_counts');
g_features_test = [cols for cols in test.columns if cols.startswith('g-')]

c_features_test = [cols for cols in test.columns if cols.startswith('c-')]
g_sample_test = random.sample(g_features_test, 3)

c_sample_test = random.sample(c_features_test, 3)
colors = ['navy', 'r', 'g']

for col, color in zip(g_sample_test, colors):

    plt.hist(test[col], bins=50, alpha=0.5, label=col)

    plt.axvline(np.median(test[col]), linewidth=3, color=color, label='median_{}'.format(col))

plt.xlim(-7, 7)

plt.legend();
colors = ['navy', 'r', 'g']

for col, color in zip(c_sample_test, colors):

    plt.hist(test[col], bins=50, alpha=0.5, label=col)

    plt.axvline(np.median(test[col]), linewidth=3, color=color, label='median_{}'.format(col))

plt.legend();
train_targets = pd.read_csv('/kaggle/input/lish-moa/train_targets_scored.csv')
train_targets.info()
train_targets.head()
train_targets.describe()
train_targets = train_targets[train_targets['sig_id'].isin(train_sig_ids)]
train_targets.drop(columns='sig_id').sum(axis=0).nlargest(20).plot(kind='barh');
train_targets.drop(columns='sig_id').sum(axis=0).nsmallest(20).plot(kind='barh');
def preprocess(data):

    data['cp_time'] = data['cp_time'].map({24:0, 48:1, 72:2})

    data['cp_dose'] = data['cp_dose'].map({'D1':0, 'D2':1})

    return data
preprocess(train.drop(columns = ['sig_id', 'cp_type'])).info()
preprocess(test.drop(columns = ['sig_id', 'cp_type'])).info()
X_train, X_val, y_train, y_val = train_test_split(preprocess(train.drop(columns = ['sig_id', 'cp_type'])), train_targets.drop(columns = ['sig_id']), test_size=0.2)
X_train.iloc[4, :]
class TabDataset:

    

    def __init__(self, X, y):

        self.X = X

        self.y = y

    

    def __len__(self):

        return(self.X.shape[0])

    

    def __getitem__(self, i):

        

        X_i = torch.from_numpy(self.X.iloc[i, :].values.astype(np.float32))

        y_i = torch.from_numpy(self.y.iloc[i, :].values.astype(np.float32))

        

        return X_i, y_i
class TabDatasetTest:

    

    def __init__(self, X):

        self.X = X

    

    def __len__(self):

        return(self.X.shape[0])

    

    def __getitem__(self, i):

        

        X_i = torch.from_numpy(self.X.iloc[i, :].values.astype(np.float32))        

        return X_i
train_ds = TabDataset(X_train, y_train)

valid_ds = TabDataset(X_val, y_val)
test_ds = TabDatasetTest(preprocess(test.drop(columns = ['sig_id', 'cp_type'])))
test_ds[0].shape
train_ds[0][1].dtype
train_dl = DataLoader(train_ds, batch_size=16)

valid_dl = DataLoader(valid_ds, batch_size=16)

test_dl = DataLoader(test_ds)
def lin_block(in_size, out_size):

    return nn.Sequential(

        nn.Linear(in_size, out_size),

        nn.BatchNorm1d(out_size)        

    )
class Model(nn.Module):

    def __init__(self, in_size, hidden_size, out_size, num_blocks, d_rates):

        super().__init__()

        

        self.num_blocks = num_blocks

        self.drop_rates = d_rates

        self.dense0 = lin_block(in_size, hidden_size)

        

        self.dense_blocks = nn.ModuleList()    

        for i in range(self.num_blocks):

            self.dense_blocks.append(lin_block(hidden_size, hidden_size))

           

        self.final = nn.Linear(hidden_size, out_size)

                

    def forward(self, x): 



        x = F.relu(self.dense0(x))



        if self.drop_rates is not None:

            x = F.dropout(x, self.drop_rates[0])



        for i, block in enumerate(self.dense_blocks):                

            x = F.relu(block(x))



            if self.drop_rates is not None:

                x = F.dropout(x, self.drop_rates[i+1])



        x = self.final(x)

        return x            
def fit(epochs, train_dl, valid_dl, model, loss_func, optimizer, device):

    

    losses_train, losses_val = [], []

    for epoch in range(epochs):

        t0 = time.time()

        running_loss = 0.0

        valid_loss = 0.0

        for data in train_dl:

            

            model.train()

            inputs, labels = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()



            outputs = model(inputs)

            loss = loss_func(outputs, labels)

            loss.backward()

            optimizer.step()



            running_loss += loss.item()



        model.eval();

        preds = []

        targs = []



        with torch.no_grad():

            for data in valid_dl:

                inputs, labels = data[0].to(device), data[1].to(device)

                outputs = model(inputs)

                preds.append(outputs.cpu().detach())

                targs.append(labels.cpu().detach())

                

                valid_loss += loss.item()



        train_loss = running_loss / (len(train_dl)-1)

        val_loss = valid_loss / (len(valid_dl)-1)

        print(f'[{epoch + 1}, {time.time() - t0:.1f}] loss: {loss}, val_loss: {val_loss:.3f}')

        losses_train.append(train_loss)

        losses_val.append(val_loss)

        running_loss = 0.0

        valid_loss = 0.0

                

    return losses_train, losses_val
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

device
model = Model(874, 1024, 206, 2, None)

model.to(device)
loss = nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(model.parameters(), 1e-3)
losses_train, losses_val = fit(25, train_dl, valid_dl, model, loss, optimizer, device)
plt.plot(range(len(losses_train)), losses_train, label='train');

plt.plot(range(len(losses_val)), losses_val, label='val');

plt.xlabel('epoch')

plt.ylabel('loss')

plt.legend();
def inference_fn(model, dataloader, device):

    model.eval()

    preds = []

    

    for data in dataloader:

        inputs = data.to(device)



        with torch.no_grad():

            outputs = model(inputs)

        

        preds.append(outputs.sigmoid().detach().cpu().numpy())

        

    preds = np.concatenate(preds)

    

    return preds
preds = inference_fn(model, test_dl, device)
results = pd.DataFrame(preds, columns=train_targets.columns[1:])



test_sig_ids.reset_index(drop=True, inplace=True)

results['sig_id'] = test_sig_ids
sample_subs = pd.read_csv('/kaggle/input/lish-moa/sample_submission.csv')
submission = sample_subs[['sig_id']].merge(results, on='sig_id', how='left').fillna(0)
submission.to_csv('submission.csv', index=False)