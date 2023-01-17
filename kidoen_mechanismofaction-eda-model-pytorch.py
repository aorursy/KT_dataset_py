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

        

import plotly.express as px

from plotly.subplots import make_subplots

import matplotlib.pyplot as plt

import seaborn as sns



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
dftrain = pd.read_csv("/kaggle/input/lish-moa/train_features.csv")

dftest = pd.read_csv("/kaggle/input/lish-moa/test_features.csv")

df = pd.concat([dftrain,dftest])

df.head()
df.shape
sns.countplot(dftrain['cp_type']) 

# TREATMENT STATUS - trt_cp -> Treated & ctl_vehicle -> control
sns.countplot(dftrain['cp_dose'])
sns.countplot(dftrain['cp_time']) # TREATMENT TIME
df.isnull().sum()
df.info()
df.columns
train_target = pd.read_csv("../input/lish-moa/train_targets_scored.csv")

train_target.head()
x = train_target.drop(['sig_id'],axis=1).sum(axis=0).sort_values(ascending=False).reset_index()

x.columns = ['Protein/Enzyme','non-zero-records']

x.head()
top = x.head(20)

plt.figure(figsize=(20,8))

plt.title("Top 20 Protein/Enzyme Entries")

sns.barplot(top['Protein/Enzyme'],top['non-zero-records'])

plt.xticks(rotation=45)

plt.show()
bot = x.tail(20)

plt.figure(figsize=(20,8))

plt.title("Bottom 20 Protein/Enzyme Entries")

sns.barplot(bot['Protein/Enzyme'],bot['non-zero-records'])

plt.xticks(rotation=45)

plt.show()
x = train_target.drop(['sig_id'],axis=1).sum(axis=0).sort_values(ascending=False).reset_index()

x.columns = ['Protein/Enzyme','non-zero-records']

x['count'] = x['non-zero-records'] * 100 / len(train_target)

x.head()
top = x.head(40)

plt.figure(figsize=(20,8))

plt.title("Top 40 Protein/Enzyme Entries by Overall percentage")

sns.barplot(top['Protein/Enzyme'],top['count'])

plt.xticks(rotation=45)

plt.show()
bot = x.tail(20)

plt.figure(figsize=(20,8))

plt.title("Bottom 20 Protein/Enzyme Entries by Overall percentage")

sns.barplot(bot['Protein/Enzyme'],bot['count'])

plt.xticks(rotation=45)

plt.show()
x = train_target.drop(['sig_id'],axis=1).astype(bool).sum(axis=1).reset_index()

x.columns = ['row','count']

x = x.groupby(['count'])['row'].count().reset_index()

x.head()
sns.barplot(x['count'],x['row'])
px.pie(x,values=100 * x['row']/len(train_target),names='count',

      title='Number of activations in targets for every sample (Percent)')
train_target.describe()
train_columns = dftrain.columns.to_list()

g_list = [i for i in train_columns if i.startswith('g-')]

c_list = [i for i in train_columns if i.startswith('c-')]
columns = g_list + c_list

import random

forcorr = [columns[random.randint(0,len(columns)-1)] for i in range(30)]
forcorr
corrdata = df[forcorr]
plt.figure(figsize=(24,14))

plt.title("Correlation Matrix for Randomly selected 30 Features")

sns.heatmap(corrdata.corr(),annot=True)

plt.show()
import time



start = time.time()

cols = ['cp_time'] + columns

all_columns = []

for i in range(0, len(cols)):

    for j in range(i+1, len(cols)):

        if abs(dftrain[cols[i]].corr(dftrain[cols[j]])) > 0.9:

            all_columns.append(cols[i])

            all_columns.append(cols[j])



print(time.time()-start)

            
highcorrdata = df[all_columns]
plt.figure(figsize=(24,14))

plt.title("Correlation Matrix for Highly Correlated features")

sns.heatmap(highcorrdata.corr())

plt.show()
target_columns = train_target.columns.tolist()

target_columns.remove('sig_id')
correlation_matrix = pd.DataFrame()

for t_col in train_target.columns:

    corr_list = list()

    if t_col == 'sig_id':

        continue

    for col in columns:

        res = dftrain[col].corr(train_target[t_col])

        corr_list.append(res)

    correlation_matrix[t_col] = corr_list
correlation_matrix['train_features'] = columns

correlation_matrix = correlation_matrix.set_index('train_features')

correlation_matrix
targetcols = train_target.columns.tolist()

# target_columns.remove('sig_id')

foranalysis = [target_columns[random.randint(0,len(target_columns)-1)] for m in range(5)]
foranalysis
currentcols = correlation_matrix[foranalysis]
currentcols
coldf = pd.DataFrame()

tr_first_cols = list()

tr_second_cols = list()

tarcols = list()

for col in currentcols.columns:

    tarcols.append(col)

    tr_first_cols.append(currentcols[col].abs().sort_values(ascending=False).reset_index()['train_features'].head(2).values[0])

    tr_second_cols.append(currentcols[col].abs().sort_values(ascending=False).reset_index()['train_features'].head(2).values[1])
coldf['column'] = tarcols

coldf['train_1_column'] = tr_first_cols

coldf['train_2_column'] = tr_second_cols

coldf
def scatterplot(coldf,index):

    analysis = pd.DataFrame()

    analysis['color'] = train_target[coldf.iloc[index]['column']]

    analysis['x'] = dftrain[coldf.iloc[index]['train_1_column']]

    analysis['y'] = dftrain[coldf.iloc[index]['train_2_column']]

    analysis.columns = ['color',coldf.iloc[index]['train_1_column'],coldf.iloc[index]['train_2_column']]

    analysis['size'] = 1

    analysis.loc[analysis['color'] == 1, 'size'] = 10

    fig = px.scatter(

        analysis, 

        x=coldf.iloc[index]['train_1_column'], 

        y=coldf.iloc[index]['train_2_column'], 

        color="color", 

        size='size', 

        height=800,

        title='Scatter plot for ' + coldf.iloc[index]['column']

    )

    fig.show()
scatterplot(coldf, 0)
scatterplot(coldf, 1)
scatterplot(coldf, 2)
scatterplot(coldf,3)
scatterplot(coldf, 4)
train_targets_scored = pd.read_csv("/kaggle/input/lish-moa/train_targets_scored.csv")

train_targets_nonscored = pd.read_csv("/kaggle/input/lish-moa/train_targets_nonscored.csv")

train_features = pd.read_csv("/kaggle/input/lish-moa/train_features.csv")

test_features = pd.read_csv("/kaggle/input/lish-moa/test_features.csv")

sample_submission = pd.read_csv("/kaggle/input/lish-moa/sample_submission.csv")
import torch

import torch.nn as nn
sample_submission.head()
train_targets_scored.head()
train_features.head()
test_features.head()
train_features[:1][[col for col in train_features.columns if 'g-' in col]].values
gs = train_features[:1][[col for col in train_features.columns if 'g-' in col]].values.reshape(-1,1)

gs
plt.plot(gs)
sns.distplot(train_features['g-0'],color='red')
# train_features.loc[:,"kfold"] = -1
train_features.head(2)
train_features = pd.concat([train_features,pd.get_dummies(train_features['cp_time'],prefix = 'cp_time')],axis=1)

train_features = pd.concat([train_features,pd.get_dummies(train_features['cp_type'],prefix = 'cp_type')],axis=1)

train_features = pd.concat([train_features,pd.get_dummies(train_features['cp_dose'],prefix = 'cp_dose')],axis=1)

train_features = train_features.drop(columns=['cp_time','cp_type','cp_dose'],axis=1)

train_features.head(3)
       

class MoADataset:

    def __init__(self, dataset, targets):

        self.dataset = dataset

        self.targets = targets



    def __len__(self):

        return self.dataset.shape[0]



    def __getitem__(self, item):

        return {

            "x": torch.tensor(self.dataset[item, :], dtype=torch.float),

            "y": torch.tensor(self.targets[item, :], dtype=torch.float),

        }

        

        
    

class Model(nn.Module):

    def __init__(self, num_features, num_targets):

        super().__init__()

        self.model = nn.Sequential(

            nn.Linear(num_features, 1024),

            nn.BatchNorm1d(1024),

            nn.Dropout(0.3),

            nn.PReLU(),

            nn.Linear(1024, 1024),

            nn.BatchNorm1d(1024),

            nn.Dropout(0.4),

            nn.PReLU(),

            nn.Linear(1024, 1024),

            nn.BatchNorm1d(1024),

            nn.Dropout(0.4),

            nn.PReLU(),

            nn.Linear(1024, num_targets),

        )



    def forward(self, x):

        x = self.model(x)

        return x
!pip install pytorch-lightning
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
    

class MoADataModule(pl.LightningDataModule):

    def __init__(self, hparams, data, targets):

        super().__init__()

        self.hparams = hparams

        self.data = data

        self.targets = targets



    def prepare_data(self):

        pass



    def setup(self, stage=None):



        train_data, valid_data, train_targets, valid_targets = train_test_split(self.data, self.targets,

                                                                                test_size=0.2, random_state=42)

        self.train_dataset = MoADataset(dataset=train_data.iloc[:, 1:].values,

                                         targets=train_targets.iloc[:, 1:].values)

        self.valid_dataset = MoADataset(dataset=valid_data.iloc[:, 1:].values,

                                         targets=valid_targets.iloc[:, 1:].values)

    

    def train_dataloader(self):

        train_loader = torch.utils.data.DataLoader(

            self.train_dataset,

            batch_size=1024,

            num_workers=0,

            shuffle=True,

        )

        return train_loader



    def val_dataloader(self):

        valid_loader = torch.utils.data.DataLoader(

            self.valid_dataset,

            batch_size=1024,

            num_workers=0,

            shuffle=False,

        )



        return valid_loader



    def test_dataloader(self):

        return None

    
class LitMoA(pl.LightningModule):

    def __init__(self, hparams, model):

        super(LitMoA, self).__init__()

        self.hparams = hparams

        self.model = model

        self.criterion = nn.BCEWithLogitsLoss()

                

    def forward(self, x):

        return self.model(x)

        

    def configure_optimizers(self):

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,

                                                               patience=3, threshold=0.00001, mode="min", verbose=True)

        return ([optimizer],

                [{'scheduler': scheduler, 'interval': 'epoch', 'monitor': 'valid_loss'}])

        



    def training_step(self, batch, batch_idx):

        data = batch['x']

        target = batch['y']

        out = self(data)

        loss = self.criterion(out, target)

        logs = {'train_loss': loss}        

        return {'loss': loss, 'log': logs, 'progress_bar': logs}

    

    def training_epoch_end(self, outputs):

        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()

        logs = {'train_loss': avg_loss}

        return {'log': logs, 'progress_bar': logs}

    

    def validation_step(self, batch, batch_idx):

        data = batch['x']

        target = batch['y']

        out = self(data)

        loss = self.criterion(out, target)

        

        logs = {'valid_loss': loss}

        

        return {'loss': loss, 'log': logs, 'progress_bar': logs}

    

    def validation_epoch_end(self, outputs):

        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()

        logs = {'valid_loss': avg_loss}

        return {'log': logs, 'progress_bar': logs}

    

        

        
trainer = pl.Trainer(gpus=1,max_epochs=50,weights_summary='full')

train_features.head()
train_targets_scored.head()
net = Model(879, 206) # number of features, number of targets

model = LitMoA(hparams={}, model=net)

dm = MoADataModule(hparams={}, data=train_features, targets=train_targets_scored)
trainer.fit(model,dm)
test_features = pd.concat([test_features, pd.get_dummies(test_features['cp_time'], prefix='cp_time')], axis=1)

test_features = pd.concat([test_features, pd.get_dummies(test_features['cp_dose'], prefix='cp_dose')], axis=1)

test_features = pd.concat([test_features, pd.get_dummies(test_features['cp_type'], prefix='cp_type')], axis=1)

test_features = test_features.drop(['cp_type', 'cp_time', 'cp_dose'], axis=1)
class TestMoADataset:

    def __init__(self, dataset):

        self.dataset = dataset



    def __len__(self):

        return self.dataset.shape[0]



    def __getitem__(self, item):

        return {

            "x": torch.tensor(self.dataset[item, :], dtype=torch.float),

        }
test_dataset = TestMoADataset(dataset=test_features.iloc[:, 1:].values)
test_loader = torch.utils.data.DataLoader(

            test_dataset,

            batch_size=1024,

            num_workers=0,

            shuffle=False,

        )
predictions = np.zeros((test_features.shape[0], 206))

inference_model = model.model

inference_model.eval()

for ind, batch in enumerate(test_loader):

    p = inference_model(batch['x'])[0].detach().cpu().numpy()

    predictions[ind * 1024:(ind + 1) * 1024] = p
test_features1 = pd.read_csv('/kaggle/input/lish-moa/test_features.csv')

s = pd.DataFrame({'sig_id': test_features1['sig_id'].values})
s
for col in train_targets_scored.columns[1:].tolist():

    s[col] = 0
s.loc[:, train_targets_scored.columns[1:]] = predictions
s.head()
test_features1.loc[test_features1['cp_type'] =='ctl_vehicle', 'sig_id']
s.loc[s['sig_id'].isin(test_features1.loc[test_features1['cp_type'] =='ctl_vehicle', 'sig_id']), train_targets_scored.columns[1:]] = 0
s.to_csv('submission.csv', index=False)
torch.save(model.model.state_dict(), 'model.pt')