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
!pip install pytorch-lightning
test_features = pd.read_csv('/kaggle/input/lish-moa/test_features.csv')

sample_submission= pd.read_csv('/kaggle/input/lish-moa/sample_submission.csv')

train_features = pd.read_csv('/kaggle/input/lish-moa/train_features.csv')

train_targets_scored = pd.read_csv('/kaggle/input/lish-moa/train_targets_scored.csv')

train_targets_nonscored = pd.read_csv('/kaggle/input/lish-moa/train_targets_nonscored.csv')
train_targets_scored.head()
train_targets_scored.shape
train_features.head()
train_features.shape
train_features['cp_type'].value_counts()
import matplotlib.pyplot as plt

%matplotlib inline

train_features['cp_time'].value_counts()
cs = train_features[:1][[col for col in train_features.columns if 'c-' in col]].values.reshape(-1, 1)
plt.plot(cs)
plt.plot(sorted(cs))
train_features = pd.concat([train_features, pd.get_dummies(train_features['cp_time'], prefix='cp_time')], axis=1)

train_features = pd.concat([train_features, pd.get_dummies(train_features['cp_dose'], prefix='cp_dose')], axis=1)

train_features = pd.concat([train_features, pd.get_dummies(train_features['cp_type'], prefix='cp_type')], axis=1)

train_features = train_features.drop(['cp_type', 'cp_time', 'cp_dose'], axis=1)
train_features.shape
import torch

import pandas as pd

import torch.nn as nn



class MoADataset :

    def __init__(self, dataset, targets) :

        self.dataset = dataset

        self.targets = targets

    

    def __len__(self) :

        return self.dataset.shape[0]

    

    def __getitem__(self, item) :

        return {

            "x" : torch.tensor(self.dataset[item,:], dtype = torch.float),

            "y" : torch.tensor(self.targets[item,:], dtype = torch.float)

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

            nn.Dropout(0.3),

            nn.PReLU(),

            nn.Linear(1024, num_targets),

        )



    def forward(self, x):

        x = self.model(x)

        return x

        
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

                                                                                test_size=0.1, random_state=42)

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

        valid_loader = torch.utils.data.DataLoader(  self.valid_dataset,

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

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

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
trainer = pl.Trainer(gpus=1,

                    max_epochs=5,

                    weights_summary='full')
train_features.shape
train_targets_scored.shape
net = Model(879, 206)

model = LitMoA(hparams = {}, model=net)

dm = MoADataModule(hparams = {}, data = train_features, targets = train_targets_scored)

trainer.fit(model, dm)
test_features = pd.concat([test_features, pd.get_dummies(test_features['cp_time'], prefix='cp_time')], axis=1)

test_features = pd.concat([test_features, pd.get_dummies(test_features['cp_dose'], prefix='cp_dose')], axis=1)

test_features = pd.concat([test_features, pd.get_dummies(test_features['cp_type'], prefix='cp_type')], axis=1)

test_features = test_features.drop(['cp_type', 'cp_time', 'cp_dose'], axis=1)
class TestMoADataset :

    def __init__(self, dataset) :

        self.dataset = dataset

        

    def __len__(self) :

        return self.dataset.shape[0]

    

    def __getitem__(self, item) :

        return {

            'x' : torch.tensor(self.dataset[item, :], dtype=torch.float)

        }
test_dataset = TestMoADataset(dataset=test_features.iloc[:, 1:].values)
test_loader = torch.utils.data.DataLoader(test_dataset,

            batch_size=1024,

            num_workers=0,

            shuffle=False,

        )
predictions = np.zeros((test_features.shape[0], 206))

inference_model = model.model

inference_model.eval()

for ind, batch in enumerate(test_loader) :

    p = torch.sigmoid(inference_model(batch['x'])).detach().cpu().numpy()

    predictions[ind*1024 : (ind+1)*1024] = p
predictions.shape
test_features1 = pd.read_csv('/kaggle/input/lish-moa/test_features.csv')

s = pd.DataFrame({'sig_id': test_features1['sig_id'].values})
s
for col in train_targets_scored.columns[1:].tolist():

    s[col] = 0
s.shape
s.loc[:, train_targets_scored.columns[1:]] = predictions
s.head()
test_features1.loc[test_features1['cp_type'] =='ctl_vehicle', 'sig_id']
s.loc[s['sig_id'].isin(test_features1.loc[test_features1['cp_type'] =='ctl_vehicle', 'sig_id']), train_targets_scored.columns[1:]] = 0
s.to_csv('submission.csv', index=False)



torch.save(model.model.state_dict(), 'model_shashank_copied.pt')
