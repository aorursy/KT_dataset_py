LR = 0.001462129310551811

# LR = 0.008

N_LAYERS = 5

F_DROPOUT = 0.407

LAYERS = [987,1206,2390,2498,3449]
LABEL_SMOOTHING = 0.008
# Install required modules only once

import sys

import subprocess

import pkg_resources



required = {'iterative-stratification', 'pytorch-lightning'}

installed = {pkg.key for pkg in pkg_resources.working_set}

missing = required - installed



if missing:

    python = sys.executable

    subprocess.check_call([python, '-m', 'pip', 'install', *missing], stdout=subprocess.DEVNULL)
import pandas as pd

import matplotlib.pyplot as plt

from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

import torch

import torch.nn as nn

import pytorch_lightning as pl

from pytorch_lightning.callbacks import ModelCheckpoint

from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split

import os

from ranger_py import Ranger

from mish_activation import *  
test_features = pd.read_csv('../input/lish-moa/test_features.csv')

train_features = pd.read_csv('../input/lish-moa/train_features.csv')

train_targets_scored = pd.read_csv('../input/lish-moa/train_targets_scored.csv')

train_targets_nonscored = pd.read_csv('../input/lish-moa/train_targets_nonscored.csv')

sample_submission = pd.read_csv('../input/lish-moa/sample_submission.csv')
# Though number of rows is not huge, number of features is 875, 

# which is very huge here, unlike in common cases.

train_features.shape, test_features.shape, train_targets_scored.shape
train_features.head()
train_features.cp_type.value_counts(), train_features.cp_time.value_counts(), train_features.cp_dose.value_counts()
train_targets_scored.head()
# Column wise sums across all rows

train_targets_scored.sum()[1:].sort_values().head(100)
gs = train_features[7:8][[col for col in train_features.columns if 'g-' in col]].values.reshape(-1, 1)

plt.plot(gs)
plt.plot(sorted(gs))
# Let us look at spread of each columns

figure = plt.figure(figsize=(15,8))

ax1 = figure.add_subplot(4,2,1)

ax2 = figure.add_subplot(4,2,2)

ax3 = figure.add_subplot(4,2,3)

ax4 = figure.add_subplot(4,2,4)

ax5 = figure.add_subplot(4,2,5)

ax6 = figure.add_subplot(4,2,6)

ax7 = figure.add_subplot(4,2,7)

ax8 = figure.add_subplot(4,2,8)



ax1.hist(train_features['c-1'])

ax2.hist(train_features['g-1'])

ax3.hist(train_features['c-2'])

ax4.hist(train_features['g-2'])

ax5.hist(train_features['c-25'])

ax6.hist(train_features['g-25'])

ax7.hist(train_features['c-49'])

ax8.hist(train_features['g-49'])
class MoADataset(Dataset):

    

    def __init__(self, features, targets):

        self.features = features

        self.targets = targets

        

    def __len__(self):

        return self.features.shape[0]

        

    def __getitem__(self, index):

        return {

            "x": torch.tensor(self.features[index, :], dtype=torch.float),

            "y": torch.tensor(self.targets[index, :], dtype=torch.float)

        }

    
class MoADataModule(pl.LightningDataModule):

    def __init__(self, batch_size=1024, fold= 0):

        super().__init__()

        self.batch_size = batch_size

        self.fold = fold

        

    def prepare_data(self):

        # Even in multi-GPU training. this method is called only from a single GPU. 

        # So this method ideal for download, stratification etc. 

        # Startification on multi-label dataset is tricky. 

        # scikit-learn stratified KFold cannot be used. 

        # So we are using interative-stratification

        if os.path.isfile("train_folds.csv"):

            return

        complete_training_data = self._read_data()        

        self._startify_and_save(complete_training_data)        

        

    def _read_data(self):

        features = pd.read_csv('../input/lish-moa/train_features.csv')

        # Convert categorical features into OHE

        features = pd.concat([features, pd.get_dummies(features['cp_time'], prefix='cp_time')], axis=1)

        features = pd.concat([features, pd.get_dummies(features['cp_dose'], prefix='cp_dose')], axis=1)

        features = pd.concat([features, pd.get_dummies(features['cp_type'], prefix='cp_type')], axis=1)

        # Delete original categorical features

        features = features.drop(['cp_time', 'cp_dose', 'cp_type'], axis=1)

        targets_scored = pd.read_csv('../input/lish-moa/train_targets_scored.csv')

        merged = features.merge(targets_scored, how="inner", on="sig_id")

        return merged

        

    def _startify_and_save(self, data):

        # New column to hold the fold number

        data.loc[:, "kfold"] = -1



        # Shuffle the dataframe

        data = data.sample(frac=1).reset_index(drop=True)        

        

        # 5 Folds

        mskf = MultilabelStratifiedKFold(n_splits=5, shuffle=False, random_state=None) 

        # trn_ and val_ are indices

        targets = data.drop(['kfold', 'sig_id'], axis=1)                                                                                       

        for fold_, (trn_,val_) in enumerate(mskf.split(X=data, y=targets.iloc[:, 879:])): 

            # We are just filling the vaidation indices. 

            # All other data are for training (trn indices are not required)

            data.loc[val_, "kfold"] = fold_

    

        # We are saving the result to the disk so that other GPUs can pick it from there. 

        # Rather if we do "self.startified_data = train_targets_scored", 

        # other GPUs will not be able to read this 

        data.to_csv("train_folds.csv", index=False)   

        

    def setup(self, stage=None):

        # In multi-GPU training, this method is run on each GPU. 

        # So ideal for each training/valid split

        data = pd.read_csv("train_folds.csv")

        

        training_data = data[data.kfold != self.fold]

        training_data = training_data.drop(['kfold', 'sig_id'], axis=1)

        validation_data = data[data.kfold == self.fold]

        validation_data = validation_data.drop(['kfold', 'sig_id'], axis=1)

        self.train_dataset = MoADataset(training_data.iloc[:, :879].values, training_data.iloc[:, 879:].values)

        self.valid_dataset = MoADataset(validation_data.iloc[:, :879].values, validation_data.iloc[:, 879:].values)        



    

    def train_dataloader(self):

        return DataLoader(self.train_dataset, self.batch_size, num_workers=4, shuffle=True)

    

    def val_dataloader(self):

        return DataLoader(self.valid_dataset, self.batch_size, num_workers=4, shuffle=False)    

            

        
class Model(nn.Module):

    def __init__(self, num_features, num_targets):

        super().__init__()

        layers = []

        

        # Intermediate layers

        in_size = num_features   

        for i in range(N_LAYERS):

            out_size = LAYERS[i]

            layers.append(torch.nn.Linear(in_size, out_size, bias=False))

            layers.append(nn.BatchNorm1d(out_size))

            layers.append(nn.Dropout(F_DROPOUT))

            layers.append(nn.PReLU())

#             layers.append(nn.BatchNorm1d(in_size))

#             layers.append(nn.Dropout(F_DROPOUT))    

#             layers.append(torch.nn.Linear(in_size, out_size))

#             layers.append(nn.PReLU())

#             layers.append(Mish())

            in_size = out_size



        # Final layer

        layers.append(torch.nn.Linear(in_size, num_targets))    

        self.model = torch.nn.Sequential(*layers)        

        

        # Initialize weights

        self.model.apply(self._init_weights)

        

    def _init_weights(self, m):

        if type(m) == nn.Linear:

            nn.init.xavier_uniform_(m.weight)

            if m.bias != None:

                m.bias.data.fill_(0.01)

        

    def forward(self, x):

        x = self.model(x)

        return x
class PLitMoAModule(pl.LightningModule):

    def __init__(self, hparams, model):

        super(PLitMoAModule, self).__init__()

        self.hparams = hparams

        self.model = model

        self.criterion = nn.BCEWithLogitsLoss()

#         self.criterion = nn.BCELoss()

        

    def forward(self, x):

        return self.model(x)

    

    def configure_optimizers(self):

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hparams["lr"] )

#         optimizer = Ranger(self.model.parameters(), lr=self.hparams["lr"] )

        scheduler = {"scheduler": 

                     torch.optim.lr_scheduler.ReduceLROnPlateau(

                        optimizer, patience=2, 

                        threshold=0.00003, 

                        mode='min', verbose=True),

                    "interval": "epoch",

                    "monitor": "val_loss"}

        return [optimizer], [scheduler]

    

    def training_step(self, batch, batch_index):

        features = batch['x']

        targets = batch['y']

        out = self(features)

        targets_smooth = targets.float() * (1 - LABEL_SMOOTHING) + 0.5 * LABEL_SMOOTHING

        loss = self.criterion(out, targets_smooth)

        logs = {"train_loss" : loss}

        return {"loss": loss, "log": logs, "progress_bar": logs}

    

    def training_epoch_end(self, outputs):

        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()

        logs = {"train_loss": avg_loss}

        return {"log": logs, "progress_bar": logs}

            

    def validation_step(self, batch, batch_index):

        features = batch['x']

        targets = batch['y']

        out = self(features)

        loss = self.criterion(out, targets)

        logs = {"val_loss" : loss}

        return {"loss": loss, "log": logs, "progress_bar": logs}

    

    def validation_epoch_end(self, outputs):

        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()

        logs = {"val_loss": avg_loss}

        return {"log": logs, "progress_bar": logs}
# Five fold training. 

for k in range(5):  

    

    checkpoint_callback = ModelCheckpoint(

        filepath='./models/model_{epoch:02d}', 

        monitor='val_loss', verbose=False, 

        save_last=False, save_top_k=1, save_weights_only=False, 

        mode='min', period=1, prefix='')

    trainer = pl.Trainer(gpus=-1 if torch.cuda.is_available() else None, max_epochs=30, checkpoint_callback=checkpoint_callback)

    dm = MoADataModule(fold=k)

    net = Model(879, 206) # Input Features, Output Targets

    pylitModel = PLitMoAModule(hparams={"lr":LR}, model=net)

    trainer.fit(pylitModel, dm)

    

    print(checkpoint_callback.best_model_path)