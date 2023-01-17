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

import optuna

import torch.nn.functional as F

from optuna.integration import PyTorchLightningPruningCallback
test_features = pd.read_csv('../input/lish-moa/test_features.csv')

train_features = pd.read_csv('../input/lish-moa/train_features.csv')

train_targets_scored = pd.read_csv('../input/lish-moa/train_targets_scored.csv')

train_targets_nonscored = pd.read_csv('../input/lish-moa/train_targets_nonscored.csv')

sample_submission = pd.read_csv('../input/lish-moa/sample_submission.csv')
# From https://github.com/eladhoffer/utils.pytorch/blob/master/cross_entropy.py

def binary_cross_entropy(inputs, target, weight=None, reduction='mean', smooth_eps=None, from_logits=False):

    """cross entropy loss, with support for label smoothing https://arxiv.org/abs/1512.00567"""

    smooth_eps = smooth_eps or 0

    if smooth_eps > 0:

        target = target.float()

        target.add_(smooth_eps).div_(2.)

    if from_logits:

        return F.binary_cross_entropy_with_logits(inputs, target, weight=weight, reduction=reduction)

    else:

        return F.binary_cross_entropy(inputs, target, weight=weight, reduction=reduction)





def binary_cross_entropy_with_logits(inputs, target, weight=None, reduction='mean', smooth_eps=None, from_logits=True):

    return binary_cross_entropy(inputs, target, weight, reduction, smooth_eps, from_logits)





class BCELoss(nn.BCELoss):

    def __init__(self, weight=None, size_average=None, reduce=None, reduction='mean', smooth_eps=None, from_logits=False):

        super(BCELoss, self).__init__(weight, size_average, reduce, reduction)

        self.smooth_eps = smooth_eps

        self.from_logits = from_logits



    def forward(self, input, target):

        return binary_cross_entropy(input, target,

                                    weight=self.weight, reduction=self.reduction,

                                    smooth_eps=self.smooth_eps, from_logits=self.from_logits)





class BCEWithLogitsLoss(BCELoss):

    def __init__(self, weight=None, size_average=None, reduce=None, reduction='mean', smooth_eps=None, from_logits=True):

        super(BCEWithLogitsLoss, self).__init__(weight, size_average,

                                                reduce, reduction, smooth_eps=smooth_eps, from_logits=from_logits)
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

    def __init__(self, batch_size=2048, fold= 0):

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

    def __init__(self, num_features, num_targets, trial):

        super().__init__()

        n_layers = 5  # Let us not play with this

        f_dropout = trial.suggest_float('f_dropout', 0.2, 0.5)

        LAYER_OUTPUTS = [2048, 4096, 2048, 1024, 512]

        layers = []



        # Intermediate layers

        in_size = 879   

        for i in range(n_layers):

            out_size = LAYER_OUTPUTS[i]

#             out_size = trial.suggest_int('n_units_{}'.format(i), 256, 4096)

            layers.append(torch.nn.Linear(in_size, LAYER_OUTPUTS[i], bias=False))

            layers.append(nn.BatchNorm1d(out_size))

            layers.append(nn.Dropout(f_dropout))

            layers.append(nn.PReLU())

            in_size = out_size



        # Final layer

        layers.append(torch.nn.Linear(in_size, 206))

    

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

    def __init__(self, hparams, trial):

        super(PLitMoAModule, self).__init__()

        self.hparams = hparams

        self.hparams["lr"] = trial.suggest_float('lr',

                                             1e-5, 1e-2, log=True)

        self.model= Model(879, 206, trial) # Input Features, Output Targets

        

#         smoothing_factor = trial.suggest_float('smoothing_factor', 0.01, 0.2)

#         self.tr_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        self.criterion = nn.BCEWithLogitsLoss()

        

    def forward(self, x):

        return self.model(x)

    

    def configure_optimizers(self):

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hparams["lr"])

        scheduler = {"scheduler": 

                     torch.optim.lr_scheduler.ReduceLROnPlateau(

                        optimizer, patience=2, 

                        threshold=0.0001, 

                        mode='min', verbose=True),

                    "interval": "epoch",

                    "monitor": "val_loss"}

        return [optimizer], [scheduler]

    

    def training_step(self, batch, batch_index):

        features = batch['x']

        targets = batch['y']

        out = self(features)

        loss = self.criterion(out, targets)

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
# train_targets_positive = train_targets_scored.sum()[1:]

# train_targets_negative = train_targets_scored.shape[0] - train_targets_scored.sum()[1:]

# pos_weight = train_targets_negative/train_targets_positive   

# pos_weight = torch.tensor(pos_weight)  
def objective(trial):



    metrics_callback = MetricsCallback()

    trainer = pl.Trainer(

        logger=False,

        max_epochs=50,

        gpus=-1 if torch.cuda.is_available() else None,

        callbacks=[metrics_callback],

        checkpoint_callback=False, # Do not save any checkpoints

        early_stop_callback=PyTorchLightningPruningCallback(trial, monitor="val_loss"),

    )  

    

#     model = PLitMoAModule(hparams={"lr":1e-3}, trial=trial, pos_weight=pos_weight)

    model = PLitMoAModule(hparams={"lr":1e-3}, trial=trial)

    dm = MoADataModule(fold=1)

    trainer.fit(model, dm)



    return metrics_callback.metrics[-1]["val_loss"].item()
from pytorch_lightning import Callback

class MetricsCallback(Callback):

    """PyTorch Lightning metric callback."""



    def __init__(self):

        super().__init__()

        self.metrics = []



    def on_validation_end(self, trainer, pl_module):

        self.metrics.append(trainer.callback_metrics)
pruner = optuna.pruners.MedianPruner() 

# pruner = optuna.pruners.NopPruner()



study = optuna.create_study(direction="minimize", pruner=pruner)

study.optimize(objective, n_trials=100, gc_after_trial=True, timeout=None)



print("Number of finished trials: {}".format(len(study.trials)))



print("Best trial:")

trial = study.best_trial



print("  Value: {}".format(trial.value))



print("  Params: ")

for key, value in trial.params.items():

    print("    {}: {}".format(key, value))