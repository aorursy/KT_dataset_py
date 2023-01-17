# Install pytorch-lightning

!pip install ../input/pytorch-lightening/tensorboard-2.2.0-py3-none-any.whl -q

!pip install ../input/pytorch-lightening/pytorch_lightning-0.9.0-py3-none-any.whl -q
import pandas as pd

import pytorch_lightning as pl

from torch.utils.data import Dataset, DataLoader

import os

from  torch import nn

import torch

import numpy as np
class MoATestDataset(Dataset):

    

    def __init__(self, features):

        self.features = features

        

    def __len__(self):

        return self.features.shape[0]

        

    def __getitem__(self, index):

        return {

            "x": torch.tensor(self.features[index, :], dtype=torch.float)

        }
LR = 0.001

F_DROPOUT = 0.45

LAYERS = [4096, 2048, 1024, 512]

LABEL_SMOOTHING = 0.001



class Model(nn.Module):

    def __init__(self, num_features, num_targets):

        super().__init__()

        layers = []

        

        # Intermediate layers

        in_size = num_features   

        for i in range(len(LAYERS)):

            out_size = LAYERS[i]

            layers.append(torch.nn.Linear(in_size, out_size, bias=False))

            layers.append(nn.BatchNorm1d(out_size))

            layers.append(nn.Dropout(F_DROPOUT))

            layers.append(nn.PReLU())

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

        

    def forward(self, x):

        return self.model(x)

    

    def configure_optimizers(self):

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hparams["lr"])

        scheduler = {"scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, threshold=0.00001, mode='min', verbose=True),

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
sample_submission = pd.read_csv('../input/lish-moa/sample_submission.csv')

test_features = pd.read_csv('../input/lish-moa/test_features.csv')
sample_submission.shape, test_features.shape
# Convert categorical features into OHE

test_features = pd.concat([test_features, pd.get_dummies(test_features['cp_time'], prefix='cp_time')], axis=1)

test_features = pd.concat([test_features, pd.get_dummies(test_features['cp_dose'], prefix='cp_dose')], axis=1)

test_features = pd.concat([test_features, pd.get_dummies(test_features['cp_type'], prefix='cp_type')], axis=1)

# Delete original categorical features

test_features = test_features.drop(['cp_time', 'cp_dose', 'cp_type'], axis=1)
batch_size=1024

testDataSet = MoATestDataset(test_features.iloc[:, 1:].values)

testDataLoader = DataLoader(testDataSet, batch_size=batch_size, num_workers=4, shuffle=False)

net = Model(879, 206) # Input Features, Output Targets

# pylitModel = PLitMoAModule(hparams={"lr":1e-3}, model=net)

trainer = pl.Trainer()
# Empty array with as many rows

predictions = np.zeros((test_features.shape[0], 206))

model_path = '../input/optuna-moa-pytorch-lightning-kfold/models/'

for modelFileName in os.listdir(model_path):

    # Load each K-Fold model

    model = PLitMoAModule.load_from_checkpoint(checkpoint_path=f"{model_path}{modelFileName}", model=net)

    for index, batch in enumerate(testDataLoader):

        # Sigmoid is used to convert eact predictions into a range between 0 and 1 (probability)

        batch_predictions = torch.sigmoid(model(batch['x'])).detach().cpu().numpy()

        start_index = index*batch_size

        end_index = index*batch_size + batch_predictions.shape[0]

        predictions[start_index:end_index] = predictions[start_index:end_index] + batch_predictions



# Average predictions across KFolds

predictions = np.true_divide(predictions, 5)
sample_submission.iloc[:, 1:] = predictions
# https://www.kaggle.com/c/lish-moa/discussion/180165 

vehicle_indices = test_features[test_features["cp_type_ctl_vehicle"]==1].index.tolist()

sample_submission.iloc[vehicle_indices, 1:] = np.zeros((1, 206))
sample_submission.to_csv('submission.csv', index=False)