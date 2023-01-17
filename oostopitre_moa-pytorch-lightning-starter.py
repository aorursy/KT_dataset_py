import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
!pip install ../input/installs/tensorboard-2.2.0-py3-none-any.whl -qqq

!pip install ../input/installs/pytorch_lightning-0.9.0-py3-none-any.whl -qqq
import pytorch_lightning as pl

import torch



print(f'PyTorch Lightning: {pl.__version__}')

print(f'PyTorch: {torch.__version__}')
# IMPORTS

import os

import warnings



import numpy as np

import pandas as pd

import pytorch_lightning as pl

import torch

import torch.nn.functional as F

from pytorch_lightning.metrics.functional import accuracy

from sklearn.model_selection import train_test_split

from torch import nn

from torch.utils.data import DataLoader

# CONFIG



# Data

class config:

    DATA_DIR = '../input/lish-moa/'

    # MLP Training

    BATCH_SIZE = 32

    EPOCHS = 10

    LR = 3e-4

    MOMENTUM = 0.9

    NUM_CLASSES = 206

# DATASET



class MOADataset():

    def __init__(self, root, key='train'):

        self.data_dir = os.path.join(os.path.expanduser(root))



        # key - train or valid or test

        self.key = key

        self.df, self.target = self.process_df()

        self.df = self.df.drop(['sig_id', 'cp_type', 'cp_time', 'cp_dose'], axis=1)

        self.target = self.target.drop(['sig_id'], axis=1)



    def __getitem__(self, index):

        row, target = self.df.iloc[index].values, self.target.iloc[index].values

        return torch.from_numpy(row).float(), torch.from_numpy(target).long()



    def __len__(self):

        return len(self.df)



    def process_df(self):

        if self.key == 'train':

            features = pd.read_csv(os.path.join(self.data_dir, 'train_features.csv'))

            labels = pd.read_csv(os.path.join(self.data_dir, 'train_targets_scored.csv'))

            features, _, labels, _ = train_test_split(

                features,

                labels,

                test_size=0.2,

                random_state=0,

            )

        elif self.key == 'valid':

            features = pd.read_csv(os.path.join(self.data_dir, 'train_features.csv'))

            labels = pd.read_csv(os.path.join(self.data_dir, 'train_targets_scored.csv'))

            _, features, _, labels = train_test_split(

                features,

                labels,

                test_size=0.2,

                random_state=0,

            )

        elif self.key == 'test':

            features = pd.read_csv(os.path.join(self.data_dir, 'test_features.csv'))

            labels = pd.read_csv(os.path.join(self.data_dir, 'sample_submission.csv'))



        return features, labels





class MOADataModule(pl.LightningDataModule):

    def __init__(self, data_dir, batch_size, num_classes):

        super().__init__()



        self.data_dir = data_dir

        self.batch_size = batch_size

        self.num_classes = num_classes



    def prepare_data(self):

        pass



    def setup(self, stage=None):

        if stage == 'fit':

            self.tabular_train = MOADataset(

                self.data_dir,

                key='train',

            )

            self.tabular_valid = MOADataset(

                self.data_dir,

                key='valid',

            )



            assert self.tabular_train.df.shape[1] == self.tabular_valid.df.shape[1]

            self.num_features = self.tabular_train.df.shape[1]



        if stage == 'test' or stage is None:

            self.tabular_test = MOADataset(

                self.data_dir,

                key='test',

            )



    def train_dataloader(self):

        return DataLoader(self.tabular_train, batch_size=self.batch_size, shuffle=True)



    def val_dataloader(self):

        return DataLoader(self.tabular_valid, batch_size=self.batch_size, shuffle=False)



    def test_dataloader(self):

        return DataLoader(self.tabular_test, batch_size=self.batch_size, shuffle=False)

# MODEL



class MOAModel(pl.LightningModule):

    def __init__(self, num_features, num_classes, learning_rate=2e-4, hidden_size=64):



        super().__init__()



        self.hidden_size = hidden_size

        self.learning_rate = config.LR



        # Build model

        self.model = nn.Sequential(

            nn.Linear(num_features, hidden_size),

            nn.ReLU(),

            nn.Dropout(0.1),

            nn.Linear(hidden_size, hidden_size),

            nn.ReLU(),

            nn.Dropout(0.1),

            nn.Linear(hidden_size, num_classes),  # batchsize x num_classes

        )



    def configure_optimizers(self):

        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        return optimizer



    def forward(self, x):

        x = self.model(x)

        return x  # batchsize x num_classes



    def training_step(self, batch, batch_idx):

        x, y = batch

        y_hat = self(x)

        loss = F.binary_cross_entropy_with_logits(y_hat, y.type_as(y_hat))

        result = pl.TrainResult(minimize=loss)

        result.log('train_loss', loss, prog_bar=True)

        return result



    def validation_step(self, batch, batch_idx):

        x, y = batch

        y_hat = self(x)

        loss = F.binary_cross_entropy_with_logits(y_hat, y.type_as(y_hat))

        result = pl.EvalResult(checkpoint_on=loss)

        result.log('val_loss', loss, prog_bar=True)

        return result



    def test_step(self, batch, batch_idx):

        x, y = batch

        logits = self(x)

        preds = F.sigmoid(logits)

        result = pl.EvalResult()

        result.write('preds', preds, filename='predictions.pt')

        return result



# TRAINING

DATA_DIR = config.DATA_DIR



pl.seed_everything(1234)



dm = MOADataModule(

    data_dir=DATA_DIR,

    batch_size=config.BATCH_SIZE,

    num_classes=config.NUM_CLASSES,

)

dm.setup(stage='fit')

model = MOAModel(dm.num_features, dm.num_classes)
# add multiple loggers

tb_logger = pl.loggers.TensorBoardLogger('tb_logs/', name='default')

csv_logger = pl.loggers.CSVLogger('csv_logs/', name='default')



trainer = pl.Trainer(

    # fast_dev_run=config.DEBUG,

    # num_sanity_val_steps=5,

    # limit_train_batches=5,

    # limit_val_batches=5,

    # limit_test_batches=5,

    gpus=(1 if torch.cuda.is_available() else 0),

    max_epochs=config.EPOCHS,

    progress_bar_refresh_rate=30,

    weights_summary='top',

#     logger=[tb_logger, csv_logger],

)



trainer.fit(model, datamodule=dm)
# TESTING



trainer.test(model=model, datamodule=dm)
sample_submission = pd.read_csv('../input/lish-moa/sample_submission.csv')

sample_cols = list(sample_submission.columns)[1:]



pred_values = torch.load('predictions.pt')

preds = pd.DataFrame([x['preds'] for x in pred_values], columns=sample_cols)



submission = pd.concat([sample_submission[['sig_id']], preds], axis=1)

submission.head()



submission.to_csv('submission.csv', index=False)