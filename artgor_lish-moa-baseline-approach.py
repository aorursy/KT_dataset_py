!pip install /kaggle/input/tensorboard-220/tensorboard-2.2.0-py3-none-any.whl -q

!pip install /kaggle/input/pytorch-lightning/pytorch_lightning-0.9.0-py3-none-any.whl -q
import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import os

from typing import List, Dict, Optional

from pytorch_lightning import Callback

import numpy as np

from torch.utils.data import Dataset

from sklearn.model_selection import RepeatedKFold

import pandas as pd

import pytorch_lightning as pl

import torch

from sklearn.model_selection import train_test_split

import torch

from torch import nn as nn

import torch.nn.functional as F

from torch.autograd import Variable

import math

import random

from torch.utils.data import TensorDataset, DataLoader

from typing import Dict, Union

from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
def set_seed(seed: int = 666) -> None:

    np.random.seed(seed)

    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    torch.backends.cudnn.benchmark = False

    torch.backends.cudnn.deterministic = True

    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)

    torch.cuda.manual_seed_all(seed)

set_seed()
sub = pd.read_csv('/kaggle/input/lish-moa/sample_submission.csv')

train_features = pd.read_csv('/kaggle/input/lish-moa/train_features.csv')

test_features = pd.read_csv('/kaggle/input/lish-moa/test_features.csv')

train_targets_nonscored = pd.read_csv('/kaggle/input/lish-moa/train_targets_nonscored.csv')

train_targets_scored = pd.read_csv('/kaggle/input/lish-moa/train_targets_scored.csv')
train_features.shape
train_features.head()
train_features.cp_type.value_counts(normalize=True)
control_group = train_features.loc[train_features.cp_type == 'ctl_vehicle', 'sig_id']

train_targets_scored.loc[train_targets_scored['sig_id'].isin(control_group)].sum()[1:].sum()
test_features.cp_type.value_counts(normalize=True)
train_features.cp_time.value_counts(normalize=True)
train_features.cp_dose.value_counts(normalize=True)
plt.hist(train_targets_scored.mean())

plt.title('Distribution of mean target in each target column');
train_targets_scored.mean().min(), train_targets_scored.mean().mean(), train_targets_scored.mean().max()
no_control_target = train_targets_scored.loc[~train_targets_scored['sig_id'].isin(control_group)]
plt.hist(no_control_target.mean())

plt.title('Distribution of mean target in each target column without control group');
no_control_target.mean().min(), no_control_target.mean().mean(), no_control_target.mean().max()
plt.plot(train_features.loc[train_features['sig_id'] == 'id_79fb45fe7', [col for col in train_features if 'g-' in col]].values.reshape(-1, 1));

plt.title('g- value of id_79fb45fe7');
plt.plot(sorted(train_features.loc[train_features['sig_id'] == 'id_79fb45fe7', [col for col in train_features if 'g-' in col]].values.reshape(-1, 1)))

plt.title('sorted g- value of id_79fb45fe7');
s = pd.DataFrame({'sig_id': test_features['sig_id'].values})
s[train_targets_scored.columns[1:]] = 0.0001
control_group = test_features.loc[test_features.cp_type == 'ctl_vehicle', 'sig_id']
s.loc[s['sig_id'].isin(control_group), train_targets_scored.columns[1:]] = 0
s.to_csv('basic_submission.csv', index=False)
train_features = pd.concat([train_features, pd.get_dummies(train_features['cp_time'], prefix='cp_time')], axis=1)

train_features = pd.concat([train_features, pd.get_dummies(train_features['cp_dose'], prefix='cp_dose')], axis=1)

train_features = pd.concat([train_features, pd.get_dummies(train_features['cp_type'], prefix='cp_type')], axis=1)

# train_features = train_features.loc[train_features['cp_type'] != 'ctl_vehicle']

train_features = train_features.drop(['cp_type', 'cp_time', 'cp_dose'], axis=1)

# train_targets_scored = train_targets_scored.loc[train_targets_scored['sig_id'].isin(train_features['sig_id'])]
class MoADataset(Dataset):

    def __init__(

        self,

        data,

        targets = None,

        targets1 = None,

        mode = 'train'

    ):

        """



        Args:

        """



        self.mode = mode

        self.data = data

        self.targets = targets

        self.targets1 = targets1



    def __getitem__(self, idx: int) -> Dict[str, np.array]:

        data = self.data[idx]

        if self.targets is not None:

            target = self.targets[idx]

            target1 = self.targets1[idx]

        else:

            target = np.zeros((206,))

            target1 = np.zeros((402,))

            

        sample = {'data': torch.tensor(data).float(),

                  'target': torch.tensor(target).float(),

                  'target1': torch.tensor(target1).float()}



        return sample



    def __len__(self) -> int:

        return len(self.data)
class MoADataModule(pl.LightningDataModule):

    def __init__(self, hparams: Dict,

                 train_data, train_targets, train_targets1,

                 valid_data, valid_targets, valid_targets1):

        super().__init__()

        self.hparams = hparams

        self.train_data = train_data

        self.train_targets = train_targets

        self.train_targets1 = train_targets1

        self.valid_data = valid_data

        self.valid_targets = valid_targets

        self.valid_targets1 = valid_targets1



    def prepare_data(self):

        pass



    def setup(self, stage=None):



        

        self.train_dataset = MoADataset(data=self.train_data.iloc[:, 1:].values,

                                        targets=self.train_targets.iloc[:, 1:].values,

                                        targets1=self.train_targets1.iloc[:, 1:].values)

        self.valid_dataset = MoADataset(data=self.valid_data.iloc[:, 1:].values,

                                        targets=self.valid_targets.iloc[:, 1:].values,

                                        targets1=self.valid_targets1.iloc[:, 1:].values)



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

n_h_layers = 2048

learning_rate = 1e-3

criterion = nn.BCEWithLogitsLoss()



class Net(nn.Module):

    def __init__(self, n_in, n_h, n_out, n_out1):

        super(Net, self).__init__()

        self.fc1 = nn.Linear(n_in, n_h)

        self.fc2 = nn.Linear(n_h, math.ceil(n_h/4))

        self.fc3 = nn.Linear(math.ceil(n_h/4), n_out)

        self.fc4 = nn.Linear(math.ceil(n_h/4), n_out1)

        self.bn = nn.BatchNorm1d(n_in)

        self.bn1 = nn.BatchNorm1d(n_h)

        self.bn2 = nn.BatchNorm1d(math.ceil(n_h/4))

        self.drop = nn.Dropout(0.2)

        self.n_out = n_out

        self.selu = nn.SELU()

        self.sigm = nn.Sigmoid()

    def forward(self, x, targets, targets1):

        

        

        self.loss = criterion

        x = self.fc1(self.bn(x))

        x = self.selu(x)

        x = self.fc2(self.drop(self.bn1(x)))

        x = self.selu(x)

        

        # scored targets

        x1 = self.fc3(self.bn2(x))

        # non scored targets

        x2 = self.fc4(self.bn2(x))

        loss = (self.loss(x1, targets) + self.loss(x2, targets1)) / 2

        real_loss = self.loss(x1, targets)

        # probabilities

        out = self.sigm(x1)

        return out, loss, real_loss

    

class LitMoA(pl.LightningModule):

    def __init__(self, hparams, model):

        super(LitMoA, self).__init__()

        self.hparams = hparams

        self.model = model



    def forward(self, x, targets, targets1, *args, **kwargs):

        return self.model(x, targets, targets1)



    def configure_optimizers(self):

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=0.001)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=2)



        return (

            [optimizer],

            [{'scheduler': scheduler, 'interval': 'epoch', 'monitor': 'valid_loss'}],

        )



    def training_step(

        self, batch: torch.Tensor, batch_idx: int

    ) -> Union[int, Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]]:

        data = batch['data']

        target = batch['target']

        target1 = batch['target1']

        out, loss, real_loss = self(data, target, target1)

        logs = {'train_loss': loss, 'real_train_loss': real_loss}

        return {

            'loss': loss, 'real_train_loss': real_loss,

            'log': logs,

            'progress_bar': logs

        }



    def training_epoch_end(self, outputs):

        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()

        real_avg_loss = torch.stack([x['real_train_loss'] for x in outputs]).mean()

        logs = {'train_loss': avg_loss, 'real_train_loss': real_avg_loss}

        return {'log': logs, 'progress_bar': logs}



    def validation_step(

        self, batch: torch.Tensor, batch_idx: int

    ) -> Union[int, Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]]:

        data = batch['data']

        target = batch['target']

        target1 = batch['target1']

        out, loss, real_loss = self(data, target, target1)

        logs = {'valid_loss': loss, 'real_valid_loss': real_loss}



        return {

            'loss': loss, 'real_valid_loss': real_loss,

            'log': logs,

            'progress_bar': logs,

        }



    def validation_epoch_end(self, outputs):

        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()

        real_avg_loss = torch.stack([x['real_valid_loss'] for x in outputs]).mean()



        logs = {'valid_loss': avg_loss, 'real_valid_loss': real_avg_loss}

        return {'valid_loss': avg_loss, 'log': logs, 'progress_bar': logs}

test_features = pd.concat([test_features, pd.get_dummies(test_features['cp_time'], prefix='cp_time')], axis=1)

test_features = pd.concat([test_features, pd.get_dummies(test_features['cp_dose'], prefix='cp_dose')], axis=1)

test_features = pd.concat([test_features, pd.get_dummies(test_features['cp_type'], prefix='cp_type')], axis=1)

# test_features = test_features.loc[test_features['cp_type'] != 'ctl_vehicle']

test_features = test_features.drop(['cp_type', 'cp_time', 'cp_dose'], axis=1)

test_dataset = MoADataset(data=test_features.iloc[:, 1:].values)

test_loader = torch.utils.data.DataLoader(

            test_dataset,

            batch_size=1024,

            num_workers=0,

            shuffle=False,

        )
class MetricsCallback(Callback):

    """PyTorch Lightning metric callback."""



    def __init__(self):

        super().__init__()

        self.metrics = []



    def on_validation_end(self, trainer, pl_module):

        self.metrics.append(trainer.callback_metrics)
hparams = {}
rkf = RepeatedKFold(n_splits=6, n_repeats=8, random_state=42)

n = rkf.get_n_splits()

all_predictions = None

scores = []

for train_index, valid_index in rkf.split(train_features):

    

    net = Net(n_in = 879, n_h = n_h_layers, n_out = 206, n_out1 = 402)

    # split data

    train_data, valid_data = train_features.iloc[train_index, :], train_features.iloc[valid_index, :]

    train_targets_scored_train, train_targets_scored_valid = train_targets_scored.iloc[train_index, :], train_targets_scored.iloc[valid_index, :]

    train_targets_nonscored_train, train_targets_nonscored_valid = train_targets_nonscored.iloc[train_index, :], train_targets_nonscored.iloc[valid_index, :]

            

    model = LitMoA(hparams=hparams, model=net)

    

    dm = MoADataModule(hparams=hparams,

                       train_data=train_data, train_targets=train_targets_scored_train, train_targets1=train_targets_nonscored_train,

                       valid_data=valid_data, valid_targets=train_targets_scored_valid, valid_targets1=train_targets_nonscored_valid)

    

    hparams = {}

    metrics_callback = MetricsCallback()

    trainer = pl.Trainer(

            early_stop_callback=EarlyStopping(monitor='valid_loss', patience=10, mode='min'),

            checkpoint_callback=ModelCheckpoint(monitor='valid_loss', save_top_k=1, filepath='{epoch}_{valid_loss:.4f}', mode='min'),

            gpus=1,

            max_epochs=50,

            log_save_interval=100,

            num_sanity_val_steps=0,

            gradient_clip_val=0.5,

            weights_summary='full',

            callbacks = [metrics_callback]

    )

    

    

    trainer.fit(model, dm)

    

    score = metrics_callback.metrics[-1]['real_valid_loss'].item()

    scores.append(score)

    

    predictions = np.zeros((test_features.shape[0], 206))

    model_inference = model.model

    model_inference.eval()

    

    for ind, batch in enumerate(test_loader):

        p = model_inference(batch['data'], batch['target'], batch['target1'])[0].detach().cpu().numpy()

        predictions[ind * 1024:(ind + 1) * 1024] = p

        

    if all_predictions is None:

        all_predictions = predictions

    else:

        all_predictions += predictions

        

all_predictions = all_predictions / n
print(f'Mean score: {np.mean(scores):.4f}. Std score: {np.std(scores):.4f}')
all_predictions.max()
plt.hist(all_predictions.mean())

plt.title('Distribution of prediction means');
s = pd.DataFrame({'sig_id': test_features['sig_id'].values})
for col in train_targets_scored.columns[1:].tolist():

    s[col] = 0
s.loc[:, train_targets_scored.columns[1:]] = all_predictions
test_features = pd.read_csv('/kaggle/input/lish-moa/test_features.csv')

s.loc[s['sig_id'].isin(test_features.loc[test_features['cp_type'] =='ctl_vehicle', 'sig_id']), train_targets_scored.columns[1:]] = 0
s.to_csv('submission.csv', index=False)