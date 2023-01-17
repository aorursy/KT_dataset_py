

import copy
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import GroupKFold
from torch.utils.data import Dataset
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, Subset
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
from tqdm.notebook import trange
from time import time


root_dir = Path('/kaggle/input/osic-pulmonary-fibrosis-progression')
model_dir = Path('/kaggle/working')
num_kfolds = 5
batch_size = 32
learning_rate = 3e-3
num_epochs = 1000
es_patience = 20
quantiles = (0.2, 0.5, 0.8)
model_name ='tenchSquad'
tensorboard_dir = Path('/kaggle/working/runs')


class ClinicalDataset(Dataset):
    def __init__(self, root_dir, mode, transform=None):
        self.transform = transform
        self.mode = mode

        tr = pd.read_csv(Path(root_dir)/"train.csv")
        tr.drop_duplicates(keep=False, inplace=True, subset=['Patient', 'Weeks'])
        chunk = pd.read_csv(Path(root_dir)/"test.csv")

        sub = pd.read_csv(Path(root_dir)/"sample_submission.csv")
        sub['Patient'] = sub['Patient_Week'].apply(lambda x: x.split('_')[0])
        sub['Weeks'] = sub['Patient_Week'].apply(lambda x: int(x.split('_')[-1]))
        sub = sub[['Patient', 'Weeks', 'Confidence', 'Patient_Week']]
        sub = sub.merge(chunk.drop('Weeks', axis=1), on="Patient")

        tr['WHERE'] = 'train'
        chunk['WHERE'] = 'val'
        sub['WHERE'] = 'test'
        data = tr.append([chunk, sub])

        data['min_week'] = data['Weeks']
        data.loc[data.WHERE == 'test', 'min_week'] = np.nan
        data['min_week'] = data.groupby('Patient')['min_week'].transform('min')

        base = data.loc[data.Weeks == data.min_week]
        base = base[['Patient', 'FVC']].copy()
        base.columns = ['Patient', 'min_FVC']
        base['nb'] = 1
        base['nb'] = base.groupby('Patient')['nb'].transform('cumsum')
        base = base[base.nb == 1]
        base.drop('nb', axis=1, inplace=True)

        data = data.merge(base, on='Patient', how='left')
        data['base_week'] = data['Weeks'] - data['min_week']
        del base

        COLS = ['Sex', 'SmokingStatus']
        self.FE = []
        for col in COLS:
            for mod in data[col].unique():
                self.FE.append(mod)
                data[mod] = (data[col] == mod).astype(int)

        data['age'] = (data['Age'] - data['Age'].min()) / \
                      (data['Age'].max() - data['Age'].min())
        data['BASE'] = (data['min_FVC'] - data['min_FVC'].min()) / \
                       (data['min_FVC'].max() - data['min_FVC'].min())
        data['week'] = (data['base_week'] - data['base_week'].min()) / \
                       (data['base_week'].max() - data['base_week'].min())
        data['percent'] = (data['Percent'] - data['Percent'].min()) / \
                          (data['Percent'].max() - data['Percent'].min())
        self.FE += ['age', 'percent', 'week', 'BASE']

        self.raw = data.loc[data.WHERE == mode].reset_index()
        del data

    def __len__(self):
        return len(self.raw)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = {
            'patient_id': self.raw['Patient'].iloc[idx],
            'features': self.raw[self.FE].iloc[idx].values,
            'target': self.raw['FVC'].iloc[idx]
        }
        if self.transform:
            sample = self.transform(sample)

        return sample

    def group_kfold(self, n_splits):
        gkf = GroupKFold(n_splits=n_splits)
        groups = self.raw['Patient']
        for train_idx, val_idx in gkf.split(self.raw, self.raw, groups):
            train = Subset(self, train_idx)
            val = Subset(self, val_idx)
            yield train, val

    def group_split(self, test_size=0.2):
        """To test no-kfold
        """
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size)
        groups = self.raw['Patient']
        idx = list(gss.split(self.raw, self.raw, groups))
        train = Subset(self, idx[0][0])
        val = Subset(self, idx[0][1])
        return train, val


class QuantModel(nn.Module):
    def __init__(self, in_tabular_features=9, out_quantiles=3):
        super(QuantModel, self).__init__()
        self.fc1 = nn.Linear(in_tabular_features, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, out_quantiles)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def quantile_loss(preds, target, quantiles):
    assert not target.requires_grad
    assert preds.size(0) == target.size(0)
    losses = []
    for i, q in enumerate(quantiles):
        errors = target - preds[:, i]
        losses.append(torch.max((q - 1) * errors, q * errors).unsqueeze(1))
    loss = torch.mean(torch.sum(torch.cat(losses, dim=1), dim=1))
    return loss

# Helper class that monitors training
class Monitor:
    def __init__(self, model, es_patience, experiment_name, tensorboard_dir,
                 num_epochs, dataset_sizes, model_file):

        self.model = model
        self.model_file = model_file
        self.es_patience = es_patience
        self.tensorboard_dir = tensorboard_dir
        self.dataset_sizes = dataset_sizes
        date_time = datetime.now().strftime("%Y%m%d-%H%M")
        log_dir = tensorboard_dir / f'{experiment_name}-{date_time}'
        self.w = SummaryWriter(log_dir)

        self.bar = trange(num_epochs, desc=experiment_name)

        self.epoch_loss = {'train': np.inf, 'val': np.inf}
        self.epoch_metric = {'train': -np.inf, 'val': -np.inf}
        self.best_loss = np.inf
        self.best_model_wts = None

        self.e = {'train': 0, 'val': 0}  # epoch counter
        self.t = {'train': 0, 'val': 0}  # global time-step (never resets)
        self.running_loss = 0.0
        self.running_metric = 0.0
        self.es_counter = 0

    def reset_epoch(self):
        self.running_loss = 0.0
        self.running_metric = 0.0

    def step(self, loss, inputs, preds, targets, phase):
        self.running_loss += loss.item() * inputs.size(0)
        self.running_metric += self.metric(preds, targets).sum()
        self.t[phase] += 1

    def log_epoch(self, phase):
        self.epoch_loss[phase] = self.running_loss / self.dataset_sizes[phase]
        self.epoch_metric[phase] = self.running_metric / self.dataset_sizes[phase]
        self.bar.set_postfix(
            a_train_loss=f'{self.epoch_loss["train"]:0.1f}',
            b_val_loss=f'{self.epoch_loss["val"]:0.1f}',
            c_train_metric=f'{self.epoch_metric["train"]:0.4f}',
            d_val_metric=f'{self.epoch_metric["val"]:0.4f}',
            es_counter=self.es_counter
        )
        self.w.add_scalar(
            f'Loss/{phase}', self.epoch_loss[phase], self.e[phase])
        self.w.add_scalar(
            f'Accuracy/{phase}', self.epoch_metric[phase], self.e[phase])

        self.e[phase] += 1

        # Early stop and model backup
        early_stop = False
        if phase == 'val':
            if self.epoch_loss['val'] < self.best_loss:
                self.best_loss = self.epoch_loss['val']
                self.best_model_wts = copy.deepcopy(self.model.state_dict())
                torch.save(self.best_model_wts, self.model_file)
                self.es_counter = 0
            else:
                self.es_counter += 1
                if self.es_counter >= self.es_patience:
                    early_stop = True
                    self.bar.close()

        return early_stop

    @staticmethod
    def metric(preds, targets):
        sigma = preds[:, 2] - preds[:, 0]
        sigma[sigma < 70] = 70
        delta = (preds[:, 1] - targets).abs()
        delta[delta > 1000] = 1000
        return -np.sqrt(2) * delta / sigma - torch.log(np.sqrt(2) * sigma)
models = []

# Load the data
data = ClinicalDataset(root_dir=root_dir, mode='train')
folds = data.group_kfold(num_kfolds)
t0 = time()

for fold, (trainset, valset) in enumerate(folds):
    # Prepare to save model weights
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    now = datetime.now()
    fname = f'{model_name}-{now.year}{now.month:02d}{now.day:02d}_{fold}.pth'
    model_file = Path(model_dir) / fname

    dataset_sizes = {'train': len(trainset), 'val': len(valset)}
    dataloaders = {
        'train': DataLoader(trainset, batch_size=batch_size,
                            shuffle=True, num_workers=2),
        'val': DataLoader(valset, batch_size=batch_size,
                          shuffle=False, num_workers=2)
    }

    # Create the model and optimizer
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = QuantModel().to(device)
    optimizer = Adam(model.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=20, gamma=0.5)
    monitor = Monitor(
        model=model,
        es_patience=es_patience,
        experiment_name=f'{model_name}_fold_{fold}',
        tensorboard_dir=tensorboard_dir,
        num_epochs=num_epochs,
        dataset_sizes=dataset_sizes,
        model_file=model_file
    )

    # Training loop
    for epoch in monitor.bar:
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            monitor.reset_epoch()

            # Iterate over data
            for batch in dataloaders[phase]:
                inputs = batch['features'].float().to(device)
                targets = batch['target'].to(device)

                # zero the parameter gradients
                optimizer.zero_grad()
                # forward
                # track gradients if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    preds = model(inputs)
                    loss = quantile_loss(preds, targets, quantiles)
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                monitor.step(loss, inputs, preds, targets, phase)

            # epoch statistics
            early_stop = monitor.log_epoch(phase)

        if early_stop:
            break

        # Updates the learning rate
        scheduler.step()

    # load best model weights
    model.load_state_dict(monitor.best_model_wts)
    models.append(model)

print(f'Training complete! Time: {timedelta(seconds=time() - t0)}')

data = ClinicalDataset(root_dir, mode='test')
avg_preds = np.zeros((len(data), len(quantiles)))

for model in models:
    dataloader = DataLoader(data, batch_size=batch_size,
                            shuffle=False, num_workers=2)
    preds = []
    for batch in dataloader:
        inputs = batch['features'].float()
        with torch.no_grad():
            x = model(inputs)
            preds.append(x)

    preds = torch.cat(preds, dim=0).numpy()
    avg_preds += preds

avg_preds /= len(models)
df = pd.DataFrame(data=avg_preds, columns=list(quantiles))
df['Patient_Week'] = data.raw['Patient_Week']
df['FVC'] = df[quantiles[1]]
df['Confidence'] = df[quantiles[2]] - df[quantiles[0]]
df = df.drop(columns=list(quantiles))
df.to_csv('submission.csv', index=False)
print(len(df))
df.head()