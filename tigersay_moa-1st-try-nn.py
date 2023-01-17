import numpy as np

import pandas as pd

from sklearn.model_selection import StratifiedKFold, KFold

from sklearn.metrics import log_loss

from tqdm.notebook import tqdm



import torch

import torch.nn as nn

import torch.optim as optim

import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader



import sys

sys.path.append('../input/iterative-stratification/iterative-stratification-master')

from iterstrat.ml_stratifiers import MultilabelStratifiedKFold



import warnings

warnings.filterwarnings('ignore')



#used net arch from kaggle.com/nicohrubec/pytorch-multilabel-neural-network/
X_train = pd.read_csv('../input/lish-moa/train_features.csv')

y_train = pd.read_csv('../input/lish-moa/train_targets_scored.csv')

X_test = pd.read_csv('../input/lish-moa/test_features.csv')



submit = pd.read_csv('../input/lish-moa/sample_submission.csv')
def preprocess(df):

    df = df.copy()

    df.loc[:, 'cp_type'] = df.loc[:, 'cp_type'].map({'trt_cp': 0, 'ctl_vehicle': 1})

    df.loc[:, 'cp_dose'] = df.loc[:, 'cp_dose'].map({'D1': 0, 'D2': 1})

    del df['sig_id']

    return df



train = preprocess(X_train)

test = preprocess(X_test)



del y_train['sig_id']



y_train = y_train.loc[train['cp_type']==0].reset_index(drop=True)

train = train.loc[train['cp_type']==0].reset_index(drop=True)
nfolds = 7

nstarts = 1

nepochs = 50

batch_size = 128

val_batch_size = batch_size * 4

ntargets = y_train.shape[1]

targets = [col for col in y_train.columns]

criterion = nn.BCELoss()

kfold = MultilabelStratifiedKFold(n_splits=nfolds, random_state=517, shuffle=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train = train.values

test = test.values

y_train = y_train.values
class Model(nn.Module):

    def __init__(self, num_columns):

        super(Model, self).__init__()

        self.batch_norm1 = nn.BatchNorm1d(num_columns)

        self.dropout1 = nn.Dropout(0.2)

        self.dense1 = nn.utils.weight_norm(nn.Linear(num_columns, 2048))

        

        self.batch_norm2 = nn.BatchNorm1d(2048)

        self.dropout2 = nn.Dropout(0.5)

        self.dense2 = nn.utils.weight_norm(nn.Linear(2048, 1024))

        

        self.batch_norm3 = nn.BatchNorm1d(1024)

        self.dropout3 = nn.Dropout(0.5)

        self.dense3 = nn.utils.weight_norm(nn.Linear(1024, 206))

    

    def forward(self, x):

        x = self.batch_norm1(x)

        x = self.dropout1(x)

        x = F.relu(self.dense1(x))

        

        x = self.batch_norm2(x)

        x = self.dropout2(x)

        x = F.relu(self.dense2(x))

        

        x = self.batch_norm3(x)

        x = self.dropout3(x)

        x = F.sigmoid(self.dense3(x))

        

        return x
class Dataset(Dataset):

    def __init__(self, df, targets, mode='train'):

        self.mode = mode

        #self.feats = feats_idx

        #self.data = df[:, feats_idx]

        self.data = df

        if mode=='train':

            self.targets = targets

    

    def __getitem__(self, idx):

        if self.mode == 'train':

            return torch.FloatTensor(self.data[idx]), torch.FloatTensor(self.targets[idx])

        elif self.mode == 'test':

            return torch.FloatTensor(self.data[idx]), 0

        

    def __len__(self):

        return len(self.data)
for n, (tr, te) in enumerate(kfold.split(y_train, y_train)):

    print(f'Train fold {n+1}')

    xtrain, xval = train[tr], train[te]

    ytrain, yval = y_train[tr], y_train[te]



    train_set = Dataset(xtrain, ytrain)

    val_set = Dataset(xval, yval)



    dataloaders = {

        'train': DataLoader(train_set, batch_size=batch_size, shuffle=True),

        'val': DataLoader(val_set, batch_size=val_batch_size, shuffle=False)

    }



    model = Model(X_train.shape[1]-1).to(device)

    checkpoint_path = f'repeat:{1}_Fold:{n+1}.pt'

    optimizer = optim.Adam(model.parameters(), weight_decay=1e-5)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, eps=1e-4, verbose=True)

    best_loss = {'train': np.inf, 'val': np.inf}



    for epoch in range(nepochs):

        epoch_loss = {'train': 0.0, 'val': 0.0}



        for phase in ['train', 'val']:

            if phase == 'train':

                model.train()

            else:

                model.eval()



            running_loss = 0.0



            for i, (x, y) in enumerate(dataloaders[phase]):

                x, y = x.to(device), y.to(device)



                optimizer.zero_grad()



                with torch.set_grad_enabled(phase=='train'):

                    preds = model(x)

                    loss = criterion(preds, y)



                    if phase=='train':

                        loss.backward()

                        optimizer.step()



                running_loss += loss.item() / len(dataloaders[phase])



            epoch_loss[phase] = running_loss



        print("Epoch {}/{}   -   loss: {:5.5f}   -   val_loss: {:5.5f}".format(epoch+1, nepochs, epoch_loss['train'], epoch_loss['val']))



        scheduler.step(epoch_loss['val'])



        if epoch_loss['val'] < best_loss['val']:

            best_loss = epoch_loss

            torch.save(model.state_dict(), checkpoint_path)
oof = np.zeros((len(train), nstarts, ntargets))

oof_targets = np.zeros((len(train), ntargets))

preds = np.zeros((len(test), ntargets))
def mean_log_loss(y_true, y_pred):

    metrics = []

    for i, target in enumerate(targets):

        metrics.append(log_loss(y_true[:, i], y_pred[:, i].astype(float), labels=[0,1]))

    return np.mean(metrics)
seed_targets = []

seed_oof = []

seed_preds = np.zeros((len(test), ntargets, nfolds))



for n, (tr, te) in enumerate(kfold.split(y_train, y_train)):

    xval, yval = train[te], y_train[te]

    fold_preds = []



    val_set = Dataset(xval, yval)

    test_set = Dataset(test, None, mode='test')



    dataloaders = {

        'val': DataLoader(val_set, batch_size=val_batch_size, shuffle=False),

        'test': DataLoader(test_set, batch_size=val_batch_size, shuffle=False)

    }



    checkpoint_path = f'repeat:{1}_Fold:{n+1}.pt'

    model = Model(X_train.shape[1]-1).to(device)

    model.load_state_dict(torch.load(checkpoint_path))

    model.eval()



    for phase in ['val', 'test']:

        for i, (x, y) in enumerate(dataloaders[phase]):

            if phase == 'val':

                x, y = x.to(device), y.to(device)

            elif phase == 'test':

                x = x.to(device)



            with torch.no_grad():

                batch_preds = model(x)



                if phase == 'val':

                    seed_targets.append(y)

                    seed_oof.append(batch_preds)

                elif phase == 'test':

                    fold_preds.append(batch_preds)



    fold_preds = torch.cat(fold_preds, dim=0).cpu().numpy()

    seed_preds[:, :, n] = fold_preds



seed_targets = torch.cat(seed_targets, dim=0).cpu().numpy()

seed_oof = torch.cat(seed_oof, dim=0).cpu().numpy()

seed_preds = np.mean(seed_preds, axis=2)



oof_targets = seed_targets

oof[:, 0, :] = seed_oof

preds += seed_preds / nstarts



oof = np.mean(oof, axis=1)

print("Overall score is {:5.5f}".format(mean_log_loss(oof_targets, oof)))
submit[targets] = preds

submit.loc[X_test['cp_type']=='ctl_vehicle', targets] = 0

submit.to_csv('submission.csv', index=False)