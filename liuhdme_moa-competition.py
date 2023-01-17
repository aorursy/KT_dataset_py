import os

import sys

import pickle

import random



import pandas as pd

import numpy as np

from scipy import stats

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

sys.path.append('../input/iterative-stratification/iterative-stratification-master')

from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

sys.path.append('../input/rank-gauss')

from gauss_rank_scaler import GaussRankScaler

import tqdm

import seaborn as sns

import matplotlib.pyplot as plt



import torch

import torch.nn as nn

from torch.utils.data import TensorDataset, DataLoader

from torch.nn.modules.loss import _WeightedLoss

import torch.nn.functional as F



from warnings import simplefilter

simplefilter(action='ignore', category=FutureWarning)



sns.set()
# parameters

data_path = '../input/lish-moa/'

no_ctl = True

scale = 'rankgauss'

variance_threshould = 0.7

decompo = 'PCA'

ncompo_genes = 80

ncompo_cells = 10

encoding = 'dummy'

base_seed = 2020
def seed_everything(seed_value):

    random.seed(seed_value)

    np.random.seed(seed_value)

    torch.manual_seed(seed_value)

    os.environ['PYTHONHASHSEED'] = str(seed_value)

    

    if torch.cuda.is_available(): 

        torch.cuda.manual_seed(seed_value)

        torch.cuda.manual_seed_all(seed_value)

        torch.backends.cudnn.deterministic = True

        torch.backends.cudnn.benchmark = False
seed_everything(base_seed)
# read data

test = pd.read_csv(data_path + 'test_features.csv')

train = pd.read_csv(data_path + 'train_features.csv')

targets = pd.read_csv(data_path + 'train_targets_scored.csv')



if no_ctl:

    # 删掉 cp_type==ctl_vehicle 的样本

    print('not_ctl')

    train = train[train['cp_type']!='ctl_vehicle']

    test = test[test['cp_type']!='ctl_vehicle']

    targets = targets.iloc[train.index]

    train.reset_index(drop=True, inplace=True)

    test.reset_index(drop=True, inplace=True)

    targets.reset_index(drop=True, inplace=True)
# 筛掉方差小于 variance_threshould 的特征

data_all = pd.concat([train, test], ignore_index=True)

cols_numeric = [feat for feat in list(data_all.columns) if feat not in ['sig_id', 'cp_type', 'cp_time', 'cp_dose']]

mask = (data_all[cols_numeric].var() >= variance_threshould).values

tmp = data_all[cols_numeric].loc[:, mask]

data_all = pd.concat([data_all[['sig_id', 'cp_type', 'cp_time', 'cp_dose']], tmp], axis=1)

cols_numeric = [feat for feat in list(data_all.columns) if feat not in ['sig_id', 'cp_type', 'cp_time', 'cp_dose']]
def scale_minmax(col):

    return (col - col.min()) / (col.max() - col.min())



def scale_norm(col):

    return (col - col.mean()) / col.std()



if scale == 'boxcox':

    # 通过 BoxCox 正态化

    print('boxcox')

    data_all[cols_numeric] = data_all[cols_numeric].apply(scale_minmax, axis=0)

    trans = []

    for feat in cols_numeric:

        trans_var, lambda_var = stats.boxcox(data_all[feat].dropna() + 1)

        trans.append(scale_minmax(trans_var))

    data_all[cols_numeric] = np.asarray(trans).T

    

elif scale == 'norm':

    # 通过标准化正态化

    print('norm')

    data_all[cols_numeric] = data_all[cols_numeric].apply(scale_norm, axis=0)

    

elif scale == 'minmax':

    # 归一化

    print('minmax')

    data_all[cols_numeric] = data_all[cols_numeric].apply(scale_minmax, axis=0)

    

elif scale == 'rankgauss':

    # RankGauss

    print('rankgauss')

    scaler = GaussRankScaler()

    data_all[cols_numeric] = scaler.fit_transform(data_all[cols_numeric])

    

else:

    pass
# 降维

from sklearn.decomposition import PCA



if decompo == 'PCA':

    print('PCA')

    GENES = [col for col in data_all.columns if col.startswith('g-')]

    CELLS = [col for col in data_all.columns if col.startswith('c-')]



    pca_genes = PCA(n_components=ncompo_genes, random_state=base_seed).fit_transform(data_all[GENES])

    pca_cells = PCA(n_components=ncompo_cells, random_state=base_seed).fit_transform(data_all[CELLS])

    pca_genes = pd.DataFrame(pca_genes, columns=[f'pca_g-{i}' for i in range(ncompo_genes)])

    pca_cells = pd.DataFrame(pca_cells, columns=[f'pca_c-{i}' for i in range(ncompo_cells)])

    data_all = pd.concat([data_all, pca_genes, pca_cells], axis=1)

else:

    pass
# Encoding

if encoding == 'lb':

    print('Label Encoding')

    for feat in ['cp_time', 'cp_dose']:

        data_all[feat] = LabelEncoder().fit_transform(data_all[feat])

elif encoding == 'dummy':

    print('One-hot')

    data_all = pd.get_dummies(data_all, columns=['cp_time', 'cp_dose'])
# 特征生成

GENES = [col for col in data_all.columns if col.startswith('g-')]

CELLS = [col for col in data_all.columns if col.startswith('c-')]

for stats in tqdm.tqdm(['sum', 'mean', 'std', 'kurt', 'skew']):

    data_all['g_'+stats] = getattr(data_all[GENES], stats)(axis=1)

    data_all['c_'+stats] = getattr(data_all[CELLS], stats)(axis=1)

    data_all['gc_'+stats] = getattr(data_all[GENES+CELLS], stats)(axis=1)

    

# for cell in CELLS:

#     data_all[cell + '_squared'] = data_all[cell] ** 2
# 保存数据

import pickle



with open('data_all.pickle', 'wb') as f:

    pickle.dump(data_all, f)
# 读取数据

with open('data_all.pickle', 'rb') as f:

    data_all = pickle.load(f)
# 得到 train_df, test_df

features_todrop = ['sig_id', 'cp_type']

data_all.drop(features_todrop, axis=1, inplace=True)

try:

    targets.drop('sig_id', axis=1, inplace=True)

except:

    pass



train_df = data_all[:train.shape[0]]

train_df.reset_index(drop=True, inplace=True)

train_df = pd.concat([train_df, targets], axis=1)

test_df = data_all[train_df.shape[0]:]

test_df.reset_index(drop=True, inplace=True)
# Hypter-parameters

device = ('cuda' if torch.cuda.is_available() else 'cpu')



# model

hsize = 1024

dropratio = 0.2



# train

batchsize = 128

lr = 0.001

wd = 1e-5

smoothing = 0.001

p_min = smoothing

p_max = 1 - smoothing

nepoch = 20

earlystop = True

earlystop_step = 10



# lr_scheduler, options: ['OneCycleLR', 'ReduceLROnPlateau', 'both']

lr_scheduler = 'OneCycleLR'

# OneCycleLR

pct_start = 0.1

div_factor = 1e3

# ReduceLROnPlateau

factor=0.5

patience=3



# kfold

nseed = 5

nfold = 7

eval_strategy = 'kfold'
# 模型结构

class Model(nn.Module):

    def __init__(self, n_features, n_targets, hidden_size=512, dropratio=0.2):

        super(Model, self).__init__()

        self.batch_norm1 = nn.BatchNorm1d(n_features)

        self.dropout1 = nn.Dropout(dropratio)

        self.dense1 = nn.utils.weight_norm(nn.Linear(n_features, hidden_size))

        

        self.batch_norm2 = nn.BatchNorm1d(hidden_size)

        self.dropout2 = nn.Dropout(dropratio)

        self.dense2 = nn.utils.weight_norm(nn.Linear(hidden_size, hidden_size))

        

        self.batch_norm3 = nn.BatchNorm1d(hidden_size)

        self.dropout3 = nn.Dropout(dropratio)

        self.dense3 = nn.utils.weight_norm(nn.Linear(hidden_size, n_targets))

        

        self.relu = nn.ReLU()

        

    def forward(self, x):

        x = self.batch_norm1(x)

        x = self.dropout1(x)

        x = self.relu(self.dense1(x))

        

        x = self.batch_norm2(x)

        x = self.dropout2(x)

        x = self.relu(self.dense2(x))

        

        x = self.batch_norm3(x)

        x = self.dropout3(x)

        x = self.dense3(x)

        

        return x
# label smmothing

class SmoothCrossEntropyLoss(_WeightedLoss):

    def __init__(self, weight=None, reduction='mean', smoothing=0.0):

        super().__init__(weight=weight, reduction=reduction)

        self.smoothing = smoothing

        self.weight = weight

        self.reduction = reduction



    @staticmethod

    def _smooth(targets, n_classes, smoothing=0.0):

        assert 0 <= smoothing <= 1

        with torch.no_grad():

#             targets = targets * (1.0 - smoothing) + 0.5 * smoothing

            targets = targets * (1 - smoothing) + torch.ones_like(targets).to(device) * smoothing / n_classes

        return targets



    def forward(self, inputs, targets):

        targets = SmoothCrossEntropyLoss()._smooth(targets, inputs.shape[1], self.smoothing)



        if self.weight is not None:

            inputs = inputs * self.weight.unsqueeze(0)



        loss = F.binary_cross_entropy_with_logits(inputs, targets)



        return loss
def running_train(X_train, Y_train, X_val, Y_val, dataloader, i_fold=None, seed=None):

    # prepare for train

    model = Model(n_features=X_train.shape[1], n_targets=Y_train.shape[1], hidden_size=hsize, dropratio=dropratio).to(device)

    criterion = SmoothCrossEntropyLoss(smoothing=smoothing)

    metric = lambda inputs, targets : F.binary_cross_entropy((torch.clamp(torch.sigmoid(inputs), p_min, p_max)), targets)

    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr, weight_decay=wd)

    if lr_scheduler == 'OneCycleLR' or lr_scheduler == 'both':

        scheduler1 = torch.optim.lr_scheduler.OneCycleLR(optimizer=optimizer, pct_start=pct_start, div_factor=div_factor, 

                                                    max_lr=1e-2, epochs=nepoch, steps_per_epoch=len(dataloader))

    if lr_scheduler == 'ReduceLROnPlateau' or lr_scheduler == 'both':

        scheduler2 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=0.5, patience=3)

    # train

    min_valmetric = np.inf

    step = 0

    for epoch in range(nepoch):

        train_loss = 0

        train_metric = 0

        for i, (X, Y) in enumerate(dataloader):

            model.train()

            predictions = model(X.to(device=device))

            loss = criterion(predictions, Y.to(device=device))

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            if lr_scheduler == 'OneCycleLR' or lr_scheduler == 'both':

                scheduler1.step()



            train_loss += loss.item()

            train_metric += metric(predictions, Y.to(device=device))



        train_loss /= len(dataloader)

        train_metric /= len(dataloader)

        model.eval()

        predictions = model(X_val.to(device=device))



        val_loss = criterion(predictions, Y_val.to(device=device))

        val_metric = metric(predictions, Y_val.to(device=device))

        if lr_scheduler == 'ReduceLROnPlateau' or lr_scheduler == 'both':

            scheduler2.step(val_metric)

        print('Epoch {}/{}, Train Loss={:5f}, Train Metric={:.5f}, Val Loss={:.5f}, Val Metric={:.5f}'.format(

            epoch + 1, nepoch, train_loss, train_metric, val_loss, val_metric))

        if val_metric.item() < min_valmetric:

            min_valmetric = val_metric.item()

            model_name = 'model_{}_{}.pth'.format(i_fold + 1, seed) if eval_strategy == 'kfold' else 'model_single.pth'

            torch.save(model.state_dict(), model_name)

        elif earlystop:

            step += 1

            if step > earlystop_step:

                break
# kfold

def train_kfold_model(train_df):

    X_train_val = torch.from_numpy(train_df.iloc[:, :-targets.shape[1]].values).to(dtype=torch.float32)

    Y_train_val = torch.from_numpy(train_df.iloc[:, -targets.shape[1]:].values).to(dtype=torch.float32)

    X_test = torch.from_numpy(test_df.values).to(dtype=torch.float32)



    oof = torch.zeros(X_train_val.shape[0], Y_train_val.shape[1]) # 用于计算 cv_score

    prediction_test = torch.zeros(X_test.shape[0], Y_train_val.shape[1]) # 用于计算 submission



    for i_seed in range(nseed):

        seed = random.randint(0, base_seed)

        seed_everything(seed)

        print('Seed: {}, {}/{}'.format(seed, i_seed + 1, nseed))

        mskf = MultilabelStratifiedKFold(n_splits=nfold, random_state=seed, shuffle=True)

        for i_fold, (train_idx, val_idx) in enumerate(mskf.split(X_train_val, Y_train_val)):

            print("# Fold: {}/{} (seed: {}/{})".format(i_fold + 1, nfold, i_seed + 1, nseed))



            # dataset and dataloader

            X_train, Y_train = X_train_val[train_idx], Y_train_val[train_idx]

            X_val, Y_val = X_train_val[val_idx], Y_train_val[val_idx]

            dataset = TensorDataset(X_train, Y_train)

            dataloader = DataLoader(dataset, batch_size=batchsize, shuffle=True)



            # train

            running_train(X_train, Y_train, X_val, Y_val, dataloader, i_fold=i_fold, seed=seed)



            # predict on oof

            print('predict on oof...', end='')

            model = Model(n_features=X_train.shape[1], n_targets=Y_train.shape[1], hidden_size=hsize, dropratio=dropratio).to(device)

            model.load_state_dict(torch.load('model_{}_{}.pth'.format(i_fold + 1, seed)))

            model.eval()

            oof[val_idx] += torch.clamp(torch.sigmoid(model(X_val.to(device=device)).detach().cpu()), p_min, p_max) / nseed

            print('  done.')

            # predict on test

            print('predict on test...', end='')

            prediction_test += torch.clamp(torch.sigmoid(model(X_test.to(device=device)).detach().cpu()), p_min, p_max) / (nfold * nseed)

            print('  done.\n')



    cv_score = F.binary_cross_entropy(oof, Y_train_val)

    print('{} folds cv_score: {:.5f}'.format(nfold, cv_score))

    

    return oof, pd.DataFrame(prediction_test.numpy(), columns=targets.columns)
# single model

def train_single_model(train_df):

    # obtain train, val, test

    train_df, val_df = train_test_split(train_df, test_size=0.2, shuffle=True, random_state=base_seed)



    X_train = train_df.iloc[:, :-targets.shape[1]].values

    Y_train = train_df.iloc[:, -targets.shape[1]:].values

    X_val = val_df.iloc[:, :-targets.shape[1]].values

    Y_val = val_df.iloc[:, -targets.shape[1]:].values

    X_test = test_df.values



    # dataset 和 dataloader

    X_train = torch.from_numpy(X_train).to(dtype=torch.float32)

    Y_train = torch.from_numpy(Y_train).to(dtype=torch.float32)

    X_val = torch.from_numpy(X_val).to(dtype=torch.float32)

    Y_val = torch.from_numpy(Y_val).to(dtype=torch.float32)

    X_test = torch.from_numpy(X_test.astype(np.float32))

    dataset = TensorDataset(X_train, Y_train)

    dataloader = DataLoader(dataset, batch_size=batchsize, shuffle=True)

    

    # train

    running_train(X_train, Y_train, X_val, Y_val, dataloader)



    # predict on test

    print('predict on test...', end='')

    model = Model(n_features=X_train.shape[1], n_targets=Y_train.shape[1], hidden_size=hsize, dropratio=dropratio).to(device)

    model.load_state_dict(torch.load('model_single.pth'))

    model.eval()

    predictions = torch.clamp(torch.sigmoid(model(X_test.to(device)).detach().cpu()), p_min, p_max)

    print('  done.\n')

    

    return pd.DataFrame(predictions.numpy(), columns=targets.columns)
if eval_strategy == 'kfold':

    oof, test_pred = train_kfold_model(train_df)

elif eval_strategy == 'single':

    test_pred = train_single_model(train_df)

else:

    print('eval_strategy should be \"kfold\" or \"single\"')
# submit

test = pd.read_csv(data_path + 'test_features.csv')

sig_id = test[test['cp_type']!='ctl_vehicle'].sig_id.reset_index(drop=True)

test_pred['sig_id'] = sig_id



sub = pd.merge(test[['sig_id']], test_pred, on='sig_id', how='left')

sub.fillna(0, inplace=True)

sub.to_csv('submission.csv', index=False)