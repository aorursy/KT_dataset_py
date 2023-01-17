# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



# import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import sys

sys.path.append('../input/iterative-stratification/iterative-stratification-master')

from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
import numpy as np

import random

import pandas as pd

import matplotlib.pyplot as plt

import os

import copy

import seaborn as sns



from sklearn import preprocessing

from sklearn.metrics import log_loss

from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA



import torch

import torch.nn as nn

import torch.nn.functional as F

import torch.optim as optim



import warnings

warnings.filterwarnings('ignore')
train_features = pd.read_csv('../input/lish-moa/train_features.csv')

train_targets_scored = pd.read_csv('../input/lish-moa/train_targets_scored.csv')

train_targets_nonscored = pd.read_csv('../input/lish-moa/train_targets_nonscored.csv')



test_features = pd.read_csv('../input/lish-moa/test_features.csv')

sample_submission = pd.read_csv('../input/lish-moa/sample_submission.csv')


def seed_everything(seed=42):

    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.deterministic = True

    

class MoADataset:

    def __init__(self, features, targets):

        self.features = features

        self.targets = targets

        

    def __len__(self):

        return (self.features.shape[0])

    

    def __getitem__(self, idx):

        dct = {

            'x' : torch.tensor(self.features[idx, :], dtype=torch.float),

            'y' : torch.tensor(self.targets[idx, :], dtype=torch.float)            

        }

        return dct

    

class TestDataset:

    def __init__(self, features):

        self.features = features

        

    def __len__(self):

        return (self.features.shape[0])

    

    def __getitem__(self, idx):

        dct = {

            'x' : torch.tensor(self.features[idx, :], dtype=torch.float)

        }

        return dct

    



def inference_fn(model, dataloader, device):

    model.eval()

    preds = []

    

    for data in dataloader:

        inputs = data['x'].to(device)



        with torch.no_grad():

            outputs = model(inputs)

        

        preds.append(outputs.sigmoid().detach().cpu().numpy())

        

    preds = np.concatenate(preds)

    

    return preds

   

class Model(nn.Module):

    def __init__(self, num_features, num_targets, hidden_size):

        super(Model, self).__init__()

        self.batch_norm1 = nn.BatchNorm1d(num_features)

        self.dropout1 = nn.Dropout(0.2)

        self.dense1 = nn.utils.weight_norm(nn.Linear(num_features, hidden_size))

        

        self.batch_norm2 = nn.BatchNorm1d(hidden_size)

        self.dropout2 = nn.Dropout(0.2)

        self.dense2 = nn.utils.weight_norm(nn.Linear(hidden_size, hidden_size))

        

        self.batch_norm3 = nn.BatchNorm1d(hidden_size)

        self.dropout3 = nn.Dropout(0.25)

        self.dense3 = nn.utils.weight_norm(nn.Linear(hidden_size, num_targets))

    

    def forward(self, x):

        x = self.batch_norm1(x)

        x = self.dropout1(x)

        x = F.relu(self.dense1(x))

        

        x = self.batch_norm2(x)

        x = self.dropout2(x)

        x = F.relu(self.dense2(x))

        

        x = self.batch_norm3(x)

        x = self.dropout3(x)

        x = self.dense3(x)

        

        return x



def process_data(data):

    

    data = pd.get_dummies(data, columns=['cp_time','cp_dose'])

    

    return data



seed_everything(seed=42)
GENES = [col for col in train_features.columns if col.startswith('g-')]

CELLS = [col for col in train_features.columns if col.startswith('c-')]
# GENES

n_comp = 29



data = pd.concat([pd.DataFrame(train_features[GENES]), pd.DataFrame(test_features[GENES])])

data2 = (PCA(n_components=n_comp, random_state=42).fit_transform(data[GENES]))

train2 = data2[:train_features.shape[0]]; test2 = data2[-test_features.shape[0]:]



train2 = pd.DataFrame(train2, columns=[f'pca_G-{i}' for i in range(n_comp)])

test2 = pd.DataFrame(test2, columns=[f'pca_G-{i}' for i in range(n_comp)])



# drop_cols = [f'c-{i}' for i in range(n_comp,len(GENES))]

train_features = pd.concat((train_features, train2), axis=1)

test_features = pd.concat((test_features, test2), axis=1)



#CELLS

n_comp = 4



data = pd.concat([pd.DataFrame(train_features[CELLS]), pd.DataFrame(test_features[CELLS])])

data2 = (PCA(n_components=n_comp, random_state=42).fit_transform(data[CELLS]))

train2 = data2[:train_features.shape[0]]; test2 = data2[-test_features.shape[0]:]



train2 = pd.DataFrame(train2, columns=[f'pca_C-{i}' for i in range(n_comp)])

test2 = pd.DataFrame(test2, columns=[f'pca_C-{i}' for i in range(n_comp)])



# drop_cols = [f'c-{i}' for i in range(n_comp,len(CELLS))]

train_features = pd.concat((train_features, train2), axis=1)

test_features = pd.concat((test_features, test2), axis=1)
from sklearn.feature_selection import VarianceThreshold



var_thresh = VarianceThreshold(threshold=0.7)

data = train_features.append(test_features)

data_transformed = var_thresh.fit_transform(data.iloc[:, 4:])



train_features_transformed = data_transformed[ : train_features.shape[0]]

test_features_transformed = data_transformed[-test_features.shape[0] : ]





train_features = pd.DataFrame(train_features[['sig_id','cp_type','cp_time','cp_dose']].values.reshape(-1, 4),\

                              columns=['sig_id','cp_type','cp_time','cp_dose'])



train_features = pd.concat([train_features, pd.DataFrame(train_features_transformed)], axis=1)





test_features = pd.DataFrame(test_features[['sig_id','cp_type','cp_time','cp_dose']].values.reshape(-1, 4),\

                             columns=['sig_id','cp_type','cp_time','cp_dose'])



test_features = pd.concat([test_features, pd.DataFrame(test_features_transformed)], axis=1)



train_features
train = train_features.merge(train_targets_scored, on='sig_id')

train = train[train['cp_type']!='ctl_vehicle'].reset_index(drop=True)

test = test_features

target = train[train_targets_scored.columns]



train = train.drop('cp_type', axis=1)

test = test.drop('cp_type', axis=1)
target_cols = target.drop('sig_id', axis=1).columns.values.tolist()
folds = train.copy()



mskf = MultilabelStratifiedKFold(n_splits=3)



for f, (t_idx, v_idx) in enumerate(mskf.split(X=train, y=target)):

    folds.loc[v_idx, 'kfold'] = int(f)



folds['kfold'] = folds['kfold'].astype(int)
feature_cols = [c for c in process_data(folds).columns if c not in target_cols]

feature_cols = [c for c in feature_cols if c not in ['kfold','sig_id']]

len(feature_cols)
# HyperParameters



DEVICE = ('cuda' if torch.cuda.is_available() else 'cpu')

EPOCHS = 20

BATCH_SIZE = 64

LEARNING_RATE = 1e-3

WEIGHT_DECAY = 1e-5

NFOLDS = 3

EARLY_STOPPING_STEPS = 11

EARLY_STOP = True



num_features=len(feature_cols)

num_targets=len(target_cols)

hidden_size=1024
def run_inference(fold,seed):    

    test_ = process_data(test)

    x_test = test_[feature_cols].values

    testdataset = TestDataset(x_test)

    testloader = torch.utils.data.DataLoader(testdataset, batch_size=BATCH_SIZE, shuffle=False)

    

    model = Model(

        num_features=num_features,

        num_targets=num_targets,

        hidden_size=hidden_size,

    )

    

    model.load_state_dict(torch.load(f"../input/pytorch-moa-0-01867/FOLD{fold}_.pth"))

    model.to(DEVICE)

    

    predictions = np.zeros((len(test_), target.iloc[:, 1:].shape[1]))

    predictions = inference_fn(model, testloader, DEVICE)

    

    return predictions
def run_k_fold(NFOLDS, seed):

    predictions = np.zeros((len(test), len(target_cols)))

    

    for fold in range(NFOLDS):

        pred_ = run_inference(fold, seed)

        

        predictions += pred_ / NFOLDS

        

    return predictions
from tqdm.notebook import tqdm
SEED = [42]

predictions_1 = np.zeros((len(test), len(target_cols)))



for seed in tqdm(SEED):

    predictions_ = run_k_fold(NFOLDS, seed)

    predictions_1 += predictions_ / len(SEED)



test[target_cols] = predictions_1
# sub1 = sample_submission.drop(columns=target_cols).merge(test[['sig_id']+target_cols], on='sig_id', how='left').fillna(0)
from sklearn.preprocessing import QuantileTransformer
train_features = pd.read_csv('../input/lish-moa/train_features.csv')

train_targets_scored = pd.read_csv('../input/lish-moa/train_targets_scored.csv')

train_targets_nonscored = pd.read_csv('../input/lish-moa/train_targets_nonscored.csv')



test_features = pd.read_csv('../input/lish-moa/test_features.csv')

sample_submission = pd.read_csv('../input/lish-moa/sample_submission.csv')
#RankGauss



for col in (GENES + CELLS):



    transformer = QuantileTransformer(n_quantiles=100, random_state=0, output_distribution="normal")

    vec_len = len(train_features[col].values)

    vec_len_test = len(test_features[col].values)

    raw_vec = train_features[col].values.reshape(vec_len, 1)

    transformer.fit(raw_vec)



    train_features[col] = transformer.transform(raw_vec).reshape(1, vec_len)[0]

    test_features[col] = transformer.transform(test_features[col].values.reshape(vec_len_test, 1)).reshape(1, vec_len_test)[0]
def seed_everything(seed=42):

    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.deterministic = True

    

seed_everything(seed=42)
train_targets_scored.sum()[1:].sort_values()

train_features['cp_type'].unique()

# GENES

n_comp = 50



data = pd.concat([pd.DataFrame(train_features[GENES]), pd.DataFrame(test_features[GENES])])

data2 = (PCA(n_components=n_comp, random_state=42).fit_transform(data[GENES]))

train2 = data2[:train_features.shape[0]]; test2 = data2[-test_features.shape[0]:]



train2 = pd.DataFrame(train2, columns=[f'pca_G-{i}' for i in range(n_comp)])

test2 = pd.DataFrame(test2, columns=[f'pca_G-{i}' for i in range(n_comp)])



# drop_cols = [f'c-{i}' for i in range(n_comp,len(GENES))]

train_features = pd.concat((train_features, train2), axis=1)

test_features = pd.concat((test_features, test2), axis=1)
#CELLS

n_comp = 15



data = pd.concat([pd.DataFrame(train_features[CELLS]), pd.DataFrame(test_features[CELLS])])

data2 = (PCA(n_components=n_comp, random_state=42).fit_transform(data[CELLS]))

train2 = data2[:train_features.shape[0]]; test2 = data2[-test_features.shape[0]:]



train2 = pd.DataFrame(train2, columns=[f'pca_C-{i}' for i in range(n_comp)])

test2 = pd.DataFrame(test2, columns=[f'pca_C-{i}' for i in range(n_comp)])



# drop_cols = [f'c-{i}' for i in range(n_comp,len(CELLS))]

train_features = pd.concat((train_features, train2), axis=1)

test_features = pd.concat((test_features, test2), axis=1)
from sklearn.feature_selection import VarianceThreshold





var_thresh = VarianceThreshold(threshold=0.5)

data = train_features.append(test_features)

data_transformed = var_thresh.fit_transform(data.iloc[:, 4:])



train_features_transformed = data_transformed[ : train_features.shape[0]]

test_features_transformed = data_transformed[-test_features.shape[0] : ]





train_features = pd.DataFrame(train_features[['sig_id','cp_type','cp_time','cp_dose']].values.reshape(-1, 4),\

                              columns=['sig_id','cp_type','cp_time','cp_dose'])



train_features = pd.concat([train_features, pd.DataFrame(train_features_transformed)], axis=1)





test_features = pd.DataFrame(test_features[['sig_id','cp_type','cp_time','cp_dose']].values.reshape(-1, 4),\

                             columns=['sig_id','cp_type','cp_time','cp_dose'])



test_features = pd.concat([test_features, pd.DataFrame(test_features_transformed)], axis=1)



train_features
train = train_features.merge(train_targets_scored, on='sig_id')

train = train[train['cp_type']!='ctl_vehicle'].reset_index(drop=True)

test = test_features

target = train[train_targets_scored.columns]
train = train.drop('cp_type', axis=1)

test = test.drop('cp_type', axis=1)
target_cols = target.drop('sig_id', axis=1).columns.values.tolist()

folds = train.copy()



mskf = MultilabelStratifiedKFold(n_splits=5)



for f, (t_idx, v_idx) in enumerate(mskf.split(X=train, y=target)):

    folds.loc[v_idx, 'kfold'] = int(f)



folds['kfold'] = folds['kfold'].astype(int)

folds
class MoADataset:

    def __init__(self, features, targets):

        self.features = features

        self.targets = targets

        

    def __len__(self):

        return (self.features.shape[0])

    

    def __getitem__(self, idx):

        dct = {

            'x' : torch.tensor(self.features[idx, :], dtype=torch.float),

            'y' : torch.tensor(self.targets[idx, :], dtype=torch.float)            

        }

        return dct

    

class TestDataset:

    def __init__(self, features):

        self.features = features

        

    def __len__(self):

        return (self.features.shape[0])

    

    def __getitem__(self, idx):

        dct = {

            'x' : torch.tensor(self.features[idx, :], dtype=torch.float)

        }

        return dct
def inference_fn(model, dataloader, device):

    model.eval()

    preds = []

    

    for data in dataloader:

        inputs = data['x'].to(device)



        with torch.no_grad():

            outputs = model(inputs)

        

        preds.append(outputs.sigmoid().detach().cpu().numpy())

        

    preds = np.concatenate(preds)

    

    return preds
class Model(nn.Module):

    def __init__(self, num_features, num_targets, hidden_size):

        super(Model, self).__init__()

        self.batch_norm1 = nn.BatchNorm1d(num_features)

        self.dropout1 = nn.Dropout(0.2)

        self.dense1 = nn.utils.weight_norm(nn.Linear(num_features, hidden_size))

        

        self.batch_norm2 = nn.BatchNorm1d(hidden_size)

        self.dropout2 = nn.Dropout(0.5)

        self.dense2 = nn.utils.weight_norm(nn.Linear(hidden_size, hidden_size))

        

        self.batch_norm3 = nn.BatchNorm1d(hidden_size)

        self.dropout3 = nn.Dropout(0.5)

        self.dense3 = nn.utils.weight_norm(nn.Linear(hidden_size, num_targets))

    

    def forward(self, x):

        x = self.batch_norm1(x)

        x = self.dropout1(x)

        x = F.relu(self.dense1(x))

        

        x = self.batch_norm2(x)

        x = self.dropout2(x)

        x = F.relu(self.dense2(x))

        

        x = self.batch_norm3(x)

        x = self.dropout3(x)

        x = self.dense3(x)

        

        return x
feature_cols = [c for c in process_data(folds).columns if c not in target_cols]

feature_cols = [c for c in feature_cols if c not in ['kfold','sig_id']]

len(feature_cols)
# HyperParameters



DEVICE = ('cuda' if torch.cuda.is_available() else 'cpu')

EPOCHS = 25

BATCH_SIZE = 128

LEARNING_RATE = 1e-3

WEIGHT_DECAY = 1e-5

NFOLDS = 5

EARLY_STOPPING_STEPS = 10

EARLY_STOP = False



num_features=len(feature_cols)

num_targets=len(target_cols)

hidden_size=1024
def run_inference(fold,seed):    

    test_ = process_data(test)

    x_test = test_[feature_cols].values

    testdataset = TestDataset(x_test)

    testloader = torch.utils.data.DataLoader(testdataset, batch_size=BATCH_SIZE, shuffle=False)

    

    model = Model(

        num_features=num_features,

        num_targets=num_targets,

        hidden_size=hidden_size,

    )

    

    model.load_state_dict(torch.load(f"../input/moa-pytorch-nn-pca-rankgauss/FOLD{fold}_.pth"))

    model.to(DEVICE)

    

    predictions = np.zeros((len(test_), target.iloc[:, 1:].shape[1]))

    predictions = inference_fn(model, testloader, DEVICE)

    

    return predictions
def run_k_fold(NFOLDS, seed):

    predictions = np.zeros((len(test), len(target_cols)))

    

    for fold in range(NFOLDS):

        pred_ = run_inference(fold, seed)

        

        predictions += pred_ / NFOLDS

        

    return predictions
# Averaging on multiple SEEDS



SEED = [0, 1, 2, 3 ,4, 5]

predictions_2 = np.zeros((len(test), len(target_cols)))



for seed in tqdm(SEED):

    

    predictions_ = run_k_fold(NFOLDS, seed)

    predictions_2 += predictions_ / len(SEED)



test[target_cols] = predictions_2
# sub2 = sample_submission.drop(columns=target_cols).merge(test[['sig_id']+target_cols], on='sig_id', how='left').fillna(0)
prediction = 0.8*predictions_1 + 0.2*predictions_2

sub = pd.read_csv('/kaggle/input/lish-moa/sample_submission.csv')

sub.iloc[:,1:] = prediction
# sub.loc[test_features[test_features.cp_type=='ctl_vehicle'].sig_id]= 0

sub.to_csv('submission.csv', index=False)
sub.head()