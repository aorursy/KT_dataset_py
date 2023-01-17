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
## Install iterativestrat

import sys

sys.path.append('/kaggle/input/iterativestrat/iterative-stratification-master')

from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
## Install PYTORCH Lightning



!pip install ../input/pytorch-lightning/pytorch_lightning-0.8.5-py3-none-any.whl

import os

from glob import glob

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import joblib

import torch

import torch.nn as nn

import random

from torch.nn import functional as F

from tqdm.auto import tqdm

from sklearn.metrics import roc_auc_score

import pytorch_lightning as pl

from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
def seed_everything(seed):

    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = True



seed_everything(123)
data_dir = '/kaggle/input/lish-moa/'

from glob import glob

glob(data_dir + "*.csv")
train_feats = pd.read_csv(data_dir + "train_features.csv")

test_feats = pd.read_csv(data_dir + "test_features.csv")

train_targets_sc = pd.read_csv(data_dir + "train_targets_scored.csv")

sample_submission = pd.read_csv(data_dir + "sample_submission.csv")
fold = train_targets_sc.copy()

target_cols = [c for c in train_targets_sc.columns if c not in ['sig_id']]

mls = MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=234)

for n, (train_index, val_index) in enumerate(mls.split(fold, fold[target_cols])):

    fold.loc[val_index, 'fold'] = int(n)

fold['fold'] = fold['fold'].astype(int)
target_cols = [c for c in train_targets_sc.columns if c not in ["sig_id", 'fold']]

cat_cols = ['cp_type', 'cp_time', 'cp_dose']

numerical_cols = [c for c in train_feats.columns if train_feats.dtypes[c] != 'object']

numerical_cols = [c for c in numerical_cols if c not in cat_cols]
## SOURCE : https://www.kaggle.com/yasufuminakama/moa-pytorch-nn-starter

def cate2num(df):

    df['cp_type'] = df['cp_type'].map({'trt_cp': 0, 'ctl_vehicle': 1})

    df['cp_time'] = df['cp_time'].map({24: 2, 48: 3, 72: 4})

    df['cp_dose'] = df['cp_dose'].map({'D1': 5, 'D2': 6})

    return df



train_feats = cate2num(train_feats)

test_feats = cate2num(test_feats)
train_df = pd.merge(train_feats, fold, on = 'sig_id', how = 'inner')
test_df = test_feats
class DataSet(object):

  def __init__(self, folds, fold_df, num_features, cat_features, debugging = False):

    """

    folds = list of fold numbers to load data from

    """

    

    df = fold_df[fold_df.fold.isin(folds)]



    if debugging:

      _, df = train_test_split(df , stratify = df['fold'], test_size = 0.05, random_state = 10)



    self.uids = list(df['sig_id'].values)

    self.target = df[target_cols].values

    self.num_features = df[numerical_cols].values

    self.cat_features = df[cat_cols].values



  def __len__(self):

    return len(self.uids)

  

  def __getitem__(self, idx):



    num = torch.FloatTensor(self.num_features[idx])

    cat = torch.LongTensor(self.cat_features[idx])

    target = torch.tensor(self.target[idx]).float()



    return {

        'num' : num, 

        'target' : target,

        'cat' : cat

    } 



class TestDataSet(object):

  def __init__(self, df, num_features, cat_features, debugging = False):

    """

    folds = list of fold numbers to load data from

    """

    self.uids = list(df['sig_id'].values)

    self.num_features = df[numerical_cols].values

    self.cat_features = df[cat_cols].values

  def __len__(self):

    return len(self.uids)



  def __getitem__(self, idx):



    num = torch.FloatTensor(self.num_features[idx])

    cat = torch.LongTensor(self.cat_features[idx])

    return {

        'num' : num, 

        'cat' : cat,

        'uid' : self.uids[idx]

    } 
DEVICE = 'cuda'

BS = 128
def loss_fn(out, target):

  loss = nn.BCEWithLogitsLoss()(out, target)

  return loss

class litModel(pl.LightningModule):

    def __init__(self, fold_number, max_grad_norm=1000, gradient_accumulation_steps=1, hidden_size=512, dropout=0.5, lr=1e-2,

batch_size=128, epochs=20, total_cate_size=7, emb_size=4, num_features = numerical_cols, cat_features = cat_cols):

        ## Model Architecture Source : https://www.kaggle.com/yasufuminakama/moa-pytorch-nn-starter

        super(litModel, self).__init__()

        self.fold_number = fold_number

        self.cate_emb = nn.Embedding(total_cate_size, emb_size, padding_idx=0)

        self.emb_drop = nn.Dropout(0.2)

        self.cont_emb = nn.Sequential(

                          nn.Linear(len(num_features), hidden_size),

                          nn.BatchNorm1d(hidden_size),

                          nn.Dropout(dropout),

                          nn.PReLU(),

                          nn.Linear(hidden_size, hidden_size),

                          nn.BatchNorm1d(hidden_size),

                          nn.Dropout(dropout),

                          )

        head_hidden_size = hidden_size + len(cat_features)*emb_size

        self.head = nn.Sequential(

                          nn.Linear(head_hidden_size, head_hidden_size),

                          nn.BatchNorm1d(head_hidden_size),

                          nn.Dropout(0.1),

                          nn.Linear(head_hidden_size, len(target_cols)),

                          )



    def _forward_impl(self, cont_x, cate_x):

        # See note [TorchScript super()]

        batch_size = cate_x.size(0)

        cate_emb = self.cate_emb(cate_x).view(batch_size, -1)

        cate_emb = self.emb_drop(cate_emb)

        cont_emb = self.cont_emb(cont_x)

        x = torch.cat((cont_emb, cate_emb), 1)

        x = self.head(x)

        return x



    def forward(self, cont_x, cat_x):

        return self._forward_impl(cont_x, cat_x)



    def prepare_data(self):

        train_fold = [x for x in range(5) if x!= self.fold_number]

        test_fold = [self.fold_number]

        print("TRAINING FOLD : ", train_fold)

        print("TESTING FOLD : ", test_fold )

        self.train_data_set = DataSet(train_fold, train_df, numerical_cols, cat_cols, False)

        self.valid_data_set = DataSet(test_fold, train_df, numerical_cols, cat_cols, False)



    def train_dataloader(self):

        return torch.utils.data.DataLoader(self.train_data_set, BS, True, num_workers= 4)

    

    def val_dataloader(self):

        return torch.utils.data.DataLoader(self.valid_data_set, BS, False, num_workers= 4)



    def configure_optimizers(self):

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(

            optimizer,

            T_max = 100

        )

        return [optimizer]

  

    def training_step(self, batch, batch_idx):

        num_x = batch['num']

        cat_x = batch['cat']

        target = batch['target']

        output = self.forward(num_x, cat_x)

        loss = loss_fn(output, target)

        return {'loss' : loss, 'log' : {'loss' : loss}}

    

    def validation_step(self, batch, batch_idx):

        num_x = batch['num']

        cat_x = batch['cat']

        target = batch['target']

        output = self.forward(num_x, cat_x)

        loss = loss_fn(output, target)

        return {'val_loss' : loss, 'y': target.detach(), 'y_hat': output.detach(), 'log' : {'loss' : loss}}



    def validation_epoch_end(self, outputs):

        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()

        y = torch.cat([x['y'] for x in outputs])

        y_hat = torch.cat([x['y_hat'] for x in outputs])

        acc = (y_hat.round() == y).float().mean().item()

        print(f"Epoch {self.current_epoch} acc:{acc} loss:{avg_loss}")

        tensorboard_logs = {'val_loss': avg_loss, 'val_acc': acc}

        return {'avg_val_loss': avg_loss, 'val_acc': acc, 'log' : {'val_loss': avg_loss}}
model = litModel(fold_number = 2)  

model.prepare_data()

early_stop_callback = pl.callbacks.EarlyStopping(

  monitor='val_loss',

  min_delta=0.00,

  patience=3,

  verbose=False,

  mode='min'

)



checkpoint_callback = pl.callbacks.ModelCheckpoint("fold_2/{epoch:02d}_{val_loss:.4f}",

                                                  save_top_k=1, monitor='val_loss', mode='min')



trainer = pl.Trainer(max_epochs=40,gpus = 1, early_stop_callback=early_stop_callback, checkpoint_callback=checkpoint_callback, progress_bar_refresh_rate=0)



trainer.fit(model)
PATH = glob("/kaggle/working/fold_2/*.ckpt")[0]
m = trainer.model.load_from_checkpoint(checkpoint_path=PATH)

test_data = TestDataSet(test_df, numerical_cols, cat_cols)

test_loader = torch.utils.data.DataLoader(test_data, BS, False, num_workers= 4,  drop_last=False, pin_memory=False)
def inference_fn(test_loader, model, device):



    model.eval()

    preds = []

    model.to(device)

    out_df = []

    for step, obj in enumerate(test_loader):



        cate_x = obj['cat']

        cont_x = obj['num']

        uid = obj['uid']

        cont_x,  cate_x = cont_x.to(device), cate_x.to(device)



        with torch.no_grad():

            pred = model(cont_x, cate_x)





        p = pred.sigmoid().detach().cpu().numpy()

        preds.append(p)

        temp = pd.DataFrame(columns=['sig_id'] + target_cols)

        temp["sig_id"] = uid

        temp.loc[:, target_cols] = p



        out_df.append(temp)

    preds = np.concatenate(preds)



    return preds, out_df
preds, out_df = inference_fn(test_loader, m, 'cuda')

out_df = pd.concat(out_df)

out_df.to_csv("submission.csv", index = False)