!pip install ../input/pytorchlightning/tensorboard-2.2.0-py3-none-any.whl

!pip install ../input/pytorchlightning/pytorch_lightning-0.9.0-py3-none-any.whl

import sys

sys.path.append('../input/iterative-stratification/iterative-stratification-master')
import os

import gc

import glob

import random

import itertools

import numpy as np

import pandas as pd



from sklearn.model_selection import KFold

from sklearn.decomposition import PCA

from iterstrat.ml_stratifiers import MultilabelStratifiedKFold



import torch

from torch import nn, optim

from torch.utils.data import Dataset, DataLoader

from torch.utils.data.sampler import RandomSampler, SequentialSampler

from torch.optim.lr_scheduler import CosineAnnealingLR



import pytorch_lightning as pl

from pytorch_lightning import Trainer

from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping



import warnings

warnings.filterwarnings('ignore')
pl.__version__
class cfg:

    seed = 42

    g_comp = 29

    c_comp = 4

    fold = 0

    emb_dims = [(2, 15), (3, 20), (2, 15)]

    

    dropout_rate = 0.4

    hidden_size = 2048

    

    batch_size = 128

    lr = 0.001

    epoch = 15
def seed_everything(seed):

    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = True
seed_everything(cfg.seed)
class MoADataset(Dataset):

    def __init__(self, df, feature_cols, target_cols, phase='train'):

        self.df = df

        self.cat_cols = ['cp_type', 'cp_time', 'cp_dose']

        self.cont_cols = [c for c in feature_cols if c not in self.cat_cols]

        self.target_cols = target_cols

        self.phase = phase

        

        self.cont_features = self.df[self.cont_cols].values

        self.cat_features = self.df[self.cat_cols].values

        self.targets = self.df[self.target_cols].values



    def __len__(self):

        return len(self.df)



    def __getitem__(self, idx):

        cont_f = torch.tensor(self.cont_features[idx, :], dtype=torch.float)

        cat_f = torch.tensor(self.cat_features[idx, :], dtype=torch.long)



        if self.phase != 'test':

            target = torch.tensor(self.targets[idx, :], dtype=torch.float)



            return cont_f, cat_f, target



        else:

            sig_id = self.df['sig_id'].iloc[idx]

            return cont_f, cat_f, sig_id
class DataModule(pl.LightningDataModule):

    def __init__(self, data_dir, cfg, cv):

        super(DataModule, self).__init__()

        self.cfg = cfg

        self.data_dir = data_dir

        self.cv = cv



    def prepare_data(self):

        # Prepare Data

        train_target = pd.read_csv(os.path.join(self.data_dir, 'train_targets_scored.csv'))

        train_feature = pd.read_csv(os.path.join(self.data_dir, 'train_features.csv'))

        test = pd.read_csv(os.path.join(self.data_dir, 'test_features.csv'))



        train = pd.merge(train_target, train_feature, on='sig_id')

        self.target_cols = [c for c in train_target.columns if c != 'sig_id']



        test['is_train'] = 0

        train['is_train'] = 1

        self.df = pd.concat([train, test], axis=0, ignore_index=True)

        self.test_id = test['sig_id'].values

        

        # Label Encoding

        self.df = self.Encode(self.df)

        # add PCA Features

        self.df = self.add_PCA(self.df, g_comp=self.cfg.g_comp, c_comp=self.cfg.c_comp)

        self.feature_cols = [c for c in self.df.columns if c not in self.target_cols + ['sig_id', 'is_train', 'fold']]



        del train, train_target, train_feature, test

        gc.collect()



    def setup(self, stage=None):

        # Split Train, Test

        df = self.df[self.df['is_train'] == 1].reset_index(drop=True)

        test = self.df[self.df['is_train'] == 0].reset_index(drop=True)

        self.test_id = test['sig_id'].values



        # Split Train, Validation

        df['fold'] = -1

        for i, (trn_idx, val_idx) in enumerate(self.cv.split(df, df[self.target_cols])):

            df.loc[val_idx, 'fold'] = i

        fold = self.cfg.fold

        train = df[df['fold'] != fold].reset_index(drop=True)

        val = df[df['fold'] == fold].reset_index(drop=True)



        self.train_dataset = MoADataset(train, self.feature_cols, self.target_cols, phase='train')

        self.val_dataset = MoADataset(val, self.feature_cols, self.target_cols, phase='train')

        self.test_dataset = MoADataset(test, self.feature_cols, self.target_cols, phase='test')



        del df, test, train, val

        gc.collect()



    def train_dataloader(self):

        return DataLoader(self.train_dataset,

                          batch_size=self.cfg.batch_size,

                          pin_memory=True,

                          sampler=RandomSampler(self.train_dataset), drop_last=False)



    def val_dataloader(self):

        return DataLoader(self.val_dataset,

                          batch_size=self.cfg.batch_size,

                          pin_memory=True,

                          sampler=SequentialSampler(self.val_dataset), drop_last=False)



    def test_dataloader(self):

        return DataLoader(self.test_dataset,

                          batch_size=self.cfg.batch_size,

                          pin_memory=False,

                          shuffle=False, drop_last=False)

    

    def Encode(self, df):

        cp_type_encoder = {

            'trt_cp': 0,

            'ctl_vehicle': 1

        }



        cp_time_encoder = {

            24: 0,

            48: 1,

            72: 2,

        }



        cp_dose_encoder = {

            'D1': 0,

            'D2': 1

        }



        df['cp_type'] = df['cp_type'].map(cp_type_encoder)

        df['cp_time'] = df['cp_time'].map(cp_time_encoder)

        df['cp_dose'] = df['cp_dose'].map(cp_dose_encoder)



        for c in ['cp_type', 'cp_time', 'cp_dose']:

            df[c] = df[c].astype(int)



        return df

    

    

    def add_PCA(self, df, g_comp=29, c_comp=4):

        # g-features

        g_cols = [c for c in df.columns if 'g-' in c]

        temp = PCA(n_components=g_comp, random_state=self.cfg.seed).fit_transform(df[g_cols])

        temp = pd.DataFrame(temp, columns=[f'g-pca_{i}' for i in range(g_comp)])

        df = pd.concat([df, temp], axis=1)



        # c-features

        c_cols = [c for c in df.columns if 'c-' in c]

        temp = PCA(n_components=c_comp, random_state=self.cfg.seed).fit_transform(df[c_cols])

        temp = pd.DataFrame(temp, columns=[f'c-pca_{i}' for i in range(c_comp)])

        df = pd.concat([df, temp], axis=1)



        del temp



        return df
class LightningSystem(pl.LightningModule):

    def __init__(self, net, cfg, target_cols):

        super(LightningSystem, self).__init__()

        self.net = net

        self.cfg = cfg

        self.target_cols = target_cols

        self.criterion = nn.BCEWithLogitsLoss()

        self.best_loss = 1e+9



    def configure_optimizers(self):

        self.optimizer = optim.AdamW(self.parameters(), lr=self.cfg.lr, weight_decay=2e-5)

        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=self.cfg.epoch, eta_min=0)



        return [self.optimizer], [self.scheduler]



    def forward(self, cont_f, cat_f):

        return self.net(cont_f, cat_f)



    def step(self, batch):

        cont_f, cat_f, label = batch

        out = self.forward(cont_f, cat_f)

        loss = self.criterion(out, label)



        return loss, label



    def training_step(self, batch, batch_idx):

        loss, label = self.step(batch)

        logs = {'train/loss': loss.item()}



        return {'loss': loss, 'labels': label}



    def validation_step(self, batch, batch_idx):

        loss, label = self.step(batch)

        val_logs = {'val/loss': loss.item()}



        return {'val_loss': loss, 'labels': label.detach()}



    def validation_epoch_end(self, outputs):

        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()

        logs = {'val/epoch_loss': avg_loss.item()}



        return {'avg_val_loss': avg_loss}





    def test_step(self, batch, batch_idx):

        cont_f, cat_f, ids = batch

        out = self.forward(cont_f, cat_f)

        logits = torch.sigmoid(out)



        return {'pred': logits, 'id': ids}





    def test_epoch_end(self, outputs):

        preds = torch.cat([x['pred'] for x in outputs]).detach().cpu().numpy()

        res = pd.DataFrame(preds, columns=self.target_cols)



        ids = [x['id'] for x in outputs]

        ids = [list(x) for x in ids]

        ids = list(itertools.chain.from_iterable(ids))



        res.insert(0, 'sig_id', ids)



        res.to_csv('submission.csv', index=False)

        

        return {}
class LinearReluBnDropout(nn.Module):

    def __init__(self, in_features, out_features, dropout_rate):

        super(LinearReluBnDropout, self).__init__()



        self.block = nn.Sequential(

            nn.utils.weight_norm(nn.Linear(in_features, out_features)),

            nn.ReLU(inplace=True),

            nn.BatchNorm1d(out_features),

            nn.Dropout(dropout_rate)

        )



    def forward(self, x):

        x = self.block(x)



        return x





class TablarNet(nn.Module):

    def __init__(self, emb_dims, cfg, in_cont_features=875, out_features=206):

        super(TablarNet, self).__init__()



        self.embedding_layer = nn.ModuleList([nn.Embedding(x, y) for x, y in emb_dims])

        self.dropout = nn.Dropout(cfg.dropout_rate, inplace=True)



        self.first_bn_layer = nn.Sequential(

            nn.BatchNorm1d(in_cont_features),

            nn.Dropout(cfg.dropout_rate)

        )



        first_in_feature = in_cont_features + sum([y for x, y in emb_dims])



        self.block = nn.Sequential(

            LinearReluBnDropout(in_features=first_in_feature,

                                out_features=cfg.hidden_size,

                                dropout_rate=cfg.dropout_rate),

            LinearReluBnDropout(in_features=cfg.hidden_size,

                                out_features=cfg.hidden_size,

                                dropout_rate=cfg.dropout_rate)

        )



        self.last = nn.Linear(cfg.hidden_size, out_features)



    def forward(self, cont_f, cat_f):



        cat_x = [layer(cat_f[:, i]) for i, layer in enumerate(self.embedding_layer)]

        cat_x = torch.cat(cat_x, 1)

        cat_x = self.dropout(cat_x)



        cont_x = self.first_bn_layer(cont_f)



        x = torch.cat([cont_x, cat_x], 1)



        x = self.block(x)

        x = self.last(x)



        return x
def main():

    # Set data dir

    data_dir = '../input/lish-moa'

    # CV

    cv = MultilabelStratifiedKFold(n_splits=4)

    # Random Seed

    seed_everything(cfg.seed)



    # Lightning Data Module  ####################################################

    datamodule = DataModule(data_dir, cfg, cv)

    datamodule.prepare_data()

    target_cols = datamodule.target_cols

    feature_cols = datamodule.feature_cols



    # Model  ####################################################################

    # Adjust input dim (original + composition dim - category features)

    in_features = len(feature_cols) - 3

    net = TablarNet(cfg.emb_dims, cfg, in_cont_features=in_features)



    # Lightning Module  #########################################################

    model = LightningSystem(net, cfg, target_cols)



    checkpoint_callback = ModelCheckpoint(

        save_top_k=1,

        verbose=False,

        monitor='avg_val_loss',

        mode='min'

    )

    

    early_stop_callback = EarlyStopping(

        monitor='avg_val_loss',

        min_delta=0.00,

        patience=3,

        verbose=False,

        mode='min'

    )



    trainer = Trainer(

        logger=False,

        max_epochs=cfg.epoch,

        checkpoint_callback=checkpoint_callback,

        early_stop_callback=early_stop_callback,

#         gpus=1

            )



    # Train & Test  ############################################################

    # Train

    trainer.fit(model, datamodule=datamodule)



    # Test

    trainer.test(model, datamodule=datamodule)
main()