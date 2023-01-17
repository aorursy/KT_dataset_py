!pip install -qq --upgrade pytorch-lightning --use-feature=2020-resolver

!pip install -qq torchsummary --use-feature=2020-resolver

!pip install -qq iterative-stratification --use-feature=2020-resolver
import os

import pytorch_lightning as pl

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import torch

from torch.utils.data import Dataset, DataLoader

import torch.nn.functional as F

from torch import nn

from torch import optim

from sklearn.model_selection import train_test_split

import random

import warnings

from torchsummary import summary

import os

from collections import OrderedDict

from pytorch_lightning.metrics.classification import Accuracy

from pytorch_lightning.metrics.sklearns import F1

from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

from pytorch_lightning.loggers import TensorBoardLogger

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

warnings.filterwarnings('ignore')
#Config

BATCH_SIZE = 256

LEARNING_RATE = 3e-4

NUM_WORKERS = 4

EPOCHS = 50

N_GPUS = 1 if torch.cuda.is_available() else 0
train_features = pd.read_csv("/kaggle/input/lish-moa/train_features.csv")

train_targets = pd.read_csv("/kaggle/input/lish-moa/train_targets_scored.csv")

test_features = pd.read_csv("/kaggle/input/lish-moa/test_features.csv")

sample_submission = pd.read_csv("/kaggle/input/lish-moa/sample_submission.csv")

print(f"Train Dataset Size : {train_features.shape}")

print(f"Test Dataset Size : {test_features.shape}")

train_features.head()
feature_names = train_features.drop(columns = ['sig_id']).columns

feature_genes = [col for col in feature_names if col.startswith('g-')]

feature_cells = [col for col in feature_names if col.startswith('c-')]

feature_categ = [col for col in feature_names 

                       if col not in feature_cells and col not in feature_genes]

print(f"total Features : {len(feature_names)}")

print(f"Gene Features : {len(feature_genes)}")

print(f"Cell Features : {len(feature_cells)}")

print(f"Categorical Features : {len(feature_categ)}")
target_cols = train_targets.drop(columns = "sig_id").columns

train_targets[target_cols].head()
train_targets[target_cols].sum().sort_values()
single_val_labels = target_cols[train_targets[target_cols].sum() <= 1].tolist()

target_cols = target_cols[train_targets[target_cols].sum() > 1].tolist()

train_targets = train_targets[target_cols + ['sig_id']]

len(single_val_labels), len(target_cols)
train_features.shape, train_targets.shape
train_data = train_features.merge(train_targets, on = "sig_id", how = "inner")

print(f"Train Dataset Size : {train_data.shape}")

train_data.head()
def preprocess_data(data, gene_features, cell_features, categorical_features) : 

    

    new_data = pd.get_dummies(data, columns = categorical_features)

    

    return new_data
train_data = preprocess_data(train_data, 

                              feature_genes,

                              feature_cells, 

                              feature_categ)



test_data = preprocess_data(test_features, 

                           feature_genes,

                           feature_cells, 

                           feature_categ, 

                           )

train_data.head()
total_folds = train_data.copy()



mskf = MultilabelStratifiedKFold(n_splits=5, shuffle = True, random_state = 1997)



for f, (train_idx, val_idx) in enumerate(mskf.split(X=train_data, 

                                                    y=train_targets.drop(columns = ['sig_id']))):

    total_folds.loc[val_idx, 'kfold'] = int(f)



total_folds['kfold'] = total_folds['kfold'].astype(int)

total_folds
ignore_features = ['sig_id', 'kfold']

categorical_features = [col for col in total_folds.columns 

                        if col not in feature_cells and 

                        col not in feature_genes and 

                        col not in target_cols and 

                        col not in ignore_features]

categorical_features
import pickle as pkl

total_folds.to_csv("train_data_with_folds.csv", index = False)

test_data.to_csv("test_data_processed.csv", index = False)

with open('column_details.pkl', 'wb') as f : 

    pkl.dump(

    {

        'cells' : feature_cells, 

        'genes' : feature_genes, 

        'targets' : target_cols, 

        'single_val_targets' : single_val_labels, 

        

    }, f)
class MOADataset() : 

    def __init__(self, data, categ_feat, cell_feat, gene_feat, targets = None) : 

        self.data = data

        self.cell_features = cell_feat

        self.gene_features = gene_feat

        self.categ_features = categ_feat

        self.targets = targets

        self.col_names = categ_feat + gene_feat + cell_feat + targets if targets else []

        

    def get_col_names(self) : 

        return self.col_names

        

    def __len__(self) : 

        return len(self.data)

    

    def __getitem__(self, idx) : 

        categ = torch.tensor(self.data.loc[idx, self.categ_features], dtype = torch.float)

        gene = torch.tensor(self.data.loc[idx, self.gene_features], dtype = torch.float)

        cell = torch.tensor(self.data.loc[idx, self.cell_features], dtype = torch.float)

        if self.targets : 

            target = torch.tensor(self.data.loc[idx, self.targets], dtype = torch.float)

        else : 

            target = torch.tensor(1, dtype = torch.float)

        

        return (categ, gene, cell, target)

    
train_dataset = MOADataset(total_folds, categorical_features, 

                           feature_cells, feature_genes, target_cols)

train_dataset[20]
class LinearBlock(nn.Module) : 

    def __init__(self, n_inp, n_out, drop_out = 0.2) : 

        super(LinearBlock, self).__init__()

        self.linear = nn.Linear(n_inp, n_out)

        self.batch_norm = nn.BatchNorm1d(n_out)

        self.relu = nn.ReLU()

        self.drop_out = nn.Dropout(drop_out)

        

    def forward(self, inp) : 

        inp_orig = inp

        inp = self.relu(self.batch_norm(self.linear(inp)))

        inp = self.drop_out(inp)

        

        return inp + inp_orig

    

class MOANet(nn.Module) : 

    def __init__(self, n_categ, n_cell, n_gene, n_targ, n_hid_categ, n_hid_cell, 

                 n_hid_gene, n_inter_dim, n_inter_layers, drop_out = 0.2) : 

        super(MOANet, self).__init__()

        self.categ_layer = nn.Linear(n_categ, n_hid_categ)

        self.cell_layer = nn.Linear(n_cell, n_hid_cell)

        self.gene_layer = nn.Linear(n_gene, n_hid_gene)

        

        layers = []

        for i in range(n_inter_layers) : 

            if i == 0 : 

                layers.append(nn.Linear(n_hid_categ + n_hid_cell + n_hid_gene, n_inter_dim))

            else : 

                layers.append(LinearBlock(n_inter_dim, n_inter_dim, drop_out))

                            

        self.inter_layer = nn.Sequential(*layers)

        self.fc = nn.Linear(n_inter_dim, n_targ)

        

    def forward(self, categ, gene, cell) : 

        categ = self.categ_layer(categ)

        cell = self.cell_layer(cell)

        gene = self.gene_layer(gene)

        

        #print(categ.shape, cell.shape, gene.shape)

        tot = torch.cat([categ, cell, gene], 1)

        tot = self.inter_layer(tot)

        tot = self.fc(tot)

        tot = torch.sigmoid(tot)

        

        return tot
len(categorical_features), len(feature_cells), len(feature_genes), len(target_cols)
model = MOANet(7, 100, 772, 204, 8, 128, 512, 1024, 4)

train_dataloader = DataLoader(train_dataset, shuffle = True, batch_size = 32)

batch = next(iter(train_dataloader))

output = model(batch[0], batch[1], batch[2])


summary(model, 

        [(len(categorical_features), ), (len(feature_genes), ), (len(feature_cells), )], 

        device = 'cpu')
class MOADataModule(pl.LightningDataModule) : 

    def __init__(self, train_data, test_data, batch_size, num_workers, targets = None) : 

        super(MOADataModule, self).__init__(self)

        self.features = train_data

        self.batch_size = batch_size

        self.num_workers = num_workers

        self.targets = targets

        self.dataset = MOADataset

        self.test = test_data



    def setup(self) : 

        feature_names = self.features.drop(columns = ['sig_id']).columns

        feature_genes = [col for col in feature_names if col.startswith('g-')]

        feature_cells = [col for col in feature_names if col.startswith('c-')]

        feature_categ = [col for col in feature_names 

                        if col not in feature_cells and col not in feature_genes]

        

        target_cols = self.targets.drop(columns = "sig_id").columns

        self.single_val_labels = target_cols[self.targets[target_cols].sum() <= 1].tolist()

        self.target_cols = target_cols[self.targets[target_cols].sum() > 1].tolist()

        self.targets = self.targets[self.target_cols + ['sig_id']]

        

        train_data = self.features.merge(self.targets, on = "sig_id", how = "inner")

        

        train_data = pd.get_dummies(train_data, columns = feature_categ)

        test_data = pd.get_dummies(self.test, columns = feature_categ)



        total_folds = train_data.copy()



        mskf = MultilabelStratifiedKFold(n_splits=5, shuffle = True, random_state = 1997)



        for f, (train_idx, val_idx) in enumerate(mskf.split(X=train_data, 

                                                            y=train_targets.drop(columns = ['sig_id']))):

            total_folds.loc[val_idx, 'kfold'] = int(f)



        total_folds['kfold'] = total_folds['kfold'].astype(int)

        

        ignore_features = ['kfold', 'sig_id']

        categorical_features = [col for col in total_folds.columns 

                            if col not in feature_cells and 

                            col not in feature_genes and 

                            col not in target_cols and 

                            col not in ignore_features]



        train_data = total_folds[total_folds['kfold'] != 4].reset_index(drop = True)

        val_data = total_folds[total_folds['kfold'] == 4].reset_index(drop = True)

        

        self.train_indices = train_data['sig_id']

        self.val_indices = val_data['sig_id']

        self.test_indices = test_data['sig_id']

        

        self.train_ds = self.dataset(train_data, categorical_features, 

                                       feature_cells, feature_genes, self.target_cols)

        self.val_ds = self.dataset(val_data, categorical_features, 

                                       feature_cells, feature_genes, self.target_cols)

        self.test_ds = self.dataset(test_data, categorical_features, 

                                       feature_cells, feature_genes )

        

        print(f"""Setup Completed with Dataset Sizes \nTrain Dataset Size : {len(self.train_ds)} 

        Val Dataset Size : {len(self.val_ds)} \nTrain Dataset Size : {len(self.test_ds)}""")



    def __len__(self) : 

        return len(self.train_dataloader())

        

    def train_dataloader(self) : 

        return DataLoader(self.train_ds, shuffle = True, batch_size = self.batch_size, num_workers = self.num_workers)



    def val_dataloader(self) : 

        return DataLoader(self.val_ds, shuffle = False, batch_size = self.batch_size, num_workers = self.num_workers)



    def test_dataloader(self) : 

        return DataLoader(self.test_ds, shuffle = False, batch_size = self.batch_size, num_workers = self.num_workers)



    def transfer_batch_to_device(self, batch, device):

        for i in range(len(batch)) : 

            batch[i] = batch[i].to(device)

        return batch



    def get_dataloader(self, ds_type = 'train') : 

        if ds_type == 'train' : 

            dataloader = self.train_dataloader()

        elif ds_type == 'valid' : 

            dataloader = self.val_dataloader()

        else : 

            dataloader = self.test_dataloader()

        return dataloader

    

    def get_indices(self, ds_type = 'test') : 

        if ds_type == 'train' : 

            indices = self.train_indices

        elif ds_type == 'valid' : 

            indices = self.val_indices

        else : 

            indices = self.test_indices

        return indices

    

    

    def show_batch(self, ds_type = 'train', rows = 10, device = 'cpu') : 

        from IPython.display import display, HTML



        dataloader = self.get_dataloader(ds_type)

        batch = next(iter(dataloader))

        for i in range(len(batch)) : 

            batch[i] = pd.DataFrame(batch[i].detach().cpu().numpy())

        print(len(batch))

        

        data = pd.concat([batch[0], batch[1], batch[2], batch[3]], axis = 1)

        print(data.shape)

        data.columns = self.train_ds.col_names

        

        display(HTML(data.sample(rows).to_html()))

train_features = pd.read_csv("/kaggle/input/lish-moa/train_features.csv")

train_targets = pd.read_csv("/kaggle/input/lish-moa/train_targets_scored.csv")

test_features = pd.read_csv("/kaggle/input/lish-moa/test_features.csv")

sample_submission = pd.read_csv("/kaggle/input/lish-moa/sample_submission.csv")

print(f"Train Dataset Size : {train_features.shape}")

print(f"Test Dataset Size : {test_features.shape}")

train_features.head()
pl_datamodule = MOADataModule(train_features, test_features, BATCH_SIZE, NUM_WORKERS, train_targets)

pl_datamodule.setup()

pl_datamodule.show_batch()
class MOALightningModule(pl.LightningModule) : 

    def __init__(self, n_categ, n_cell, n_gene, n_targ, n_hid_categ, n_hid_cell, 

                 n_hid_gene, n_inter_dim, n_inter_layers, drop_out, batch_size, lr) : 

        

        super(MOALightningModule, self).__init__()

        self.save_hyperparameters()

        self.model = MOANet(self.hparams.n_categ, self.hparams.n_cell, self.hparams.n_gene, 

                            self.hparams.n_targ, self.hparams.n_hid_categ, self.hparams.n_hid_cell, 

                            self.hparams.n_hid_gene, self.hparams.n_inter_dim, 

                            self.hparams.n_inter_layers, self.hparams.drop_out)

        self.batch_size = self.hparams.batch_size

        self.lr = self.hparams.lr

        self.loss = nn.BCELoss()

        self.accuracy = Accuracy()

        self.f1 = F1()

        

    def forward(self, inp) : 

        return self.model(inp[0], inp[1], inp[2])

    

    def configure_optimizers(self) : 

        opt = optim.Adam(self.model.parameters(), lr = self.hparams.lr)

        return [opt]

    

    def shared_step(self, batch) : 

        outputs = self.forward(batch)

        targets = batch[3]

        loss = self.loss(outputs, targets)

        

        outputs = (outputs > 0.5) * 1

        acc = self.accuracy(outputs, targets)

        f1 = self.f1(outputs, targets)

        

        return (loss, acc, f1)

    

    def training_step(self, batch, batch_idx) : 

        loss, acc, f1 = self.shared_step(batch)

        out = OrderedDict(

        {

            'loss' : loss, 

            'metric' : acc, 

            'progress_bar' : {'Train_Loss' : loss, 'Train_Acc' : acc, 'Train_F1' : f1}, 

            'log' : {'Train_Loss' : loss, 'Train_Acc' : acc, 'Train_F1' : f1}, 

        })

        

        return out

    

    def validation_step(self, batch, batch_idx) : 

        loss, acc, f1 = self.shared_step(batch)

        out = OrderedDict(

        {

            'loss' : loss, 

            'metric' : acc, 

            'progress_bar' : {'Val_Loss' : loss, 'Val_Acc' : acc, 'Val_F1' : f1}, 

            'log' : {'Val_Loss' : loss, 'Val_Acc' : acc, 'Val_F1' : f1}, 

        })

        

        return out

            
# Load the TensorBoard notebook extension

%reload_ext tensorboard

%tensorboard --logdir ./tb_logs
# ------------------------

# 1 INIT LIGHTNING MODEL

# ------------------------

moa_lightning_module = MOALightningModule(

    7, 100, 772, 204, 8, 128, 512, 1024, 4, 0.2, batch_size = BATCH_SIZE, lr = LEARNING_RATE

)



# ------------------------

# 2 INIT TRAINER

# ------------------------

tb_logger = TensorBoardLogger('tb_logs', name='moa_model')

early_stop = pl.callbacks.EarlyStopping(

    monitor='Val_Loss',

    patience=5,

    strict=False,

    verbose=True,

    mode='min'

)



trainer = pl.Trainer(logger = tb_logger, gpus = N_GPUS, early_stop_callback = early_stop, 

                     max_steps = EPOCHS * len(pl_datamodule), 

                     )



# ------------------------

# 3 START TRAINING

# ------------------------

trainer.fit(moa_lightning_module, datamodule = pl_datamodule)

trainer.save_checkpoint('moa_model.ckpt')
model = trainer.model.model

optimizer = trainer.model.configure_optimizers()[0]

torch.save({

    'model_state_dict' : model.state_dict(), 

    'opt_state_dict' : optimizer.state_dict()

}, 'moa_model.pt')
from tqdm import tqdm 

def get_predictions(model, dataloader) : 

    outputs = []

    for batch in tqdm(dataloader) : 

        output = model(batch[0], batch[1], batch[2])

        output = (output > 0.5) * 1

        outputs.append(output)

        

    outputs = torch.cat(outputs, 0)

    outputs = outputs.detach().cpu().numpy()

    

    return outputs

    

test_dataloader = pl_datamodule.test_dataloader()

test_outputs = get_predictions(model, test_dataloader)

test_indices = pl_datamodule.get_indices('test')

test_outputs[0]
test_outputs.shape, test_indices.shape
signle_val_targets = pl_datamodule.single_val_labels

target_cols = pl_datamodule.target_cols

len(target_cols)
test_targets = pd.DataFrame(test_outputs, columns = target_cols)

test_indices = pd.Series(test_indices, name = 'sig_id')

test_submission = pd.concat([test_indices, test_targets], axis = 1)

for col in signle_val_targets : 

    test_submission[col] = 0

test_submission.head()
test_submission.to_csv('submission.csv', index = False)