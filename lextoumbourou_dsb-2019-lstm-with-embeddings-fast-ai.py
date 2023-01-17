%load_ext autoreload

%autoreload 2



import random

import os

from copy import copy

from pathlib import Path

from multiprocessing import Pool



import pandas as pd

import numpy as np



from sklearn.model_selection import GroupKFold

from sklearn.metrics import cohen_kappa_score



from fastai.tabular.transform import add_datepart, cont_cat_split

from fastai.tabular.transform import FillMissing, Categorify, Normalize

from fastai.layers import embedding

from fastai.basic_data import DataBunch

from fastai.basic_data import DatasetType

from fastai.basic_train import Learner

from fastai.basic_data import DataBunch

from fastai.layers import LabelSmoothingCrossEntropy

from fastai.metrics import KappaScore



import torch

from torch.utils import data

from torch import nn

from torch import Tensor

from torch.utils.data import TensorDataset

from torch.utils.data import DataLoader

from torch.nn.utils.rnn import pad_sequence

import torch.nn.functional as F

from tqdm.notebook import tqdm





def seed_everything(seed):

    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.deterministic = True
MAX_SEQ_LEN = 300

SEED = 420

BATCH_SIZE = 64

NUM_FOLDS = 5
seed_everything(SEED)
DATA_PATH = Path('/kaggle/input/')

OUTPUT_PATH = Path('/kaggle/working/')

(OUTPUT_PATH/'cache').mkdir(exist_ok=True)
train_df = pd.read_feather(DATA_PATH/'dsb-2019-eda-and-data-preparation/train.fth')

test_df = pd.read_feather(DATA_PATH/'dsb-2019-eda-and-data-preparation/test.fth')



train_labels = pd.read_feather(DATA_PATH/'dsb-2019-eda-and-data-preparation/train_labels.fth')

test_labels = pd.read_feather(DATA_PATH/'dsb-2019-eda-and-data-preparation/test_labels.fth')
NUM_LABELS = len(train_labels)
train_df = add_datepart(df=train_df, field_name='timestamp', drop=False, time=True)

test_df = add_datepart(df=test_df, field_name='timestamp', drop=False, time=True)
continuous_features, categorical_features = cont_cat_split(train_df)
excluded_date_feats = set([

    'timestampIs_month_end',

    'timestampIs_month_start',

    'timestampIs_quarter_end',

    'timestampIs_quarter_start',

    'timestampIs_year_end',

    'timestampIs_year_start',

    'timestampYear'

])



categorical_features = [c for c in categorical_features if c not in excluded_date_feats]
continuous_features
categorical_features
categorical_features = [c for c in categorical_features if c not in ('game_session', 'installation_id', 'timestamp')]
NUM_FEATS = len(continuous_features + categorical_features)
fm = FillMissing(cat_names=copy(categorical_features), cont_names=copy(continuous_features))

fm(train_df)

fm(test_df, test=True)
cy = Categorify(cat_names=copy(categorical_features), cont_names=copy(continuous_features))

cy(train_df)

cy(test_df, test=True)
nm = Normalize(cat_names=copy(categorical_features), cont_names=copy(continuous_features))

nm(train_df)

nm(test_df, test=True)
class TSDataset(data.Dataset):



    def __init__(self, labels, df, max_seq_len=MAX_SEQ_LEN):

        self.labels = labels

        self.df = df

        self.max_seq_len = max_seq_len

        

    def __len__(self):

        return len(self.labels)

    

    def __getitem__(self, i):

        row = self.labels.iloc[i]



        X_clipped = self.df.loc[row.start_idx:row.end_idx, categorical_features + continuous_features].values

        

        X = np.zeros((self.max_seq_len, NUM_FEATS))

        X_clipped = X_clipped[-self.max_seq_len:]



        if len(X_clipped):

            X[-len(X_clipped):] = X_clipped

    

        return Tensor(X).float(), row.accuracy_group
def _numericise_cats(df, cat_cols):

    for col in cat_cols:

        if df[col].dtype.name == 'category':

            df[col] = df[col].cat.codes + 1

  

    return df
train_df = _numericise_cats(train_df[categorical_features + continuous_features].copy(), cat_cols=categorical_features)

test_df = _numericise_cats(test_df[categorical_features + continuous_features].copy(), cat_cols=categorical_features)
def embedding_size_rule(number_categories):

    return min(600, round(1.6 * number_categories**0.56))
emb_sizes, cat_sizes = {}, {}



for col in categorical_features:

    num_cats = train_df[col].nunique() + 1

    emb_sizes[col] = embedding_size_rule(num_cats)

    cat_sizes[col] = num_cats
emb_sizes
cat_sizes
LSTM_UNITS = 128

DENSE_HIDDEN_UNITS = 4 * LSTM_UNITS





class TimeSeriesLSTM(nn.Module):

    def __init__(self, emb_drop=0.5, lstm_1_dropout=0.3, lstm_2_dropout=0.3):

        super().__init__()



        self.embeds = nn.ModuleList([

            embedding(cat_sizes[cat], emb_sizes[cat])

            for cat in categorical_features

        ])

        self.embedding_dropout = nn.Dropout(emb_drop)

        

        total_embeds = sum(emb_sizes.values())

        

        self.lstm1 = nn.LSTM(total_embeds + len(continuous_features), LSTM_UNITS, batch_first=True, bidirectional=True)

        self.lstm1_dropout = nn.Dropout(lstm_1_dropout)



        self.lstm2 = nn.LSTM(LSTM_UNITS * 2, LSTM_UNITS, batch_first=True, bidirectional=True)

        self.lstm2_dropout = nn.Dropout(lstm_2_dropout)

    

        self.linear1 = nn.Linear(DENSE_HIDDEN_UNITS, DENSE_HIDDEN_UNITS)

        self.linear2 = nn.Linear(DENSE_HIDDEN_UNITS, DENSE_HIDDEN_UNITS)

        

        self.linear_out = nn.Linear(DENSE_HIDDEN_UNITS, 4)

        

    def forward(self, x_input):

        x_cat = x_input[:,:,:len(categorical_features)]

        x_cont = x_input[:,:,len(categorical_features):]

        

        h_embedding = [e(x_cat[:,:,i].long()) for i, e in enumerate(self.embeds)]

        h_embedding = torch.cat(h_embedding, 2)

        h_embedding = self.embedding_dropout(h_embedding)

        

        x_cat = torch.cat([h_embedding, x_cont], 2)

        

        h_lstm1, _ = self.lstm1(x_cat)

        h_lstm1 = self.lstm1_dropout(h_lstm1)

        h_lstm2, _ = self.lstm2(h_lstm1)

        h_lstm2 = self.lstm2_dropout(h_lstm2)



        avg_pool = torch.mean(h_lstm2, 1)

        max_pool, _ = torch.max(h_lstm2, 1)

        

        h_conc = torch.cat((max_pool, avg_pool), 1)



        h_conc_linear1 = F.relu(self.linear1(h_conc))

        h_conc_linear2 = F.relu(self.linear2(h_conc))

        

        hidden = h_conc + h_conc_linear1 + h_conc_linear2

        

        result = self.linear_out(hidden)

        

        return result
kappa = KappaScore()

kappa.weights = "quadratic"
val_preds = np.zeros((len(train_labels), 4))

test_preds = np.zeros((NUM_FOLDS, len(test_labels), 4))
test_ds = TSDataset(labels=test_labels, df=test_df)

test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE)
kfold = GroupKFold(n_splits=NUM_FOLDS)



for i, (train_idx, val_idx) in enumerate(kfold.split(train_labels, train_labels.accuracy_group, train_labels.installation_id)):

    train_ds = TSDataset(labels=train_labels.iloc[train_idx], df=train_df)

    val_ds = TSDataset(labels=train_labels.iloc[val_idx], df=train_df)



    train_dl = DataLoader(train_ds, shuffle=True, batch_size=BATCH_SIZE)

    valid_dl = DataLoader(val_ds, batch_size=BATCH_SIZE)



    databunch = DataBunch(train_dl=train_dl, valid_dl=valid_dl, test_dl=test_dl)



    ts_nn = TimeSeriesLSTM()



    learner = Learner(

        data=databunch, model=ts_nn, metrics=[kappa], loss_func=LabelSmoothingCrossEntropy())

    

    learner.fit_one_cycle(4, max_lr=1e-02)

    

    val_preds_fold, val_y_fold = learner.get_preds(ds_type=DatasetType.Valid)

    val_preds[val_idx] = torch.softmax(val_preds_fold, 1).numpy()

    

    test_preds_folds, _ = learner.get_preds(ds_type=DatasetType.Test)

    test_preds[i] = torch.softmax(test_preds_folds, 1).numpy()

    

    learner.save(f'lstm_fold_{i}')
cohen_kappa_score(train_labels.accuracy_group.values, np.argmax(val_preds, 1), weights='quadratic')
np.save(OUTPUT_PATH/'val_preds', val_preds)
all_test_preds = np.mean(test_preds, 0)
np.save(OUTPUT_PATH/'test_preds', all_test_preds)