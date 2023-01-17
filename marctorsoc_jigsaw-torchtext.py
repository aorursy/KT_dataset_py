%load_ext autoreload

%autoreload 2

import logging

log = logging.getLogger(__name__)

logging.basicConfig(

    level=logging.INFO, format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"

)

import numpy as np 

import pandas as pd

from pathlib import Path

import os, re, gc, random, tqdm

 

from nltk.tokenize import TweetTokenizer

from collections import Counter



#import spacy

#nlp = spacy.load("en_core_web_sm", disable=["parser", "tagger", "ner"])



#from sklearn.model_selection import StratifiedKFold

#from sklearn.metrics import roc_auc_score



import torch

import torch.nn as nn

import torch.optim as optim

import torch.nn.functional as F



import torchtext

from torchtext import vocab, data

from torchtext.data import Field, BucketIterator, Dataset



import warnings

warnings.filterwarnings('ignore')
from datetime import datetime

import os

def log(msg):

    now = pd.to_datetime(datetime.now()).round("s")

    print(f"{now} {msg}")

    os.system(f"echo {now} {msg}")
log("Start!")
vocab.tqdm = tqdm.tqdm_notebook # Replace tqdm to tqdm_notebook in module torchtext
# if True: 

#     path = Path('../../../Documents/Marc/datasets/jigsaw-unintended-bias-in-toxicity-classification/')

#     emb_path = Path('../../../Documents/Marc/datasets/language_models')

# else:

#     path = Path('../../../jigsaw-unintended-bias-in-toxicity-classification/')

#     emb_path = Path('../../../language_models')



path = Path('../input/jigsaw-unintended-bias-in-toxicity-classification/')

emb_path = Path('../input/glove840b300dtxt')



emb_cache = './'

glove_path = emb_path / 'glove.840B.300d.txt'



device = 'cuda'

# preproc hyperparams

nrows_train = None

nrows_test = 2 # we can disable test as we are not in competition anymore

max_vocab_size = 500000

min_vocab_freq = 3 # 1

# model hyperparams

n_folds = 3

epochs = 10

batch_size = 512

bidirectional=True

n_hidden = 64 # try 128?

pos_weight = 2.7

loss_fn = torch.nn.BCEWithLogitsLoss(reduction='mean', pos_weight=(torch.Tensor([pos_weight])).to(device))



gc.enable();
# seeds

seed = 7777

random.seed(seed)

os.environ['PYTHONHASHSEED'] = str(seed)

np.random.seed(seed)

torch.manual_seed(seed)

torch.cuda.manual_seed(seed)

torch.cuda.manual_seed_all(seed)

torch.backends.cudnn.deterministic = True
# Load csv file with selected columns



identity_columns = [

    'male', 'female', 'homosexual_gay_or_lesbian', 'christian', 'jewish',

    'muslim', 'black', 'white', 'psychiatric_or_mental_illness'

]



cols = ['id', 'comment_text', 'target'] + identity_columns

dtypes = {'target': np.float16,'comment_text': object,'id': np.int32}

for c in identity_columns:

    dtypes[c] = np.float16

df = pd.read_csv(path / 'train.csv', usecols=cols, dtype=dtypes, index_col=[0], nrows=nrows_train)



# Test csv

test_df = pd.read_csv(

    path / 'test.csv', dtype={'comment_text': object,'id': np.int32}, index_col=[0], nrows=nrows_test

)



test_df['prediction'] = 0
class RegExCleaner:

    def __init__(self, expressions=[]):

        r"""Create class from compiled expressions: [(re.compile(pattern), repl)]"""

        self.expressions = expressions



    @staticmethod

    def _compile(expressions):

        regexps = []

        for pattern, repl in expressions.items():

            regexps.append((re.compile(pattern), repl))

        return regexps



    @classmethod

    def from_dict(cls, custom_dic):

        r"""Create class from dictionary with flexible patterns {pattern : replacing}"""

        return cls(cls._compile(custom_dic))



    @classmethod

    def from_vocab(cls, vocab):

        r"""Create class from vocabulary with fixed patterns {pattern : replacing}"""

        pattern = re.compile("|".join(map(re.escape, vocab.keys())))

        repl = lambda match: vocab[match.group(0)]

        return cls([(pattern, repl)])



    def __add__(self, b):

        return RegExCleaner(self.expressions + b.expressions)



    def __call__(self, s):

        for regex, repl in self.expressions:

            s = regex.sub(repl, s)

        return s

%%time

from tqdm._tqdm_notebook import tqdm_notebook as tqdm_pandas

tqdm_pandas.pandas()

#from preprocessing import RegExCleaner



# TODO: this can be improved using SpaCy, careful strip_handles removes the user names, so

# we need to add this to the cleaner before using spaCy

tknzr = TweetTokenizer(strip_handles=True, reduce_len=True)

# TODO: better cleaning / normalisation using contractions dict

cleaner = RegExCleaner.from_dict({r'https?:/\/\S+':r' ',

                                  r'[^A-Za-z0-9!.,?$\'\"]+':r' '})

def preparation(s): 

    s = cleaner(s)

    return tknzr.tokenize(s)



#df.comment_text = df.comment_text.progress_apply(preparation)

#test_df.comment_text = test_df.comment_text.progress_apply(preparation)

df.comment_text = df.comment_text.apply(preparation)

test_df.comment_text = test_df.comment_text.apply(preparation)



# count unique words

counter = Counter()

for comment in df.comment_text:

    counter.update(comment)
%%time

# this is done once, and after that the vectors are compressed to emb_cache, so the loading will

# go from 5+ min in my modest laptop, to 10 seconds

vec = vocab.Vectors(glove_path, cache=emb_cache)

log("Vectors loaded, creating vocab")

vocabulary = vocab.Vocab(counter, max_size=max_vocab_size, min_freq=min_vocab_freq, 

                         vectors=vec, specials=['<pad>', '<unk>'])

torch.zero_(vocabulary.vectors[1]); # fill <unk> token as 0



del vec

gc.collect();

log(f"Embedding vocab size: {vocabulary.vectors.size(0)}")
# function for transform sentence to sequence of encoded words

import tqdm

from torchtext.data import BucketIterator, Dataset, Example

import numpy as np





def sentence2numbers(s, vocabulary, fill_as=1):  # fill <unk> token

    # TODO: this could be more efficient, maybe np.vectorize or pre-allocating?

    seq = []

    for w in s:

        try:

            seq.append(vocabulary.stoi[w])

        except KeyError:

            seq.append(fill_as)

    return np.array(seq, dtype=np.int32)





# Extension of pytorch Dataset to work with DataFrames

class DataFrameDataset(Dataset):

    def __init__(self, df, fields, **kwargs):

        keys = dict(fields)

        for n, f in list(keys.items()):

            if isinstance(n, tuple):

                keys.update(zip(n, f))

                del keys[n]

        keys = keys.keys()

        examples = []

        for i, row in tqdm.tqdm_notebook(df.iterrows(), total=len(df)):

            examples.append(Example.fromlist([row[k] for k in keys], fields))



        super().__init__(examples, fields, **kwargs)





# A wrapper to split a DataFrameDataset in subsets. This is just a wrapper where we

# load the dataset once and then change idx to select the fold we're working with

class DataFrameDataSubset:

    def __init__(self, df_ds, idx):

        # this should an object of type DataFrameDataset

        self.df_ds = df_ds

        self.idx = idx

        self.len = len(idx)

        self.fields = self.df_ds.fields



    def update(self, idx):

        self.idx = idx

        self.len = len(idx)



    def __getitem__(self, idx):

        return self.df_ds[self.idx[idx]]



    def __len__(self):

        return self.len





# A wrapper for BucketIterator (the equivalent or DataLoader for torchtext). This

# wrapper defines how to get the info for every batch for train/val vs test

class BatchWrapper:

    def __init__(self, dl, mode="train"):

        self.dl, self.mode = dl, mode



    def __iter__(self):

        if self.mode != "test":

            for batch in self.dl:

                yield (batch.comment_text, batch.target, batch.id)

        else:

            for batch in self.dl:

                yield (batch.comment_text, batch.id)



    def __len__(self):

        return len(self.dl)





def get_loader_from_dataset(ds, mode="train", **args):

    return BatchWrapper(BucketIterator(ds, **args), mode)

#from data_preparation import sentence2numbers

from functools import partial

# define the columns that we want to process and how to process

pad_token = 0 # '<pad>' position token



txt_field = Field(sequential=True, preprocessing=partial(sentence2numbers, vocabulary=vocabulary),

                  pad_token=pad_token, use_vocab=False)

num_field = Field(sequential=False, dtype=torch.float, use_vocab=False)

idx_field = Field(sequential=False, dtype=torch.int64, use_vocab=False)



train_fields = [

    ('id', idx_field), 

    ('target', num_field), 

    ('comment_text', txt_field),

]

test_fields = [

    ('id', idx_field), 

    ('comment_text', txt_field), 

]
# for train/val we need we load all data into the DataFrameDataSubset and for each fold we will

# select a subset of it, then create a loader

#from data_preparation import DataFrameDataset, DataFrameDataSubset

ds = DataFrameDataset(df.reset_index(), train_fields)

test_ds = DataFrameDataset(test_df.reset_index(), test_fields)

train_dss, valid_dss = DataFrameDataSubset(ds, [0]), DataFrameDataSubset(ds, [0])



# for test all data will be used for all folds, so the loader can be created here

#from data_preparation import get_loader_from_dataset

te_loader = get_loader_from_dataset(

    test_ds, mode='test', batch_size=batch_size, device=device,

    sort_key=lambda x: len(x.comment_text),

    sort_within_batch=True, shuffle=False, repeat=False

)
import torch

from torch import nn

from torch.nn import functional as F





class RecNN(nn.Module):

    def __init__(

        self, embs_vocab, hidden_size, layers=1, dropout=0.0, bidirectional=False

    ):

        super().__init__()



        self.hidden_size = hidden_size

        self.bidirectional = bidirectional

        self.num_layers = layers

        self.emb = nn.Embedding.from_pretrained(embs_vocab)



        self.line = nn.Linear(embs_vocab.size(1), embs_vocab.size(1))



        self.lstm = nn.LSTM(

            embs_vocab.size(1),

            self.hidden_size,

            num_layers=layers,

            bidirectional=bidirectional,

            dropout=dropout,

        )



        self.gru = nn.GRU(

            embs_vocab.size(1),

            self.hidden_size,

            num_layers=layers,

            bidirectional=bidirectional,

            dropout=dropout,

        )



        self.out = nn.Linear(self.hidden_size * (bidirectional + 1), 32)

        self.last = nn.Linear(32, 1)



    def forward(self, x):



        embs = self.emb(x)

        lstm, (h, c) = self.lstm(embs)



        x = F.relu(self.line(embs), inplace=True)

        gru, h = self.gru(x, h)

        lstm = lstm + gru



        lstm, _ = lstm.max(dim=0, keepdim=False)

        out = self.out(lstm)

        out = self.last(F.relu(out)).squeeze()

        return out.squeeze(-1)



    @torch.no_grad()

    def prediction(self, x):

        score = self.forward(x)

        return torch.sigmoid(score)



    @torch.no_grad()

    def evaluate(self, x, y, func):

        preds = self.forward(x)

        loss = func(preds.squeeze(-1), y)

        return preds, loss

# Thanks sakami

# https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/discussion/90527

from sklearn.metrics import roc_auc_score

import numpy as np





class JigsawEvaluator:

    def __init__(self, y_true, y_identity, power=-5, overall_model_weight=0.25):

        self.y = (y_true >= 0.5).astype(int)

        self.y_i = (y_identity >= 0.5).astype(int)

        self.n_subgroups = self.y_i.shape[1]

        self.power = power

        self.overall_model_weight = overall_model_weight



    @staticmethod

    def _compute_auc(y_true, y_pred):

        try:

            return roc_auc_score(y_true, y_pred)

        except ValueError:

            return np.nan



    def _compute_subgroup_auc(self, i, y_pred):

        mask = self.y_i[:, i] == 1

        return self._compute_auc(self.y[mask], y_pred[mask])



    def _compute_bpsn_auc(self, i, y_pred):

        mask = self.y_i[:, i] + self.y == 1

        return self._compute_auc(self.y[mask], y_pred[mask])



    def _compute_bnsp_auc(self, i, y_pred):

        mask = self.y_i[:, i] + self.y != 1

        return self._compute_auc(self.y[mask], y_pred[mask])



    def compute_bias_metrics_for_model(self, y_pred):

        records = np.zeros((3, self.n_subgroups))

        for i in range(self.n_subgroups):

            records[0, i] = self._compute_subgroup_auc(i, y_pred)

            records[1, i] = self._compute_bpsn_auc(i, y_pred)

            records[2, i] = self._compute_bnsp_auc(i, y_pred)

        return records



    def _calculate_overall_auc(self, y_pred):

        return roc_auc_score(self.y, y_pred)



    def _power_mean(self, array):

        total = sum(np.power(array, self.power))

        return np.power(total / len(array), 1 / self.power)



    def get_final_metric(self, y_pred):

        bias_metrics = self.compute_bias_metrics_for_model(y_pred)

        bias_score = np.average(

            [

                self._power_mean(bias_metrics[0]),

                self._power_mean(bias_metrics[1]),

                self._power_mean(bias_metrics[2]),

            ]

        )

        overall_score = self.overall_model_weight * self._calculate_overall_auc(y_pred)

        bias_score = (1 - self.overall_model_weight) * bias_score

        return overall_score + bias_score

import gc



import numpy as np

from sklearn.model_selection import StratifiedKFold

from torch.optim import Adam

#from data_preparation import wrapper

#from metrics import JigsawEvaluator

#from models import RecNN

#from utils import sigmoid



def sigmoid(x):

    return 1 / (1 + np.exp(-x))



def train_predict(

    train_ds, valid_ds, df, test_df, te_loader, params, vocabulary, loss_fn

):

    bs, device, n_hidden, bidir, epochs, identity_columns, n_folds, seed = params

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    y = df.target.values > 0.5



    for train_idx, valid_idx in skf.split(y, y=y):



        train_ds.update(train_idx)

        valid_ds.update(valid_idx)



        tr_loader = get_loader_from_dataset(

            train_ds,

            batch_size=bs,

            device=device,

            sort_key=lambda key: len(key.comment_text),

            sort_within_batch=True,

            shuffle=True,

            repeat=False,

        )



        v_loader = get_loader_from_dataset(

            valid_ds,

            batch_size=bs,

            device=device,

            sort_key=lambda key: len(key.comment_text),

            sort_within_batch=True,

            shuffle=False,

            repeat=False,

        )



        model = RecNN(

            vocabulary.vectors, n_hidden, layers=2, dropout=0.2, bidirectional=bidir

        ).to(device)



        opt = Adam(

            params=filter(lambda p: p.requires_grad, model.parameters()),

            lr=1e-3,

            betas=(0.75, 0.999),

            eps=1e-08,

            weight_decay=0,

        )



        for epoch in range(epochs):

            y_true_train = np.empty(0)

            y_pred_train = np.empty(0)

            total_loss_train, total_loss_valid = 0, 0

            model.train()

            tcids, vcids = [], []

            for x, target, ids in tr_loader:

                tcids.append(ids.detach().cpu().numpy())

                opt.zero_grad()

                pred = model(x)

                loss = loss_fn(pred, target)

                loss.backward()

                opt.step()



                y_true_train = np.concatenate(

                    [y_true_train, target.detach().cpu().numpy()], axis=0

                )

                y_pred_train = np.concatenate(

                    [y_pred_train, pred.detach().cpu().numpy()], axis=0

                )

                total_loss_train += loss.item()



            # Get prediction for validation part

            model.eval()

            y_true_valid, y_pred_valid = np.empty(0), np.empty(0)



            for x, target, ids in v_loader:

                vcids.append(ids.detach().cpu().numpy())

                pred, loss = model.evaluate(x, target, loss_fn)

                total_loss_valid += loss.item()



                y_true_valid = np.concatenate(

                    [y_true_valid, target.detach().cpu().data.numpy()], axis=0

                )

                y_pred_valid = np.concatenate(

                    [y_pred_valid, pred.cpu().data.numpy()], axis=0

                )



            tacc = get_accuracy(df, identity_columns, tcids, y_pred_train, y_true_train)

            vacc = get_accuracy(df, identity_columns, vcids, y_pred_valid, y_true_valid)



            tloss = total_loss_train / len(tr_loader)

            vloss = total_loss_valid / len(v_loader)

            log(

                f"Epoch {epoch + 1}: Train loss: {tloss:.4f}, BIAS AUC: {tacc:.4f}, "

                f"Valid loss: {vloss:.4f}, BIAS AUC: {vacc:.4f}"

            )



        log("\n")

        gc.collect()

        # Get prediction for test set

        preds = np.empty(0)

        cids = []

        for x, ids in te_loader:

            cids.append(ids.detach().cpu().numpy())

            pred = model.prediction(x)

            preds = np.concatenate([preds, pred.detach().cpu().numpy()], axis=0)



        # Save prediction of test to DataFrame

        cids = [item for sublist in cids for item in sublist]

        test_df.at[cids, "prediction"] = (

            test_df.loc[cids]["prediction"].values + preds / n_folds

        )





def get_accuracy(df, identity_columns, cids, y_pred, y_true):

    cids = [item for sublist in cids for item in sublist]

    scorer = JigsawEvaluator(y_true, df.loc[cids][identity_columns].values)

    return scorer.get_final_metric(sigmoid(y_pred))

#from models import RecNN

#from training import train_predict



params = (batch_size, device, n_hidden, bidirectional, epochs, identity_columns, n_folds, seed)  

train_predict(train_dss, valid_dss, df, test_df, te_loader, params, vocabulary, loss_fn)
test_df.to_csv('submission.csv', columns=['prediction'])