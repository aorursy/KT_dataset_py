import torch

import numpy as np

import pandas as pd

from tqdm import tqdm

from tensorflow.keras.preprocessing.text import Tokenizer

from tensorflow.keras.preprocessing.sequence import pad_sequences

import torch.nn as nn

from torch.utils.data import Dataset, DataLoader

import fastText

from fastText import load_model

import gc

import re

tqdm.pandas()

from gensim.models import KeyedVectors

from fastprogress import master_bar, progress_bar

from pathlib import Path
TEXT_COL = 'comment_text'

VECS_PATH = Path('../input/fasttext-crawl-300d-2m/crawl-300d-2M.vec')

TRAIN_DATA = '../input/jigsaw-unintended-bias-in-toxicity-classification/train.csv'

TEST_DATA = '../input/jigsaw-unintended-bias-in-toxicity-classification/test.csv'
def seed_torch(seed=1029):

    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.deterministic = True
misspell_dict = {"aren't": "are not", "can't": "cannot", "couldn't": "could not",

                 "didn't": "did not", "doesn't": "does not", "don't": "do not",

                 "hadn't": "had not", "hasn't": "has not", "haven't": "have not",

                 "he'd": "he would", "he'll": "he will", "he's": "he is",

                 "i'd": "I had", "i'll": "I will", "i'm": "I am", "isn't": "is not",

                 "it's": "it is", "it'll": "it will", "i've": "I have", "let's": "let us",

                 "mightn't": "might not", "mustn't": "must not", "shan't": "shall not",

                 "she'd": "she would", "she'll": "she will", "she's": "she is",

                 "shouldn't": "should not", "that's": "that is", "there's": "there is",

                 "they'd": "they would", "they'll": "they will", "they're": "they are",

                 "they've": "they have", "we'd": "we would", "we're": "we are",

                 "weren't": "were not", "we've": "we have", "what'll": "what will",

                 "what're": "what are", "what's": "what is", "what've": "what have",

                 "where's": "where is", "who'd": "who would", "who'll": "who will",

                 "who're": "who are", "who's": "who is", "who've": "who have",

                 "won't": "will not", "wouldn't": "would not", "you'd": "you would",

                 "you'll": "you will", "you're": "you are", "you've": "you have",

                 "'re": " are", "wasn't": "was not", "we'll": " will", "tryin'": "trying"}
def _get_misspell(misspell_dict):

    misspell_re = re.compile('(%s)' % '|'.join(misspell_dict.keys()))

    return misspell_dict, misspell_re
def replace_typical_misspell(text):

    misspellings, misspellings_re = _get_misspell(misspell_dict)



    def replace(match):

        return misspellings[match.group(0)]



    return misspellings_re.sub(replace, text)
puncts = [',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&', '/', '[', ']',

          '>', '%', '=', '#', '*', '+', '\\', '•', '~', '@', '£', '·', '_', '{', '}', '©', '^',

          '®', '`', '<', '→', '°', '€', '™', '›', '♥', '←', '×', '§', '″', '′', 'Â', '█',

          '½', 'à', '…', '“', '★', '”', '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶',

          '↑', '±', '¿', '▾', '═', '¦', '║', '―', '¥', '▓', '—', '‹', '─', '▒', '：', '¼',

          '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲',

          'è', '¸', '¾', 'Ã', '⋅', '‘', '∞', '∙', '）', '↓', '、', '│', '（', '»', '，', '♪',

          '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø', '¹', '≤', '‡', '√']
def clean_text(x):

    x = str(x)

    for punct in puncts + list(string.punctuation):

        if punct in x:

            x = x.replace(punct, f' {punct} ')

    return x
def clean_numbers(x):

    return re.sub('\d+', ' ', x)
def get_coefs(word, *arr):

    return word, np.asarray(arr, dtype='float32')
def load_fasttext(word_index):

    embeddings_index = dict(get_coefs(*o.strip().split(' ')) for o in open(VECS_PATH))

    nb_words = min(max_features, len(word_index))

    embedding_matrix = np.zeros((nb_words, embed_size))

    for word, i in word_index.items():

        if i >= max_features:

            continue

        embedding_vector = embeddings_index.get(word)

        if embedding_vector is not None:

            embedding_matrix[i] = embedding_vector

    return embedding_matrix

maxlen = 220

max_features = 100000

embed_size = 300

batch_size = 1024

train_epochs = 5

n_splits = 5
seed = 1024
def load_and_prec():

    train = pd.read_csv(TRAIN_DATA, index_col='id')

    test = pd.read_csv(TEST_DATA, index_col='id')

    train['comment_text'] = train['comment_text'].str.lower()

    test['comment_text'] = test['comment_text'].str.lower()

    train['comment_text'] = train['comment_text'].apply(replace_typical_misspell)

    test['comment_text'] = test['comment_text'].apply(replace_typical_misspell)

    train['comment_text'] = train['comment_text'].apply(clean_text)

    test['comment_text'] = test['comment_text'].apply(clean_text)

    train['comment_text'] = train['comment_text'].apply(clean_numbers)

    test['comment_text'] = test['comment_text'].apply(clean_numbers)

    train_x = train['comment_text'].fillna('_##_').values

    test_x = test['comment_text'].fillna('_##_').values

    tokenizer = Tokenizer(num_words=max_features)

    tokenizer.fit_on_texts(list(train_x))

    train_x = tokenizer.texts_to_sequences(train_x)

    test_x = tokenizer.texts_to_sequences(test_x)

    train_x = pad_sequences(train_x, maxlen=maxlen)

    test_x = pad_sequences(test_x, maxlen=maxlen)

    train_y = (train['target'].values > 0.5).astype(int)

    np.random.seed(seed)

    train_idx = np.random.permutation(len(train_x))

    train_x = train_x[train_idx]

    train_y = train_y[train_idx]

    return train_x, train_y, test_x, tokenizer.word_index
class Net(nn.Module):

    def __init__(self, embedding_matrix):

        super(Net, self).__init__()

        lstm_hidden_size = 120

        gru_hidden_size = 60

        self.gru_hidden_size = gru_hidden_size

        self.embedding = nn.Embedding(max_features, embed_size)

        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))

        self.embedding.weight.requires_grad = False

        self.embedding_dropout = nn.Dropout2d(0.1)

        self.lstm = nn.LSTM(embed_size, lstm_hidden_size, bidirectional=True, batch_first=True)

        self.gru = nn.GRU(lstm_hidden_size*2, gru_hidden_size, bidirectional=True, batch_first=True)

        self.linear = nn.Linear(gru_hidden_size*6, 16)

        self.relu = nn.ReLU()

        self.dropout = nn.Dropout(0.1)

        self.out = nn.Linear(16, 1)

    def forward(self, x):

        h_embedding = self.embedding(x)

        h_embedding = torch.unsqueeze(h_embedding.transpose(1, 2), 2)

        h_embedding = torch.squeeze(self.embedding_dropout(h_embedding)).transpose(1, 2)

        h_lstm, _ = self.lstm(h_embedding)

        h_gru, hh_gru = self.gru(h_lstm)

        hh_gru = hh_gru.view(-1, self.gru_hidden_size*2)

        avg_pool = torch.mean(h_gru, 1)

        max_pool, _ = torch.max(h_gru, 1)

        conc = torch.cat((hh_gru, avg_pool, max_pool), 1)

        conc = self.relu(self.linear(conc))

        conc = self.dropout(conc)

        out = self.out(conc)

        return out
from contextlib import contextmanager

import time

import string

import warnings

warnings.filterwarnings('ignore')
@contextmanager

def timer(msg):

    t0 = time.time()

    print(f'[{msg}] start.')

    yield

    elapsed_time = time.time() - t0

    print(f'[{msg}] done in {elapsed_time / 60:.2f} min.')
with timer('load data'):

    train_x, train_y, test_x, word_index = load_and_prec()

    embedding_matrix = load_fasttext(word_index)
import random

import os
seed_torch(seed)
_camel_re1 = re.compile('(.)([A-Z][a-z]+)')

_camel_re2 = re.compile('([a-z0-9])([A-Z])')

def camel2snake(name):

    s1 = re.sub(_camel_re1, r'\1_\2', name)

    return re.sub(_camel_re2, r'\1_\2', s1).lower()
class Callback():

    _order = 0

    def set_runner(self, run): self.run = run

    def __getattr__(self, k): return getattr(self.run, k)

    @property

    def name(self):

        name = re.sub(r'Callback$', '', self.__class__.__name__)

        return camel2snake(name or 'callback')

    def __call__(self, cb_name):

        f = getattr(self, cb_name, None)

        if f and f(): return True

        return False
class TrainEvalCallback(Callback):

    def begin_fit(self):

        self.run.n_epochs = 0

        self.run.n_iter = 0

    def after_batch(self):

        if not self.in_train: return

        self.run.n_epochs+=1./self.iters

        self.run.n_iter+=1

    def begin_epoch(self):

        self.run.n_epochs = self.epoch

        self.model.train()

        self.run.in_train = True

    def begin_validate(self):

        self.model.eval()

        self.run.in_train = False



class CancelTrainException(Exception): pass

class CancelEpochException(Exception): pass

class CancelBatchException(Exception): pass
from collections import Iterable
def listify(o):

    if o is None: return [] 

    if isinstance(o, list): return o

    if isinstance(o, str): return [o]

    if isinstance(o, Iterable): return list(o)

    return [o]
class Runner():

    def __init__(self, cbs=None, cb_funcs=None):

        cbs = listify(cbs)

        for cbf in listify(cb_funcs):

            cb = cbf()

            setattr(self, cb.name, cb)

            cbs.append(cb)

        self.stop, self.cbs = False, [TrainEvalCallback()] + cbs

    @property

    def opt(self): return self.learn.opt

    @property

    def model(self): return self.learn.model

    @property

    def loss_func(self): return self.learn.loss_func

    @property

    def data(self): return self.learn.data

    

    def one_batch(self, xb, yb):

        try: 

            self.xb, self.yb = xb, yb

            self('begin_batch')

            self.pred = self.model(self.xb)

            self('after_pred')

            self.loss = self.loss_func(self.pred, self.yb)

            self('after_loss')

            if not self.in_train: return

            self.loss.backward()

            self('after_backward')

            self.opt.step()

            self('after_step')

            self.opt.zero_grad()

        except CancelBatchException: self('after_cancel_batch')

        finally: self('after_batch')

    

    def all_batches(self, dl):

        self.iters = len(dl)

        try:

            for xb, yb in progress_bar(dl, leave=False): self.one_batch(xb, yb)

        except CancelEpochException: self('after_cancel_epoch')

    def fit(self, epochs, learn):

        self.epochs, self.learn, self.loss = epochs, learn, torch.tensor(0.)

        try: 

            for cb in self.cbs: cb.set_runner(self)

            self('begin_fit')

            for epoch in range(epochs):

                self.epoch = epoch

                if not self('begin_epoch'): self.all_batches(self.data.train_dl)

                with torch.no_grad():

                    if not self('begin_validate'): self.all_batches(self.data.valid_dl)

                self('after_epoch')

        except CancelTrainException: self('after_cancel_train')

        finally:

            self('after_fit')

            self.learn = None

    def __call__(self, cb_name):

        res = False

        for cb in sorted(self.cbs, key=lambda x: x._order): res = cb(cb_name) and res

        return res
class Learner():

    def __init__(self, model, opt, loss_func, data):

        self.model, self.opt, self.loss_func, self.data = model, opt, loss_func, data
def get_model(data, lr=0.005):

    model = Net(embedding_matrix).to(device)

    return model, torch.optim.Adam(model.parameters(), lr)
class DataBunch():

    def __init__(self, train_dl, valid_dl):

        self.train_dl, self.valid_dl = train_dl, valid_dl

    @property

    def train_ds(self): return self.train_dl.dataset

    

    @property

    def valid_ds(self): return self.valid_dl.dataset
from sklearn.model_selection import StratifiedKFold

splits = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=seed).split(train_x, train_y))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
x_test_cuda = torch.tensor(test_x, dtype=torch.long).to(device)

test = torch.utils.data.TensorDataset(x_test_cuda)

test_dl = DataLoader(test, batch_size=batch_size, shuffle=False)
def get_data(train_idx, valid_idx):

    x_train_ds = torch.tensor(train_x[train_idx], dtype=torch.long).to(device)

    y_train_ds = torch.tensor(train_y[train_idx, np.newaxis], dtype=torch.float32).to(device)

    x_val_ds = torch.tensor(train_x[valid_idx], dtype=torch.long).to(device)

    y_val_ds = torch.tensor(train_y[valid_idx, np.newaxis], dtype=torch.float32).to(device)

    train_ds = torch.utils.data.TensorDataset(x_train_ds, y_train_ds)

    valid_ds = torch.utils.data.TensorDataset(x_val_ds, y_val_ds)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    valid_dl = DataLoader(valid_ds, batch_size=batch_size, shuffle=False)

    data = DataBunch(train_dl, valid_dl)

    return data
loss_fn = nn.BCEWithLogitsLoss(reduction='sum')
def get_learner(train_idx, valid_idx):

    data = get_data(train_idx, valid_idx)

    learn = Learner(*get_model(data), loss_fn, data=data)

    return learn
class AvgStats():

    def __init__(self, metrics, in_train): self.metrics, self.in_train = listify(metrics), in_train

    def reset(self):

        self.tot_loss, self.count = 0., 0

        self.tot_mets = [0.]*len(self.metrics)

    @property

    def all_stats(self): return [self.tot_loss.item()] + self.tot_mets

    @property

    def avg_stats(self): return [o/self.count for o in self.all_stats]

    

    def __repr__(self):

        if not self.count: return ''

        return f"{'train' if self.in_train else 'valid'}: {self.avg_stats}"

    def accumulate(self, run):

        bn = run.xb.shape[0]

        self.tot_loss+=run.loss*bn

        self.count+=bn

        for i, m in enumerate(self.metrics):

            self.tot_mets[i]+=m(run.pred, run.yb)*bn
class AvgStatsCallBack(Callback):

    def __init__(self, metrics):

        self.train_stats, self.valid_stats = AvgStats(metrics, True), AvgStats(metrics, False)

    def begin_epoch(self):

        self.train_stats.reset()

        self.valid_stats.reset()

    def after_loss(self):

        stats = self.train_stats if self.in_train else self.valid_stats

        with torch.no_grad(): stats.accumulate(self.run)

    def after_epoch(self):

        print(self.train_stats)

        print(self.valid_stats)
import matplotlib.pyplot as plt

%matplotlib inline
from sklearn.metrics import roc_auc_score
def roc(out, y):

    score = roc_auc_score(y.cpu().detach().numpy(), out.cpu().detach().numpy())

    return score
class Recorder(Callback):

    def begin_fit(self):

        self.lrs = [[] for _ in self.opt.param_groups]

        self.losses = []

    def after_batch(self):

        if not self.in_train: return

        for pg, lr in zip(self.opt.param_groups, self.lrs): lr.append(pg['lr'])

        self.losses.append(self.loss.detach().cpu())

    

    def plot_lr(self, pgid=-1): plt.plot(self.lrs[pgid])

    def plot_loss(self, skip_last=0): plt.plot(self.losse[:len(self.losses)-skip_last])

    def plot(self, skip_last=0, pgid=-1):

        losses = [o.item() for o in self.losses]

        lrs = self.lrs[pgid]

        n = len(losses)-skip_last

        plt.xscale('log')

        plt.plot(lrs[:n], losses[:n])
class ParamScheduler(Callback):

    _order = 1

    def __init__(self, pname, sched_funcs): self.pname, self.sched_funcs = pname, sched_funcs

    def begin_fit(self):

        if not isinstance(self.sched_funcs, (list, tuple)):

            self.sched_funcs = [self.sched_funcs]*len(self.opt.param_groups)

    def set_param(self):

        assert len(self.opt.param_groups)==len(self.sched_funcs)

        for pg, f in zip(self.opt.param_groups, self.sched_funcs):

            pg[self.pname] = f(self.n_epochs/self.epochs)

    def begin_batch(self):

        if self.in_train: self.set_param()
class LR_Find(Callback):

    _order = 1

    def __init__(self, max_iter=100, min_lr=1e-6, max_lr=10):

        self.max_iter, self.min_lr, self.max_lr = max_iter, min_lr, max_lr

        self.best_loss = 1e9

    def begin_batch(self):

        if not self.in_train: return

        pos = self.n_iter/self.max_iter

        lr = self.min_lr*(self.max_lr/self.min_lr)**pos

        for pg in self.opt.param_groups: pg['lr'] = lr

    def after_step(self):

        if self.n_iter>=self.max_iter or self.loss>self.best_loss*10:

            raise CancelTrainException()

        if self.loss<self.best_loss: self.best_loss = self.loss
run = Runner(cb_funcs=[LR_Find, Recorder])
torch.cuda.empty_cache()
gc.collect()
def annealer(f):

    def _inner(start, end): return partial(f, start, end)

    return _inner
@annealer

def sched_lin(start, end, pos): return start + pos*(end-start)
import math
@annealer

def sched_cos(start, end, pos): return start + (1+math.cos(math.pi*(1-pos)))*(end-start)/2



@annealer

def sched_no(start, end, pos): return start



@annealer

def sched_exp(start, end, pos): return start*(end/start)**pos



torch.Tensor.ndim = property(lambda x: len(x.shape))
def combine_scheds(pcts, scheds):

    assert sum(pcts)==1.

    pcts = torch.tensor([0] + listify(pcts))

    assert torch.all(pcts>=0)

    pcts = torch.cumsum(pcts, 0)

    def _inner(pos):

        idx = (pos>=pcts).nonzero().max()

        actual_pos = (pos-pcts[idx])/(pcts[idx+1]-pcts[idx])

        return scheds[idx](actual_pos)

    return _inner
def pg_dicts(pgs): return [{'params': o} for o in pgs]
def cos_1cycle_anneal(start, high, end):

    return [sched_cos(start, high), sched_cos(high, end)]
from functools import partial
phases = [0.2, 0.8]
scheds = combine_scheds(phases, [sched_cos(1e-4, 5e-3), sched_cos(5e-3, 1e-3)])
from sklearn.metrics import roc_auc_score
def accuracy(out, yb): return (torch.argmax(out, dim=1)==yb.long()).float().mean()
cbfs = [Recorder, partial(AvgStatsCallBack, roc), partial(ParamScheduler, 'lr', scheds)]
run = Runner(cb_funcs=cbfs)
def train_and_eval():

    test_preds = np.zeros(len(test_x))

    for fold, (train_idx, valid_idx) in enumerate(splits):

        print('Fold:', fold)

        torch.cuda.empty_cache()

        learn = get_learner(train_idx, valid_idx)

        gc.collect()

        run = Runner(cb_funcs=cbfs)

        learn.model.train()

        run.fit(4, learn)

        learn.model.eval()

        test_preds_fold = np.zeros(len(test_dl.dataset))

        for i, (x_batch,) in enumerate(test_dl):

            with torch.no_grad():

                y_pred = learn.model(x_batch).detach()

            test_preds_fold[i*batch_size:(i+1)*batch_size] = sigmoid(y_pred.cpu().numpy())[:, 0]

        test_preds+=test_preds_fold/len(splits)

        del(learn)

        gc.collect()

        print(f'Test {fold} added')

    print('Training Completed')

    return test_preds
def sigmoid(x): return 1/(1+np.exp(-x))
preds = train_and_eval()
preds
sub = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/sample_submission.csv', index_col='id')

sub['prediction'] = preds

sub.reset_index(drop=False, inplace=True)

sub.head()
sub.to_csv('submission.csv', index=False)