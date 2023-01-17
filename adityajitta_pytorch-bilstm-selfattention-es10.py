# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.



%matplotlib inline

import os, sys

import re

import string

import pathlib

import random

from collections import Counter, OrderedDict

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import nltk

import spacy

from tqdm import tqdm, tqdm_notebook, tnrange

tqdm.pandas(desc='Progress')

import torch

import torch.nn as nn

import torch.optim as optim

import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from torch.autograd import Variable

from sklearn.model_selection import StratifiedShuffleSplit, train_test_split

from sklearn.metrics import accuracy_score, f1_score

import warnings

warnings.filterwarnings('ignore')

from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity='all'

from sklearn.model_selection import train_test_split

from sklearn.utils import class_weight



# Build Pytorch DataLoader Class (will change it to torchtext)

class QuestionDataLoader(Dataset):

    def __init__(self, df, word2idx, nlp, is_test=False, maxlen=70):

        self.maxlen = maxlen

        self.word2idx = word2idx

        self.nlp = nlp

        self.is_test = is_test

        self.df = df #pd.read_csv(df_path, error_bad_lines=False)

        self.df['question_text'] = self.df.question_text.apply(lambda x: x.strip())

        print('Indexing...')

        self.df['question_ids'] = self.df.question_text.progress_apply(self.indexer)

        print('Calculating lengths')

        self.df['lengths'] = self.df.question_ids.progress_apply(lambda x: self.maxlen if len(x) > self.maxlen else len(x))

        print('Padding')

        self.df['question_padded'] = self.df.question_ids.progress_apply(self.pad_data)



    @classmethod

    def fromfilename(cls, name, word2idx, nlp, is_test=False):

        return cls(pd.read_csv(name, error_bad_lines=False), word2idx, nlp,  is_test)

    

    def __len__(self):

        return self.df.shape[0]

    

    

    def __getitem__(self, idx):

        X = self.df.question_padded[idx]

        lens = self.df.lengths[idx]

        qids = self.df.qid[idx]

        if not self.is_test:

            y = self.df.target[idx]

        else:

            return X, lens, qids

        return X,y,lens

    

    def pad_data(self, s):

        padded = np.zeros((self.maxlen,), dtype=np.int64)

        if len(s) > self.maxlen: 

            padded[:] = s[:self.maxlen]

        else:

            padded[:len(s)] = s

        return padded

    

    def indexer(self, s):

        return [self.word2idx[w.text.lower()] for w in self.nlp(s)]

        

# Build Model RNN (GRU)



class AttentionBiLSTM(nn.Module):

    def __init__(self, vocab_size, embedding_dim, embedding_matrix, n_hidden, n_out):

        super().__init__()

        self.vocab_size, self.embedding_dim, self.embedding_matrix, self.n_hidden, self.n_out = vocab_size, embedding_dim, embedding_matrix, n_hidden, n_out

        self.emb = nn.Embedding(self.vocab_size, self.embedding_dim)

        self.emb.weight = nn.Parameter(torch.tensor(self.embedding_matrix, dtype=torch.float32))

        self.emb.weight.requires_grad = False

        self.dropout = 0.8

        self.relu = nn.ReLU()

        self.bilstm = nn.LSTM(self.embedding_dim, self.n_hidden, dropout=self.dropout, bidirectional=True)

        self.W_s1 = nn.Linear(2*self.n_hidden, 350)

        self.W_s2 = nn.Linear(350, 30)

        self.fc_layer = nn.Linear(30*2*self.n_hidden, 500)

        self.label = nn.Linear(500, self.n_out)

    

    def attn_bahdanau(self, lstm_output):

        attn_weight_matrix = self.W_s2(F.tanh(self.W_s1(lstm_output)))

        # batch, num_seq, 30

        attn_weight_matrix = attn_weight_matrix.permute(0, 2, 1)

        # batch, 30, num_seq

        attn_weight_matrix = F.softmax(attn_weight_matrix, dim=2)

        return attn_weight_matrix

    

    def forward(self, seq, lengths, gpu=True):

        bs = seq.size(1) # batch size

        h0, c0 = self.init_hidden(bs, gpu) # initialize hidden state of GRU

        embs = self.emb(seq)

        embs = pack_padded_sequence(embs, lengths) # unpad

        output, (hn, cn) = self.bilstm(embs, (h0, c0)) # 

        output, lengths = pad_packed_sequence(output) # pad the sequence to the max length in the batch

        

        output = output.permute(1, 0, 2)

        # batch, 30, num_seq # batch, num_seq, 2*hidden

        attn_matrix = self.attn_bahdanau(output)

        # batch, num_seq, 2*hidden

        hidden_matrix = torch.bmm(attn_matrix, output)

        # hidden_matrix.size() = (batch_size, 30, 2*hidden_size)

        # Let's now concatenate the hidden_matrix and connect it to the fully connected layer.

        fc_out = self.fc_layer(hidden_matrix.view(-1, hidden_matrix.size()[1]*hidden_matrix.size()[2]))

        fc_out = self.relu(fc_out)

        outp = self.label(fc_out) # 

        return F.log_softmax(outp, dim=-1)

    

    def init_hidden(self, batch_size, gpu):

        if gpu: return (Variable(torch.zeros((2,batch_size,self.n_hidden)).cuda()), Variable(torch.zeros((2,batch_size,self.n_hidden)).cuda()))

        else: return Variable(torch.zeros((2,batch_size,self.n_hidden)))
# Utils





def eval_model(model, data_iter, criterion, device):

    model.eval()

    with torch.no_grad():

        y_true_eval = list()

        y_pred_eval = list()

        ind_ids =list()

        for X, y, lengths in data_iter:

            X,y,lengths, indx = sort_batch(X,y,lengths)

            X = Variable(X.cuda())

            pred = model(X, lengths, gpu=True)

            pred_idx = torch.max(pred, dim=1)[1]

            y_pred_eval += list(pred_idx.cpu().data.numpy())

            y_true_eval += list(y.cpu().data.numpy())

            ind_ids += list(indx.cpu().numpy())

        eval_acc = f1_score(y_true_eval, y_pred_eval)

        print(f'Accuracy on eval set: {eval_acc}')

        return y_pred_eval, y_true_eval, ind_ids





def predict(model, test_iter, criterion, device):

    model.eval()

    with torch.no_grad():

        y_pred_test = list()

        ind_ids =list()

        X_temp =list()

        for X, lengths, qids in test_iter:

            X,lengths, qids = sort_batch_test(X,lengths, np.array(qids))

            X = Variable(X.cuda())

            pred = model(X, lengths, gpu=True)

            pred_idx = torch.max(pred, dim=1)[1]

            y_pred_test += list(pred_idx.cpu().data.numpy())

            ind_ids += list(np.array(qids))

            X = X.transpose(1,0)

            X_temp += list(X.cpu().numpy())

        return np.array(y_pred_test), np.array(ind_ids), X_temp



def sort_batch(X, y, lengths):

    lengths, indx = lengths.sort(dim=0, descending=True)

    X = X[indx]

    y = y[indx]

    return X.transpose(0,1), y, lengths, indx



def sort_batch_test(X, lengths, qids):

    lengths, indx = lengths.sort(dim=0, descending=True)

    X = X[indx]

    return X.transpose(0,1), lengths, qids[indx]



def fit(model, train_dl, val_dl, loss_fn, opt, epochs=3):

    num_batch = len(train_dl)

    best_loss = None

    for epoch in tnrange(epochs):      

        y_true_train = list()

        y_pred_train = list()

        total_loss_train = 0

        

        if val_dl:

            y_true_val = list()

            y_pred_val = list()

            total_loss_val = 0

        

        t = tqdm_notebook(iter(train_dl), leave=False, total=num_batch)

        count = 0

        for X,y, lengths in t:

            t.set_description(f'Epoch {epoch}')

            X,y,lengths, indx = sort_batch(X,y,lengths)

            X = Variable(X.cuda())

            y = Variable(y.cuda())

            lengths = lengths.numpy()

            count += 1

            opt.zero_grad()

            pred = model(X, lengths, gpu=True)

            loss = loss_fn(pred, y)

            #if count % 100 == 0:

            #    import pdb;pdb.set_trace()

            loss.backward()

            opt.step()

            #import pdb;pdb.set_trace()

            t.set_postfix(loss=loss.data.item())

            pred_idx = torch.max(pred, dim=1)[1]#torch.max(pred, dim=1)[1]

            

            y_true_train += list(y.cpu().data.numpy())

            y_pred_train += list(pred_idx.cpu().data.numpy())

            total_loss_train += loss.data.item()

            

        train_acc = f1_score(y_true_train, y_pred_train)

        train_loss = total_loss_train/len(train_dl)

        print(f' Epoch {epoch}: Train loss: {train_loss} acc: {train_acc}')

        

        if val_dl:

            for X,y,lengths in tqdm_notebook(val_dl, leave=False):

                X, y,lengths, idx = sort_batch(X,y,lengths)

                X = Variable(X.cuda())

                y = Variable(y.cuda())

                pred = model(X, lengths.numpy())

                loss = loss_fn(pred, y)

                pred_idx = torch.max(pred, 1)[1]#torch.max(pred, 1)[1]

                y_true_val += list(y.cpu().data.numpy())

                y_pred_val += list(pred_idx.cpu().data.numpy())

                total_loss_val += loss.data.item()

            valacc = f1_score(y_true_val, y_pred_val)

            valloss = total_loss_val/len(val_dl)

            print(best_loss, valloss)

            if (epoch > 2) and valloss > best_loss:

                print(f'Val loss: {valloss} acc: {valacc}')

                break

            best_loss = valloss

            print(f'Val loss: {valloss} acc: {valacc}')

            



def main(sample = False):

    data_root = pathlib.Path('../input')

    df_train = pd.read_csv(data_root/'train.csv', error_bad_lines=False)

    df_test = pd.read_csv(data_root/'test.csv', error_bad_lines=False)

    

    if sample:

        df_train = df_train[:50000]

    

    df_train.question_text.progress_apply(lambda x: x.strip())

    df_test.question_text.progress_apply(lambda x: x.strip())

    

    # Construct vaobaulary

    nlp = spacy.load('en',disable=['parser', 'tagger', 'ner'])

    words = Counter()

    for sent in tqdm_notebook(df_train.question_text.values):

        words.update(w.text.lower() for w in nlp(sent))

    for sent in tqdm_notebook(df_test.question_text.values):

        words.update(w.text.lower() for w in nlp(sent))

    words = sorted(words, key=words.get, reverse=True)

    words = ['_PAD','_UNK'] + words

    word2idx = {o:i for i,o in enumerate(words)}

    idx2word = {i:o for i,o in enumerate(words)}

    def indexer(s): return [word2idx[w.text.lower()] for w in nlp(s)]

    

    train_df, other_df = train_test_split(df_train, test_size=0.2)

    val_df, eval_df = train_test_split(other_df, test_size=0.5)

    

    

    # Build train, val, eval and test Datasets

    dtrain = QuestionDataLoader(train_df.reset_index(drop=True), word2idx, nlp)

    dval = QuestionDataLoader(val_df.reset_index(drop=True), word2idx, nlp)

    deval = QuestionDataLoader(eval_df.reset_index(drop=True), word2idx, nlp)

    dtest = QuestionDataLoader.fromfilename(data_root/'test.csv', word2idx, nlp, True)

    dl_train = DataLoader(

        dtrain,

        batch_size=256,

        drop_last=True)

    dl_val = DataLoader(

        dval,

        batch_size=256,

        drop_last=True)

    dl_eval = DataLoader(

        deval,

        batch_size=256,

        drop_last=True)

    dl_test = DataLoader(dtest, batch_size= 256)#len(df_test))

    

    

    # Load Embeddings

    print("Loading embeddings")

    maxlen = 120000

    embed_size = 300

    embedding_path = "../input/embeddings/wiki-news-300d-1M/wiki-news-300d-1M.vec"

    def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')

    

    nb_words = max(maxlen, len(word2idx))

    emb_mean,emb_std = -0.0033469985, 0.109855495

    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))

    

    if not sample:

        embedding_index = dict(get_coefs(*o.split(" ")) for o in open(embedding_path, encoding='utf-8', errors='ignore') if len(o)>100)

        for word, i in word2idx.items():

            if i >= nb_words:

                continue

            embedding_vector = embedding_index.get(word)

            if embedding_vector is not None:

                embedding_matrix[i] = embedding_vector

    print("Loaded embeddings")

    # Model Params

    vocab_size = nb_words#max(nb_words, len(words))

    embedding_dim = 300

    n_hidden = 200

    n_out = 2

    num_epochs = 10

    lr_rate = 3e-4

    # Start Train

    model_bilstm = AttentionBiLSTM(vocab_size, embedding_dim, embedding_matrix, n_hidden, n_out).cuda()

    opt = optim.Adam(model_bilstm.parameters(),  lr_rate)

    #fit(model=model_bilstm, train_dl=dl_train, val_dl=dl_val, loss_fn=F.nll_loss, opt=opt, epochs=num_epochs)

    fit(model=model_bilstm, train_dl=dl_train, val_dl=dl_val, loss_fn=F.nll_loss, opt=opt, epochs=num_epochs)

    # Accuracy on Eval set

    _, _, _ = eval_model(model_bilstm, dl_eval, F.nll_loss, torch.device("cuda") )

    

    # Predict on test-set

    dl_test = DataLoader(dtest, batch_size= 256)

    test_pred, test_ids, X_temp = predict(model_bilstm, dl_test, F.nll_loss, torch.device("cuda") )

    print("Predict Done")

    # Prepare submission

    sub = pd.read_csv('../input/sample_submission.csv')

    new_sub = pd.DataFrame()

    #new_sub = sub.reindex(test_ids)

    new_sub["qid"] = test_ids

    new_sub["prediction"] = test_pred

    #new_sub.sort_index(inplace=True)

    new_sub.to_csv("submission.csv", index=False)
main()
# weights = [0.4, 1]

# class_weights=torch.FloatTensor(weights).cuda()

# learn.crit = nn.CrossEntropyLoss(weight=class_weights)

#dtest = QuestionDataLoader.fromfilename(data_root/'test.csv', word2idx, nlp, True)

#X,lengths, indx = sort_batch_test(X,lengths)

#dl_test = DataLoader(dtest, batch_size= 256)