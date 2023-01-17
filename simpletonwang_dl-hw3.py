# Basic Function

import numpy as np 

import pandas as pd 

import os

import spacy

import string

import re

import numpy as np

from spacy.symbols import ORTH

from collections import Counter

from tqdm import tqdm_notebook



# Keras for text preprocessing

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences



# Pytorch

import torch

import torch.nn as nn

import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence 
train = pd.read_csv('../input/quora-question-pairs/train.csv')#.fillna('something')

test = pd.read_csv('../input/quora-question-pairs/test.csv')#.fillna('something')
train.head()
test.head()
# note that spacy_tok takes a while run it just once

def encode_sentence(path, vocab2index, N=400, padding_start=True):

    x = spacy_tok(path.read_text())

    enc = np.zeros(N, dtype=np.int32)

    enc1 = np.array([vocab2index.get(w, vocab2index["UNK"]) for w in x])

    l = min(N, len(enc1))

    if padding_start:

        enc[:l] = enc1[:l]

    else:

        enc[N-l:] = enc1[:l]

    return enc, l
re_br = re.compile(r'<\s*br\s*/?>', re.IGNORECASE)

def sub_br(x): return re_br.sub("\n", x)



my_tok = spacy.load('en')

def spacy_tok(x): return [tok.text for tok in my_tok.tokenizer(sub_br(x))]
# counts = Counter()

# for text in tqdm_notebook(train.question1):

#     counts.update(spacy_tok(text))

# for text in tqdm_notebook(train.question2):

#     counts.update(spacy_tok(text))

# for text in tqdm_notebook(test.question1):

#     counts.update(spacy_tok(text))

# for text in tqdm_notebook(teat.question2):

#     counts.update(spacy_tok(text))
MAX_LEN=60

WORD_NUM = 180000
train_q1 = train["question1"].fillna("something").values

train_q2 = train["question2"].fillna("something").values



test_q1 = test["question1"].fillna("something").values

test_q2 = test["question2"].fillna("something").values



tokenizer = Tokenizer(num_words=MAX_LEN, filters='!"#$%&()*+,-./:;<=>@[\\]^_`{|}~\t\n')

tokenizer.fit_on_texts(list(train_q1)+list(train_q2)+list(test_q1)+list(test_q2))
# train_q1_seq = tokenizer.texts_to_sequences(train_q1)

# train_q2_seq = tokenizer.texts_to_sequences(train_q2)

# test_q1_seq = tokenizer.texts_to_sequences(test_q1)

# test_q2_seq = tokenizer.texts_to_sequences(test_q2)

# train_q1_seq = pad_sequences(train_q1_seq, maxlen=MAX_LEN)

# train_q2_seq = pad_sequences(train_q2_seq, maxlen=MAX_LEN)

# test_q1_seq = pad_sequences(test_q1_seq, maxlen=MAX_LEN)

# test_q2_seq = pad_sequences(test_q2_seq, maxlen=MAX_LEN)
word_index = tokenizer.word_index
def load_glove(word_index):

    EMBEDDING_FILE = '../input/glove840b300dtxt/glove.840B.300d.txt'

    def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')[:300]

    embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE) if o.split(" ")[0] in word_index)

    

#     all_embs = np.stack(embeddings_index.values())

    emb_mean,emb_std = -0.005838499,0.48782197

#     embed_size = all_embs.shape[1]



    # word_index = tokenizer.word_index

    nb_words = min(WORD_NUM, len(word_index))

#     embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))

    embedding_matrix = np.random.normal(emb_mean, 0, (nb_words, 300))

    for word, i in tqdm_notebook(word_index.items()):

        if i >= WORD_NUM: continue

        embedding_vector = embeddings_index.get(word)

        if embedding_vector is not None: embedding_matrix[i] = embedding_vector

            

    return embedding_matrix 
glove_embeddings = load_glove(word_index)
class QuoraDataset(Dataset):

    def __init__(self, df, tokenizer, max_len=60, train=True):

        self.train = train

        self.q1 = df['question1'].fillna('something').values

        self.q2 = df['question2'].fillna('something').values

        

        self.q1 = tokenizer.texts_to_sequences(self.q1)

        self.q2 = tokenizer.texts_to_sequences(self.q2)

        

        self.q1 = pad_sequences(self.q1, maxlen=max_len)

        self.q2 = pad_sequences(self.q2, maxlen=max_len)

            

        if train:

            self.y = df['is_duplicate'].values

        

    def __len__(self):

        return len(self.q1)

    

    def __getitem__(self, idx):

        if self.train:

            return self.q1[idx], self.q2[idx], self.y[idx]

        else:

            return self.q1[idx], self.q2[idx]
from sklearn.model_selection import train_test_split
train_df, valid_df = train_test_split(train, test_size=0.2)
train_ds = QuoraDataset(train_df, tokenizer, train=True)

valid_ds = QuoraDataset(valid_df, tokenizer, train=True)

test_ds = QuoraDataset(test, tokenizer, train=False)
TRAIN_BS = 128

TEST_BS = 2048
train_dl = DataLoader(train_ds, batch_size=TRAIN_BS, shuffle=True)

valid_dl = DataLoader(valid_ds, batch_size=TEST_BS, shuffle=False)

test_dl = DataLoader(test_ds, batch_size=TEST_BS, shuffle=False)
class NeuralNet(nn.Module):

    def __init__(self, embedding_matrix, hidden_size, max_features=18000, embed_size=300):

        super(NeuralNet, self).__init__()

        

        self.embedding = nn.Embedding(max_features, embed_size)

        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))

        self.embedding.weight.requires_grad = False



        self.lstm1 = nn.LSTM(embed_size, hidden_size, bidirectional=True, batch_first=True)

        self.lstm2 = nn.LSTM(embed_size, hidden_size, bidirectional=True, batch_first=True)

        

        self.linear = nn.Linear(4*hidden_size, 2*16)

        self.relu = nn.ELU()

        self.dropout = nn.Dropout(0.1)

        

        self.out = nn.Linear(2*16, 1)

        

    def forward(self, q1, q2):

        q1_embedding = self.embedding(q1)

        q2_embedding = self.embedding(q2)

        

        q1_lstm, (h1, _) = self.lstm1(q1_embedding)

        q2_lstm, (h2, _) = self.lstm2(q2_embedding)

        

#         print(q1_lstm.shape)

#         print(q2_lstm.shape)

#         print(h1.shape)

#         print(h2.shape)

#         print(torch.mean(q1_lstm,dim=1).shape)

    

        

#         avg_pool = torch.mean(h1, 1)

#         max_pool, _ = torch.max(h2, 1)

        

        q1 = torch.cat((h1[0], h1[1]), 1)

        q2 = torch.cat((h2[0], h2[1]), 1)

        

#         q1 = self.linear(q1)

#         q2 = self.linear(q2)

        

        

        conc = self.relu(self.linear(torch.cat([q1,q2],dim=1)))

        conc = self.dropout(conc)

        out = self.out(conc)

        

        return out
def val_metrics(model, valid_dl):

    model.eval()

    correct = 0

    total = 0

    sum_loss = 0.0

    for q1, q2, y in valid_dl:

        q1 = q1.long().cuda()

        q2 = q2.long().cuda()

        y = y.float().cuda().unsqueeze(1)

        y_hat = model(q1, q2)

        loss = F.binary_cross_entropy_with_logits(y_hat, y)

        y_pred = y_hat > 0

        correct += (y_pred.float() == y).float().sum()

        total += y.shape[0]

        sum_loss += loss.item()*y.shape[0]

    return sum_loss/total, correct/total
def train_epocs(model, epochs=10, lr=0.001):

    parameters = filter(lambda p: p.requires_grad, model.parameters())

    optimizer = torch.optim.Adam(parameters, lr=lr)

    for i in range(epochs):

        model.train()

        sum_loss = 0.0

        total = 0

        for q1,q2, y in tqdm_notebook(train_dl):

            q1 = q1.long().cuda()

            q2 = q2.long().cuda()

            y = y.float().cuda()

            y_pred = model(q1, q2)

            optimizer.zero_grad()

            loss = F.binary_cross_entropy_with_logits(y_pred, y.unsqueeze(1))

            loss.backward()

            optimizer.step()

            sum_loss += loss.item()*y.shape[0]

            total += y.shape[0]

        val_loss, val_acc = val_metrics(model, valid_dl)

#         if i % 1 == 1:

        print("train loss %.3f val loss %.3f and val accuracy %.3f" % (sum_loss/total, val_loss, val_acc))
model = NeuralNet(glove_embeddings,hidden_size=128).cuda()
train_epocs(model, epochs=10, lr=0.01)
def sigmoid(x):

    return 1 / (1 + np.exp(-x))
test.head()
next(iter(test_dl))
test_preds = np.zeros(len(test_ds))

for i, (q1, q2) in enumerate(test_dl):

    q1 = q1.long().cuda()

    q2 = q2.long().cuda()

    y_pred = model(q1, q2).detach()

    test_preds[i * TEST_BS:(i+1) * TEST_BS] = sigmoid(y_pred.cpu().numpy())[:, 0]
submission = pd.read_csv('../input/quora-question-pairs/sample_submission.csv')
submission['is_duplicate'] = test_preds
submission.to_csv('submission.csv', index=False)