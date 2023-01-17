from types import SimpleNamespace

from collections import Counter

import os

import re

import pathlib

import array

import pickle

import numpy as np

import torch

import torch.nn as nn

import pandas as pd
DATASET_VERSION = 'ca-100'

COMPETITION_ROOT = '../input/vectors'

DATASET_ROOT = f'../input/cbow-preprocessing/data/{DATASET_VERSION}'

WORKING_ROOT = f'data/{DATASET_VERSION}'

DATASET_PREFIX = 'ca.wiki'
params = SimpleNamespace(

    embedding_dim = 100,

    window_size = 5,

    batch_size = 1000,

    epochs = 4,

    preprocessed = f'{DATASET_ROOT}/{DATASET_PREFIX}',

    working = f'{WORKING_ROOT}/{DATASET_PREFIX}',

    modelname = f'{WORKING_ROOT}/{DATASET_VERSION}.pt',

    train = True

)
try:

    from google.colab import drive

    drive.mount('/content/drive')

    pathlib.Path('/content/drive/My Drive/POE/vectors').mkdir(parents=True, exist_ok=True)

    os.chdir('/content/drive/My Drive/POE/vectors')

except:

    pass
class Vocabulary(object):

    def __init__(self, pad_token='<pad>', unk_token='<unk>', eos_token='<eos>'):

        self.token2idx = {}

        self.idx2token = []

        self.pad_token = pad_token

        self.unk_token = unk_token

        self.eos_token = eos_token

        if pad_token is not None:

            self.pad_index = self.add_token(pad_token)

        if unk_token is not None:

            self.unk_index = self.add_token(unk_token)

        if eos_token is not None:

            self.eos_index = self.add_token(eos_token)



    def add_token(self, token):

        if token not in self.token2idx:

            self.idx2token.append(token)

            self.token2idx[token] = len(self.idx2token) - 1

        return self.token2idx[token]



    def get_index(self, token):

        if isinstance(token, str):

            return self.token2idx.get(token, self.unk_index)

        else:

            return [self.token2idx.get(t, self.unk_index) for t in token]



    def __len__(self):

        return len(self.idx2token)



    def save(self, filename):

        with open(filename, 'wb') as f:

            pickle.dump(self.__dict__, f)



    def load(self, filename):

        with open(filename, 'rb') as f:

            self.__dict__.update(pickle.load(f))
def batch_generator(idata, target, batch_size, shuffle=True):

    nsamples = len(idata)

    if shuffle:

        perm = np.random.permutation(nsamples)

    else:

        perm = range(nsamples)



    for i in range(0, nsamples, batch_size):

        batch_idx = perm[i:i+batch_size]

        if target is not None:

            yield idata[batch_idx], target[batch_idx]

        else:

            yield idata[batch_idx], None
class CBOW(nn.Module):

    def __init__(self, num_embeddings, embedding_dim):

        super().__init__()

        self.emb = nn.Embedding(num_embeddings, embedding_dim, padding_idx=0)

        self.lin = nn.Linear(embedding_dim, num_embeddings, bias=False)



    # B = Batch size

    # W = Number of context words (left + right)

    # E = embedding_dim

    # V = num_embeddings (number of words)

    def forward(self, input):

        # input shape is (B, W)

        e = self.emb(input)

        # e shape is (B, W, E)

        u = e.sum(dim=1)

        # u shape is (B, E)

        v = self.lin(u)

        # v shape is (B, V)

        return v
def load_preprocessed_dataset(prefix):

    # Try loading precomputed vocabulary and preprocessed data files

    token_vocab = Vocabulary()

    token_vocab.load(f'{prefix}.vocab')

    data = []

    for part in ['train', 'valid', 'test']:

        with np.load(f'{prefix}.{part}.npz') as set_data:

            idata, target = set_data['idata'], set_data['target']

            data.append((idata, target))

            print(f'Number of samples ({part}): {len(target)}')

    print("Using precomputed vocabulary and data files")

    print(f'Vocabulary size: {len(token_vocab)}')

    return token_vocab, data
def train(model, criterion, optimizer, idata, target, batch_size, device, log=False):

    model.train()

    total_loss = 0

    ncorrect = 0

    ntokens = 0

    niterations = 0

    for X, y in batch_generator(idata, target, batch_size, shuffle=True):

        # Get input and target sequences from batch

        X = torch.tensor(X, dtype=torch.long, device=device)

        y = torch.tensor(y, dtype=torch.long, device=device)



        model.zero_grad()

        output = model(X)

        loss = criterion(output, y)

        loss.backward()

        optimizer.step()

        # Training statistics

        total_loss += loss.item()

        ncorrect += (torch.max(output, 1)[1] == y).sum().item()

        ntokens += y.numel()

        niterations += 1

        if niterations == 200 or niterations == 500 or niterations % 1000 == 0:

            print(f'Train: wpb={ntokens//niterations}, num_updates={niterations}, accuracy={100*ncorrect/ntokens:.1f}, loss={total_loss/ntokens:.2f}')



    total_loss = total_loss / ntokens

    accuracy = 100 * ncorrect / ntokens

    if log:

        print(f'Train: wpb={ntokens//niterations}, num_updates={niterations}, accuracy={accuracy:.1f}, loss={total_loss:.2f}')

    return accuracy, total_loss
def validate(model, criterion, idata, target, batch_size, device):

    model.eval()

    total_loss = 0

    ncorrect = 0

    ntokens = 0

    niterations = 0

    y_pred = []

    with torch.no_grad():

        for X, y in batch_generator(idata, target, batch_size, shuffle=False):

            # Get input and target sequences from batch

            X = torch.tensor(X, dtype=torch.long, device=device)

            output = model(X)

            if target is not None:

                y = torch.tensor(y, dtype=torch.long, device=device)

                loss = criterion(output, y)

                total_loss += loss.item()

                ncorrect += (torch.max(output, 1)[1] == y).sum().item()

                ntokens += y.numel()

                niterations += 1

            else:

                pred = torch.max(output, 1)[1].detach().to('cpu').numpy()

                y_pred.append(pred)



    if target is not None:

        total_loss = total_loss / ntokens

        accuracy = 100 * ncorrect / ntokens

        return accuracy, total_loss

    else:

        return np.concatenate(y_pred)
# Create working dir

pathlib.Path(WORKING_ROOT).mkdir(parents=True, exist_ok=True)
# Select device

if torch.cuda.is_available():

    device = torch.device('cuda')

else:

    device = torch.device('cpu')

    print("WARNING: Training without GPU can be very slow!")
vocab, data = load_preprocessed_dataset(params.preprocessed)
model = CBOW(len(vocab), params.embedding_dim).to(device)
# 'El Periodico' validation dataset

valid_x_df = pd.read_csv(f'{COMPETITION_ROOT}/x_valid.csv')

tokens = valid_x_df.columns[1:]

valid_x = valid_x_df[tokens].apply(vocab.get_index).to_numpy(dtype='int32')

valid_y_df = pd.read_csv(f'{COMPETITION_ROOT}/y_valid.csv')

valid_y = valid_y_df['token'].apply(vocab.get_index).to_numpy(dtype='int32')
optimizer = torch.optim.Adam(model.parameters())

criterion = nn.CrossEntropyLoss(reduction='sum')



train_accuracy = []

wiki_accuracy = []

valid_accuracy = []

for epoch in range(params.epochs):

    acc, loss = train(model, criterion, optimizer, data[0][0], data[0][1], params.batch_size, device, log=True)

    train_accuracy.append(acc)

    print(f'| epoch {epoch:03d} | train accuracy={acc:.1f}%, train loss={loss:.2f}')

    acc, loss = validate(model, criterion, data[1][0], data[1][1], params.batch_size, device)

    wiki_accuracy.append(acc)

    print(f'| epoch {epoch:03d} | valid accuracy={acc:.1f}%, valid loss={loss:.2f} (wikipedia)')

    acc, loss = validate(model, criterion, valid_x, valid_y, params.batch_size, device)

    valid_accuracy.append(acc)

    print(f'| epoch {epoch:03d} | valid accuracy={acc:.1f}%, valid loss={loss:.2f} (El Periódico)')



# Save model

torch.save(model.state_dict(), params.modelname)
# 'El Periodico' test dataset

valid_x_df = pd.read_csv(f'{COMPETITION_ROOT}/x_test.csv')

test_x = valid_x_df[tokens].apply(vocab.get_index).to_numpy(dtype='int32')

y_pred = validate(model, None, test_x, None, params.batch_size, device)

y_token = [vocab.idx2token[index] for index in y_pred]
submission = pd.DataFrame({'id':valid_x_df['id'], 'token': y_token}, columns=['id', 'token'])

print(submission.head())

submission.to_csv('submission.csv', index=False)