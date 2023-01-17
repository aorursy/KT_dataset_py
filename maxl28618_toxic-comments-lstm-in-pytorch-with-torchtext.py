import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import torch

import torch.nn as nn

from torchtext import data

from torchtext import datasets

from torchtext.data import Field

from torchtext.data import Iterator, BucketIterator

import torch.optim as optim

import os

print(os.listdir("../input"))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BATCH_SIZE = 64

MAX_VOCAB_SIZE = 25_000
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

sample = pd.read_csv('../input/sample_submission.csv')
small = train[:-10000]

valid = train[-10000:]

small.to_csv('small.csv', index=False)

valid.to_csv('valid.csv', index=False)
SEED = 1234



torch.manual_seed(SEED)

torch.backends.cudnn.deterministic = True



tokenize = lambda x: x.split()

TEXT = Field(sequential=True, tokenize=tokenize, lower=True, include_lengths=True)

LABEL = Field(sequential=False, use_vocab=False, dtype=torch.float)
from torchtext.data import TabularDataset

 

train_datafields = [(col, TEXT) if col == 'comment_text' else 

                              (col, LABEL) if col == 'target' else 

                              (col, None) 

                              for col in train.columns]

train_data, valid_data = TabularDataset.splits(

            path='',

            train='small.csv',

            validation='valid.csv',

            format='csv',

            skip_header=True,

            fields=train_datafields)



test_datafields = [('id', None), ('comment_text', TEXT)]

test_data = TabularDataset(

            path="../input/test.csv",

            format='csv',

            skip_header=True,

            fields=test_datafields)
print(f'Number of training examples: {len(train_data)}')

print(f'Number of validation examples: {len(valid_data)}')

print(f'Number of test examples: {len(test_data)}')
TEXT.build_vocab(train_data, max_size = MAX_VOCAB_SIZE)

LABEL.build_vocab(train_data)
train_iter, val_iter = BucketIterator.splits(

    (train_data, valid_data),

    batch_sizes=(BATCH_SIZE, BATCH_SIZE),

    device=device,

    sort_key=lambda x: len(x.comment_text),

    sort_within_batch=True,

    repeat=False

)



test_iter = Iterator(test_data, batch_size=1, device=device, sort=False, sort_within_batch=False, repeat=False)
# Kudos to http://mlexplained.com/2018/02/08/a-comprehensive-tutorial-to-torchtext/



class BatchWrapper:

    def __init__(self, iterator, x_var, y_vars):

        self.iterator, self.x_var, self.y_vars = iterator, x_var, y_vars

  

    def __iter__(self):

        for batch in self.iterator:

            x = getattr(batch, self.x_var)

            if self.y_vars is not None:

                y = torch.cat([getattr(batch, feat).unsqueeze(1) for feat in self.y_vars], dim=1).float()

            else:

                y = torch.zeros((1))

            yield (x, y)

    def __len__(self):

        return len(self.iterator)



train_loader = BatchWrapper(train_iter, "comment_text", ["target"])

valid_loader = BatchWrapper(val_iter, "comment_text", ["target"])

test_loader = BatchWrapper(test_iter, "comment_text", None)
class RNN(nn.Module):

    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, 

                 bidirectional, dropout, pad_idx):

        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx = pad_idx)

        self.rnn = nn.LSTM(embedding_dim, 

                           hidden_dim, 

                           num_layers=n_layers, 

                           bidirectional=bidirectional, 

                           dropout=dropout)

        self.fc = nn.Linear(hidden_dim * 2, output_dim)

        self.dropout = nn.Dropout(dropout)

 



    def forward(self, text, text_lengths):

        embedded = self.dropout(self.embedding(text))

        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths)

        packed_output, (hidden, cell) = self.rnn(packed_embedded)

        output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output)

        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1))

        return self.fc(hidden.squeeze(0))
INPUT_DIM = len(TEXT.vocab)

EMBEDDING_DIM = 100

HIDDEN_DIM = 256

OUTPUT_DIM = 1

N_LAYERS = 2

BIDIRECTIONAL = True

DROPOUT = 0.5

PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]
model = RNN(INPUT_DIM, 

            EMBEDDING_DIM, 

            HIDDEN_DIM, 

            OUTPUT_DIM, 

            N_LAYERS, 

            BIDIRECTIONAL, 

            DROPOUT, 

            PAD_IDX)
optimizer = optim.Adam(model.parameters())

loss_func = nn.BCEWithLogitsLoss()
def count_parameters(model):

    return sum(p.numel() for p in model.parameters() if p.requires_grad)



print(f'The model has {count_parameters(model):,} trainable parameters')
model = model.to(device)

loss_func = loss_func.to(device)
def train_model(model, data_loader, optimizer, loss_func):

    epoch_loss = 0

    

    model.train()

    

    for x, y in data_loader:

        optimizer.zero_grad()

        text, text_lengths = x

        preds = model(text, text_lengths)

        loss = loss_func(preds, y)

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(data_loader)
def validate_model(model, data_loader, loss_func):

    val_loss = 0.0

    model.eval()

    for x, y in data_loader:

        text, text_lengths = x

        preds = model(text, text_lengths)

        loss = loss_func(preds, y)

        val_loss += loss.item()

    return val_loss / len(data_loader)
epochs = 1



best_valid_loss = float('inf')

best_epoch = 0

        

for epoch in range(1, epochs + 1):

    epoch_loss = train_model(model, train_loader, optimizer, loss_func)

    val_loss = validate_model(model, valid_loader, loss_func)

    if val_loss < best_valid_loss:

        best_valid_loss = val_loss

        best_epoch = epoch

        print(f'Best validation loss!! {best_valid_loss}')

        torch.save(model.state_dict(), 'toxic_model.pt')

    print(f'Epoch: {epoch}, Training Loss: {epoch_loss:.4f}, Validation Loss: {val_loss:.4f}')
print(f'Best validation loss at epoch = {best_epoch}')

model.load_state_dict(torch.load('toxic_model.pt'))

test_preds = []

for i, tup in enumerate(test_loader):

    if i % 1000 == 0:

        print(f'Progress = {i/len(test_loader):.2%}')

    x, y = tup

    text, text_lengths = x

    preds = model(text, text_lengths)

    preds = preds.view(x[0].shape[1])

    preds = preds.data.cpu().numpy()

    preds = 1 / (1 + np.exp(-preds))

    test_preds.append(preds)

test_preds = np.hstack(test_preds)
submission = pd.read_csv('../input/test.csv')

submission.loc[:, 'prediction'] = test_preds

submission.drop('comment_text', axis=1).to_csv('submission.csv', index=False)
submission