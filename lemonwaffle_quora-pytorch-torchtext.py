import torch
import torch.nn as nn
import torch.optim as optim

from torchtext import data
from torchtext import datasets
from torchtext import vocab

import random
import time

import numpy as np
import pandas as pd

from sklearn.metrics import f1_score

# To run experiments deterministically
SEED = 42

torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
# For kaggle
import os
print(os.listdir("../input"))
TRAIN_PATH = '../input/train.csv'
TEST_PATH = '../input/test.csv'
GLOVE_PATH = '../input/embeddings/glove.840B.300d/glove.840B.300d.txt'
GLOVE = 'glove.840B.300d.txt'
print(os.listdir("../input/embeddings/glove.840B.300d"))
# Pretrained embedding to use
EMBEDDING_PATH = GLOVE_PATH
# Should we limit the vocab size?
# (A) 120000, 95000
MAX_SIZE = 120000
# (A) Should we limit number of words in a sentence?
MAX_LEN = 70

# Split ratio for test/valid
SPLIT_RATIO = 0.9

# (A)
BATCH_SIZE = 512

# Model parameters
# (A) Could be lesser I think.
HIDDEN_DIM = 32
# (A)
N_LAYERS = 2
BIDIRECTIONAL = True
# (C)
DROPOUT = 0.5
# Defining the Fields for our dataset
# Skipping the id column
ID = data.Field()
TEXT = data.Field(tokenize='spacy')
TARGET = data.LabelField(dtype=torch.float)

train_fields = [('id', None), ('text', TEXT), ('target', TARGET)]
test_fields = [('id', ID), ('text', TEXT)]

# Creating our train and test data
train_data = data.TabularDataset(
    path=TRAIN_PATH,
    format='csv',
    skip_header=True,
    fields=train_fields
)

test_data = data.TabularDataset(
    path=TEST_PATH,
    format='csv',
    skip_header=True,
    fields=test_fields
)

# Create validation dataset (default 70:30 split)
train_data, valid_data = train_data.split(split_ratio=SPLIT_RATIO, random_state=random.seed(SEED))
print(f'Number of training examples: {len(train_data)}')
print(f'Number of validation examples: {len(valid_data)}')
print(f'Number of test examples: {len(test_data)}')
# One training example
vars(train_data.examples[0])
# Importing the pretrained embedding
vec = vocab.Vectors(EMBEDDING_PATH)

# Build the vocabulary using only the train dataset?,
# and also by specifying the pretrained embedding
TEXT.build_vocab(train_data, vectors=vec, max_size=MAX_SIZE)
TARGET.build_vocab(train_data)
ID.build_vocab(test_data)
print(f'Unique tokens in TEXT vocab: {len(TEXT.vocab)}')
print(f'Unique tokens in TARGET vocab: {len(TARGET.vocab)}')
TEXT.vocab.vectors.shape
# Might have some confusion as to how the batch iterators are defined
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Automatically shuffles and buckets the input sequences into
# sequences of similar length
train_iter, valid_iter = data.BucketIterator.splits(
    (train_data, valid_data),
    sort_key=lambda x: len(x.text), # what function/field to use to group the data
    batch_size=BATCH_SIZE,
    device=device
)

# Don't want to shuffle test data, so use a standard iterator
test_iter = data.Iterator(
    test_data,
    batch_size=BATCH_SIZE,
    device=device,
    train=False,
    sort=False,
    sort_within_batch=False
)
class RNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim,
                 n_layers, bidirectional, dropout):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers,
                           bidirectional=bidirectional, dropout=dropout)
        # Final hidden state has both forward and backward components
        self.fc = nn.Linear(hidden_dim*2, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x: [seq_length, batch_size]
        embedded = self.dropout(self.embedding(x))
        # embedded: [seq_length, batch_size, emb_dim]
        
        output, (hidden, cell) = self.rnn(embedded)
        # output: [seq_length, batch_size, hid_dim * num_directions]
        # hidden: [num_layers * num_directions, batch_size, hid_dim]
        # cell:
        
        # Concat the final forward and backward hidden layers
        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1))
        # hidden: [batch_size, hid_dim * num_directions]
        
        return self.fc(hidden.squeeze(0))
        # return: [batch_size, 1]
emb_shape = TEXT.vocab.vectors.shape
INPUT_DIM = emb_shape[0]
EMBEDDING_DIM = emb_shape[1]
OUTPUT_DIM = 1

model = RNN(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, N_LAYERS, BIDIRECTIONAL, DROPOUT)
pretrained_embeddings = TEXT.vocab.vectors
pretrained_embeddings.shape
model.embedding.weight.data.copy_(pretrained_embeddings)
optimizer = optim.Adam(model.parameters())
criterion = nn.BCEWithLogitsLoss()

model = model.to(device)
criterion = criterion.to(device)
def train(model, iterator, optimizer, criterion):
    # Track the loss
    epoch_loss = 0
    
    model.train()
    
    for batch in iterator:
        optimizer.zero_grad()
        
        predictions = model(batch.text).squeeze(1)
        loss = criterion(predictions, batch.target)
        
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        
    return epoch_loss / len(iterator)
def evaluate(model, iterator, criterion):
    epoch_loss = 0
    
    model.eval()
    
    with torch.no_grad():
        for batch in iterator:
            predictions = model(batch.text).squeeze(1)
            loss = criterion(predictions, batch.target)

            epoch_loss += loss.item()
        
    return epoch_loss / len(iterator)
N_EPOCHS = 6

# Track time taken
start_time = time.time()

for epoch in range(N_EPOCHS):
    epoch_start_time = time.time()
    
    train_loss = train(model, train_iter, optimizer, criterion)
    valid_loss = evaluate(model, valid_iter, criterion)
    
    print(f'| Epoch: {epoch+1:02} '
          f'| Train Loss: {train_loss:.3f} '
          f'| Val. Loss: {valid_loss:.3f} '
          f'| Time taken: {time.time() - epoch_start_time:.2f}s'
          f'| Time elapsed: {time.time() - start_time:.2f}s')
# Use validation dataset
valid_pred = []
valid_truth = []
    
model.eval()
    
with torch.no_grad():
    for batch in valid_iter:
        valid_truth += batch.target.cpu().numpy().tolist()
        predictions = model(batch.text).squeeze(1)
        valid_pred += torch.sigmoid(predictions).cpu().data.numpy().tolist()
tmp = [0,0,0] # idx, cur, max
delta = 0
for tmp[0] in np.arange(0.1, 0.501, 0.01):
    tmp[1] = f1_score(valid_truth, np.array(valid_pred)>tmp[0])
    if tmp[1] > tmp[2]:
        delta = tmp[0]
        tmp[2] = tmp[1]
print('best threshold is {:.4f} with F1 score: {:.4f}'.format(delta, tmp[2]))
test_pred = []
test_id = []

model.eval()

with torch.no_grad():
    for batch in test_iter:
        predictions = model(batch.text).squeeze(1)
        test_pred += torch.sigmoid(predictions).cpu().data.numpy().tolist()
        test_id += batch.id.view(-1).cpu().numpy().tolist()
test_pred = (np.array(test_pred) >= delta).astype(int)
test_id = [ID.vocab.itos[i] for i in test_id]
submission = pd.DataFrame({'qid': test_id, 'prediction': test_pred})
submission.to_csv('submission.csv', index=False)