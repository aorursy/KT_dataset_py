# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import os

import random

import time



import torch

import torch.nn as nn

import torch.nn.functional as F

import torch.nn.utils as torch_utils

from torchtext import data



from sklearn.metrics import accuracy_score
TRAIN_PATH = "/kaggle/input/3ztr4wc27bpa3kp/train.csv"

TEST_PATH = "/kaggle/input/3ztr4wc27bpa3kp/test.csv"

SPLIT_RATIO = 0.7



N_EPOCHS = 100

BATCH_SIZE = 32

LR=5e-4

EMBEDDING_DIM = 300

HIDDEN_DIM = 300

OUTPUT_DIM = 1

N_LAYERS = 1

BIDIRECTIONAL = True

DROPOUT = 0.5

CLIP_VALUE = 5.0





DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

SEED = 1235

torch.manual_seed(SEED)



print(DEVICE)
# define field in torchtext



Data_id = data.Field(sequential=False,

               use_vocab=False,  

               batch_first=True)



Sentence1 = data.Field(sequential=True,

                       use_vocab=True,

                       batch_first=True)



Sentence2 = data.Field(sequential=True,

                       use_vocab=True,

                       batch_first=True)



Label = data.Field(sequential=False,

                   use_vocab=False,

                   batch_first=True,

                   dtype=torch.float)

# load data using torchtext



train_data = data.TabularDataset(path=TRAIN_PATH,

                                format='csv',

                                skip_header=True,

                                fields=[('id', Data_id),('sentence1', Sentence1),('sentence2', Sentence2),('label', Label)])



test_data = data.TabularDataset(path=TEST_PATH,

                                format='csv',

                                skip_header=True,

                                fields=[('id', Data_id),('sentence1', Sentence1),('sentence2', Sentence2)])



train_data, valid_data = train_data.split(split_ratio=SPLIT_RATIO,  random_state=random.seed(SEED))



print(f'training example size: {len(train_data)}')

print(f'valid example size: {len(valid_data)}')

print(f'test example size: {len(test_data)}')
# make vocab

Sentence1.build_vocab(train_data.sentence1, train_data.sentence2)

Sentence2.vocab = Sentence1.vocab



VOCAB_SIZE = len(Sentence1.vocab)

print(f'vocab size: {len(Sentence1.vocab)}')
train_iter = data.BucketIterator(

    train_data,

    batch_size=BATCH_SIZE,

    device=DEVICE

)



test_iter = data.Iterator(

    test_data,

    batch_size=BATCH_SIZE,

    device=DEVICE,

    train=False,

    sort=False,

    sort_within_batch=False

)





valid_iter = data.Iterator(

    valid_data,

    batch_size=BATCH_SIZE,

    device=DEVICE,

    train=False,

    sort=False,

    sort_within_batch=False

)

class RNN(nn.Module):

    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim,

                 n_layers, bidirectional, dropout):

        super().__init__()

        

        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        nn.init.uniform_(self.embedding .weight, -1.0, 1.0)

        

        self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers,

                           bidirectional=bidirectional, dropout=dropout)



        #latent attention vector

        self.att_vector = torch.Tensor(hidden_dim*2, 1).cuda()

        nn.init.xavier_uniform_(self.att_vector)



        self.fc = nn.Linear(hidden_dim*8, output_dim)

        self.dropout = nn.Dropout(dropout)

        

        

    def forward(self, x1, x2):

        def attention_net(hidden):

            # dot product rnn hidden and latent vector 

            attn_weights = hidden.matmul(self.att_vector).squeeze(2) 

            # softmax attn_weights

            soft_attn_weights = F.softmax(attn_weights, 1)

            # final output using weight sum

            context = torch.mul(hidden, soft_attn_weights.unsqueeze(2)).sum(1)

            return context

        

        embedded = self.dropout(self.embedding(x1))

        output, (hidden, cell) = self.rnn(embedded)

        attention_output_x1= attention_net(output)

        

        embedded = self.dropout(self.embedding(x2))

        output, (hidden, cell) = self.rnn(embedded)

        attention_output_x2 = attention_net(output)



        # [p1,p2,abs(p1-p2),p1*p2]

        return self.fc(torch.cat([attention_output_x1, attention_output_x2, 

                                  torch.abs_(attention_output_x1 - attention_output_x2), 

                                  attention_output_x1 * attention_output_x2],dim = -1))

        # return: [batch_size, 1]
model = RNN(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, N_LAYERS, BIDIRECTIONAL, DROPOUT)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

criterion = nn.BCEWithLogitsLoss()



model = model.to(DEVICE)

criterion = criterion.to(DEVICE)
def train(model, iterator, optimizer, criterion):

    # Track the loss

    epoch_loss = 0.0

    

    model.train()

    

    for batch in iterator:

        optimizer.zero_grad()

        

        predictions = model(batch.sentence1, batch.sentence2).squeeze(1)

        loss = criterion(predictions, batch.label.float())

        loss.backward()

        

        torch_utils.clip_grad_norm_(model.parameters(), CLIP_VALUE)

        optimizer.step()

        

        epoch_loss += loss.item()

    return epoch_loss / len(iterator)
def evaluate(model, iterator, criterion):

    epoch_loss = 0.0

    

    model.eval()

    

    with torch.no_grad():

        valid_pred = []

        valid_truth = []



        model.eval()



        with torch.no_grad():

            for batch in valid_iter:

                valid_truth += batch.label.cpu().numpy().tolist()

                predictions = model(batch.sentence1, batch.sentence2).squeeze(1)

                valid_pred += torch.sigmoid(predictions).cpu().data.numpy().tolist()

                

                loss = criterion(predictions, batch.label.float())

                epoch_loss += loss.item()

        valid_pred = [i>0.5 for i in valid_pred]

    return epoch_loss / len(iterator), accuracy_score(valid_truth, valid_pred)
def prediction(model, iterator):

    with torch.no_grad():

        test_pred = []

        test_id = []



        model.eval()

        for batch in iterator:

            predictions = model(batch.sentence1, batch.sentence2).squeeze(1)

            test_pred += torch.sigmoid(predictions).cpu().data.numpy().tolist()

            test_id += batch.id.view(-1).cpu().numpy().tolist()

            

        test_pred = [int(i>0.5) for i in test_pred]

        submission = pd.DataFrame({'id': test_id, 'label': test_pred})



    return submission
# Track time taken

start_time = time.time()

best_valid_score = 0.0



for epoch in range(N_EPOCHS):

    epoch_start_time = time.time()

    

    train_loss = train(model, train_iter, optimizer, criterion)

    valid_loss, valid_score = evaluate(model, valid_iter, criterion)

    

    print(f'| Epoch: {epoch+1:02} '

          f'| Train Loss: {train_loss:.3f} '

          f'| Val. Loss: {valid_loss:.3f} '

          f'| Val. Acc: {valid_score:.3f} '

          f'| Time taken: {time.time() - epoch_start_time:.2f}s'

          f'| Time elapsed: {time.time() - start_time:.2f}s')

    

    if best_valid_score < valid_score:

        best_valid_score = valid_score

        submission = prediction(model, test_iter)

        

        print(f'| Best valid score!: {valid_score:.3f}')
submission.to_csv('submission.csv', index=False)