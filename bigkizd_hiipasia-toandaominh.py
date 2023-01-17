import numpy as np

import pandas as pd

import os

import time

import gc

import random

from tqdm._tqdm_notebook import tqdm_notebook as tqdm

from keras.preprocessing import text, sequence

import torch

from torch import nn

from torch.utils import data

from torch.nn import functional as F

from sklearn.model_selection import train_test_split

import pickle

import seaborn as sns

LSTM_UNITS = 128

DENSE_HIDDEN_UNITS = 4 * LSTM_UNITS

MAX_LEN = 220

batch_size = 128



def preprocess(data):

    punct = "/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~`" + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&'

    def clean_special_chars(text, punct):

        for p in punct:

            text = text.replace(p, ' ')

        return text



    data = data.astype(str).apply(lambda x: clean_special_chars(x, punct))

    return data

train = pd.read_csv('/kaggle/input/hiipasia/train.csv')

test = pd.read_csv('/kaggle/input/hiipasia/test.csv')



x_train = preprocess(train['text'])

y_train = np.where(train['class'] >= 0.5, 1, 0)

x_test = preprocess(test['text'])



max_features = None

tokenizer = text.Tokenizer()

tokenizer.fit_on_texts(list(x_train) + list(x_test))

max_features = max_features or len(tokenizer.word_index) + 1



x_train = tokenizer.texts_to_sequences(x_train)

x_test = tokenizer.texts_to_sequences(x_test)

x_train = sequence.pad_sequences(x_train, maxlen=MAX_LEN)

x_test = sequence.pad_sequences(x_test, maxlen=MAX_LEN)



x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size = 0.2)

x_train_torch = torch.tensor(x_train, dtype=torch.long).cuda()

x_val_torch = torch.tensor(x_val, dtype=torch.long).cuda()

x_test_torch = torch.tensor(x_test, dtype=torch.long).cuda()

y_train_torch = torch.tensor(y_train[:, np.newaxis], dtype=torch.float32).cuda()

y_val_torch = torch.tensor(y_val[:, np.newaxis], dtype=torch.float32).cuda()



train_dataset = torch.utils.data.TensorDataset(x_train_torch, y_train_torch)

val_dataset = torch.utils.data.TensorDataset(x_val_torch, y_val_torch)

test_dataset = torch.utils.data.TensorDataset(x_test_torch)



train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

train.head()
train.isnull().sum(), test.isnull().sum()
sns.distplot(train['class'])




class SpatialDropout(nn.Dropout2d):

    def forward(self, x):

        x = x.unsqueeze(2)  

        x = x.permute(0, 3, 2, 1) 

        x = super(SpatialDropout, self).forward(x) 

        x = x.permute(0, 3, 2, 1)  

        x = x.squeeze(2)

        return x



class BiLSTM(nn.Module):

    def __init__(self, embedding_matrix):

        super(BiLSTM, self).__init__()

        embed_size = embedding_matrix.shape[1]

        

        self.embedding = nn.Embedding(max_features, embed_size)

        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))

        self.embedding.weight.requires_grad = True

        self.embedding_dropout = SpatialDropout(0.3)

        

        self.lstm1 = nn.LSTM(embed_size, LSTM_UNITS, bidirectional=True, batch_first=True)

        self.lstm2 = nn.LSTM(LSTM_UNITS * 2, LSTM_UNITS, bidirectional=True, batch_first=True)

    

        self.linear1 = nn.Linear(DENSE_HIDDEN_UNITS, DENSE_HIDDEN_UNITS)

        self.linear2 = nn.Linear(DENSE_HIDDEN_UNITS, DENSE_HIDDEN_UNITS)

        

        self.linear_out = nn.Linear(DENSE_HIDDEN_UNITS, 1)

        

    def forward(self, x):

        out = self.embedding(x)

        out = self.embedding_dropout(out)

        out, _ = self.lstm1(out)

        out, _ = self.lstm2(out)

        avg_pool = torch.mean(out, 1)

        max_pool, _ = torch.max(out, 1)

        concat = torch.cat((max_pool, avg_pool), 1)

        concat_linear1  = F.relu(self.linear1(concat))

        concat_linear2  = F.relu(self.linear2(concat))

        hidden = concat + concat_linear1 + concat_linear2

        out = self.linear_out(hidden)

        

        return out

    
def build_matrix(word_index, path):

    words, embeddings = pickle.load(open(path, 'rb'), encoding='bytes')

    embedding_index = {w:i  for i, w in enumerate(words)}

    embedding_matrix = np.zeros((len(word_index) + 1, embeddings.shape[1]))

    unknown_words = []

    total = 0

    for word, i in word_index.items():

        try:

            embedding_matrix[i] = embeddings[embedding_index[word]]

            total +=1

        except KeyError:

            unknown_words.append(word)

    return embedding_matrix, unknown_words



CRAWL_EMBEDDING_PATH = '/kaggle/input/embeddings/polyglot-vi.pkl'

crawl_matrix, unknown_words_crawl = build_matrix(tokenizer.word_index, CRAWL_EMBEDDING_PATH)

print('n\'unknown words (crawl): ', len(unknown_words_crawl))
model = BiLSTM(crawl_matrix)

model = model.cuda()



batch_size = 128



optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 0.6 ** epoch)

criterion = nn.BCEWithLogitsLoss(reduction='mean')
n_epochs = 20

for epoch in range(n_epochs):

    model.train()

    avg_loss = 0.

    for data in tqdm(train_loader, disable=False):

        x_batch = data[:-1]

        y_batch = data[-1]

        y_pred = model(*x_batch)            

        loss = criterion(y_pred, y_batch)

        optimizer.zero_grad()

        loss.backward()

        clipping_value = 1.0

        torch.nn.utils.clip_grad_norm_(model.parameters(), clipping_value)

        optimizer.step()

        avg_loss += loss

    print('Epoch : {}, Loss: {}'.format(epoch, avg_loss/len(train_loader)))
model.eval()

def sigmoid(x):

    return 1 / (1 + np.exp(-x))

correct = 0

total = 0

for data in val_loader:

    x_batch = data[:-1]

    y_batch = data[-1]

    y_pred = model(*x_batch)           

    loss = criterion(y_pred, y_batch)

    y_pred = sigmoid(y_pred.detach().cpu().numpy())

    y_pred = np.where(y_pred>=0.5, 1, 0)

    y_batch = np.where(y_batch.detach().cpu().numpy()>=0.5, 1, 0)

    correct += (y_pred==y_batch).sum()

    total += len(y_batch)

print('Accuracy : ', correct/total)
model.eval()

test_preds = list()

for data in test_loader:

    x_batch = data

    y_pred = model(*x_batch)           

    y_batch = np.where(y_pred.detach().cpu().numpy()>=0.5, 1, 0)

    test_preds.extend(y_batch)
submission = pd.DataFrame.from_dict({

    'id': test['id'],

    'prediction': test_preds

})

submission.to_csv('submission.csv', index=False)
from pytorch_pretrained_bert import BertTokenizer, BertForSequenceClassification, BertAdam

from pytorch_pretrained_bert import BertConfig