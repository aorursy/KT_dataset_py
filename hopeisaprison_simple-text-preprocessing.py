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
import nltk

from collections import Counter

import itertools
import torch
class InputFeatures(object):

    """A single set of features of data."""



    def __init__(self, input_ids, label_id):

        self.input_ids = input_ids

        self.label_id = label_id
class Vocab:

    def __init__(self, itos, unk_index):

        self._itos = itos

        self._stoi = {word:i for i, word in enumerate(itos)}

        self._unk_index = unk_index

        

    def __len__(self):

        return len(self._itos)

    

    def word2id(self, word):

        idx = self._stoi.get(word)

        if idx is not None:

            return idx

        return self._unk_index

    

    def id2word(self, idx):

        return self._itos[idx]
from tqdm import tqdm_notebook
class TextToIdsTransformer:

    def transform():

        raise NotImplementedError()

        

    def fit_transform():

        raise NotImplementedError()



class SimpleTextTransformer(TextToIdsTransformer):

    def __init__(self, max_vocab_size):

        self.special_words = ['<PAD>', '</UNK>', '<S>', '</S>']

        self.unk_index = 1

        self.pad_index = 0

        self.vocab = None

        self.max_vocab_size = max_vocab_size

        

    def tokenize(self, text):

        return nltk.tokenize.word_tokenize(text.lower())

        

    def build_vocab(self, tokens):

        itos = []

        itos.extend(self.special_words)

        

        token_counts = Counter(tokens)

        for word, _ in token_counts.most_common(self.max_vocab_size - len(self.special_words)):

            itos.append(word)

            

        self.vocab = Vocab(itos, self.unk_index)

    

    def transform(self, texts):

        result = []

        for text in texts:

            tokens = ['<S>'] + self.tokenize(text) + ['</S>']

            ids = [self.vocab.word2id(token) for token in tokens]

            result.append(ids)

        return result

    

    def fit_transform(self, texts):

        result = []

        tokenized_texts = [self.tokenize(text) for text in texts]

        self.build_vocab(itertools.chain(*tokenized_texts))

        for tokens in tokenized_texts:

            tokens = ['<S>'] + tokens + ['</S>']

            ids = [self.vocab.word2id(token) for token in tokens]

            result.append(ids)

        return result
def build_features(token_ids, label, max_seq_len, pad_index, label_encoding):

    if len(token_ids) >= max_seq_len:

        ids = token_ids[:max_seq_len]

    else:

        ids = token_ids + [pad_index for _ in range(max_seq_len - len(token_ids))]

    return InputFeatures(ids, label_encoding[label])

        
def features_to_tensor(list_of_features):

    text_tensor = torch.tensor([example.input_ids for example in list_of_features], dtype=torch.long)

    labels_tensor = torch.tensor([example.label_id for example in list_of_features], dtype=torch.long)

    return text_tensor, labels_tensor
from sklearn import model_selection
imdb_df = pd.read_csv('../input/imdb_master.csv', encoding='latin-1')

dev_df = imdb_df[(imdb_df.type == 'train') & (imdb_df.label != 'unsup')]

test_df = imdb_df[(imdb_df.type == 'test')]

train_df, val_df = model_selection.train_test_split(dev_df, test_size=0.05, stratify=dev_df.label)
max_seq_len=200

classes = {'neg': 0, 'pos' : 1}
text2id = SimpleTextTransformer(10000)



train_ids = text2id.fit_transform(train_df['review'])

val_ids = text2id.transform(val_df['review'])

test_ids = text2id.transform(test_df['review'])
print(train_df.review.iloc[0][:160])

print(train_ids[0][:30])
train_features = [build_features(token_ids, label,max_seq_len, text2id.pad_index, classes) 

                  for token_ids, label in zip(train_ids, train_df['label'])]



val_features = [build_features(token_ids, label,max_seq_len, text2id.pad_index, classes) 

                  for token_ids, label in zip(val_ids, val_df['label'])]



test_features = [build_features(token_ids, label,max_seq_len, text2id.pad_index, classes) 

                  for token_ids, label in zip(test_ids, test_df['label'])]
train_tensor, train_labels = features_to_tensor(train_features)

val_tensor, val_labels = features_to_tensor(val_features)

test_tensor, test_labels = features_to_tensor(test_features)
print(train_tensor.size())

print(len(text2id.vocab))
from torch.utils.data import TensorDataset,DataLoader

train_loader = DataLoader(TensorDataset(train_tensor,train_labels),64)

val_loader = DataLoader(TensorDataset(val_tensor,val_labels),64)

test_loader = DataLoader(TensorDataset(test_tensor,test_labels),64)
for xx,yy in train_loader:

    print(xx)

    print(yy)

    break
import torch.nn.functional as F

import torch.nn as nn

class intel(nn.Module):

    def __init__(self):

        super(intel, self).__init__()

        self.channel = 100

        self.embedded = nn.Embedding(10000,100)

        self.conv1 = nn.Conv1d(100, self.channel, 3)

        self.pool1 = nn.MaxPool1d(1750)

        self.norm = nn.BatchNorm1d(self.channel)

        

        self.classifier1 = nn.Linear(self.channel, 1)

        

    def forward(self,x):

        x = self.embedded(x)

        x = x.transpose(2,1)

        

        #x = self.norm(x)

        a = self.conv1(x)

        a = self.pool1(a)

        a = a.relu()

    

        e = a.view(-1, self.channel) #225*64

        e = self.classifier1(e)

        e = e.sigmoid()

        return e

    def convweight(self):

        return self.conv1.weight
torch.manual_seed(1488)

model = intel()

criterion = nn.BCELoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.0025)



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = model.to(device)

print(device)
def train(model,train_ds,val_ds,optimizer, epochs, tolerance):

    running_tolerance = tolerance

    val_loss_best = 555

    criterion = nn.BCELoss()

    for i in range(epochs):

        model.train()

        epoch_loss = 0

        val_loss = 0

        for xx,yy in train_ds:

            xx, yy = xx.cuda(), yy.cuda()

            batchsize = xx.size(0)

            optimizer.zero_grad()

            y = model.forward(xx).view(-1)

            yy = yy.float().view(-1)

            loss = criterion(y,yy)

            epoch_loss += loss

            loss.backward()

            optimizer.step()

        epoch_loss /= len(train_loader)

        with torch.no_grad():

            model.eval()

            for xx,yy in val_ds:

                xx, yy = xx.cuda(), yy.cuda()

                batchsize = xx.size(0)

                y = model.forward(xx).view(-1)

                yy = yy.float()

                loss = criterion(y,yy)

                val_loss += loss

            val_loss /= len(val_loader)

            status = "epoch=%d, loss=%f, val_loss=%f, best_loss=%f" % (i,epoch_loss,val_loss,val_loss_best)

            print(status)

            if val_loss<val_loss_best:

                torch.save(model.state_dict(), "../best_model.md")

                val_loss_best = val_loss

                running_tolerance = tolerance

            else:

                running_tolerance -=1

                if running_tolerance==0:

                    print("Stop training")   

                    break

                print("Running tolerance is ", str(running_tolerance), "best is ",str(val_loss_best))

            

    model.load_state_dict(torch.load("../best_model.md"))    

    model.eval()

    model.cpu()

a = model.convweight()

print(a)
train(model,train_loader,val_loader,optimizer,10,tolerance=5)
a = model.convweight()

print(a)
from sklearn.metrics import classification_report



model.eval()

all_preds = []

correct_preds = []

for xx,yy in test_loader:

    xx = xx.cuda()

    model.cuda()

    y_pred = model.forward(xx)

    all_preds.extend([i[0]>0.5 for i in y_pred.tolist()])

    correct_preds.extend(yy.tolist())

print(classification_report(correct_preds,all_preds))