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
import torch

from torch import nn

from torchvision import datasets, models, transforms

import torch.nn.functional as func

import torchvision

from torchvision import transforms

from torch.utils.data import TensorDataset,DataLoader

from torch import optim

from torch import device as dev

from sklearn.metrics import classification_report

import torch.utils.data as tdata

from sklearn import model_selection

from torch.utils.data import DataLoader, Dataset

import os

from sklearn.utils import shuffle

import tensorflow as tf

import nltk

from collections import Counter

import itertools

from tqdm import tqdm_notebook
seed = 9931

np.random.seed(seed)

torch.manual_seed(seed)
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
print(train_features[3].input_ids)
train_tensor, train_labels = features_to_tensor(train_features)

val_tensor, val_labels = features_to_tensor(val_features)

test_tensor, test_labels = features_to_tensor(test_features)
print(val_tensor[:2])
print(len(text2id.vocab))

vocab_size = len(text2id.vocab) + 1
print(train_tensor.size())

print(train_labels.size())




train_dataset = TensorDataset(train_tensor, train_labels)

val_dataset = TensorDataset(val_tensor, val_labels)

test_dataset = TensorDataset(test_tensor, test_labels)



print(train_dataset[0])
train_loader = DataLoader(train_dataset, batch_size = 50)

val_loader = DataLoader(val_dataset, batch_size = 50)

test_loader = DataLoader(test_dataset, batch_size = 50)
class BestModel(nn.Module):

    def __init__(self):

        super().__init__()

        self.embed = nn.Embedding(vocab_size, 50, padding_idx = 0)

        self.lstm = nn.LSTM(50, 500, batch_first = True)

        self.linear  = nn.Linear(500, 1)

        

    def forward(self, x):

        x = self.embed(x)

        x, (hn, cn) = self.lstm(x)

        hn = hn.view(hn.size()[1:])

        out = self.linear(hn)

        sig = torch.sigmoid(out)

        sig = sig.view(-1)



        return sig
model = BestModel().cuda()

optimizer = optim.Adam(model.parameters(), lr = 2e-3)

criterion = nn.BCELoss()
def train(model, train_loader, val_loader, optimizer, criterion, epochs, tries):

    

    val_loss_best = 100

    check = 0

    

    for epoche in range(epochs):

        model.train()

        val_loss = 0

        epoch_loss = 0

        

        for xx, yy in train_loader:            

            xx = xx.cuda()

            yy = yy.cuda()

            optimizer.zero_grad()

            pred = model.forward(xx)

            loss = criterion(pred, yy.type(torch.float32))

            epoch_loss += loss.item()

            loss.backward()

            optimizer.step()

        epoch_loss /= len(train_loader)

        with torch.no_grad():

            model.eval()

            for xx,yy in val_loader:

                xx = xx.cuda()

                yy = yy.cuda()

                pred = model.forward(xx)

                loss = criterion(pred, yy.type(torch.float32))

                val_loss += loss.item()

            val_loss /= len(val_loader)

            print("Epoch = ", epoche, ", Epoch_loss = ", epoch_loss, ", Val_loss = ", val_loss)

            

            if val_loss < val_loss_best:

                print("Not bad, not bad! Keep going!")

                check = 0

                torch.save(model.state_dict(), "../best_model.md")

                val_loss_best = val_loss

            else:

                check += 1

                print("Very bad! Try again!")

                if check == tries:

                    print("You were my brother Model, I loved you!")   

                    break

                    

    model.load_state_dict(torch.load("../best_model.md"))   

    model.eval()

    model.cpu()
train(model, train_loader, val_loader, optimizer, criterion, epochs = 10, tries = 3)
model.eval()

preds = []

true = []

for xx,yy in test_loader:

    xx = xx.cuda()

    model.cuda()

    pred = model.forward(xx)

    p = pred.tolist()

    j = 0

    for a in p:

        if a > 0.5:

            p[j] = 1

        else:

            p[j] = 0

        j = j + 1

    

    preds.extend(p)

    true.extend(yy.tolist())

print(classification_report(true, preds))