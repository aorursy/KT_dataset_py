import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))
import nltk

from collections import Counter

import itertools

import torch
from sklearn import model_selection

imdb_df = pd.read_csv('../input/imdb_master.csv', encoding='latin-1')

dev_df = imdb_df[(imdb_df.type == 'train') & (imdb_df.label != 'unsup')]

test_df = imdb_df[(imdb_df.type == 'test')]

train_df, val_df = model_selection.train_test_split(dev_df, test_size=0.05, stratify=dev_df.label)
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
print(train_tensor)
from torch.utils.data import TensorDataset,DataLoader

from torch.utils import data



train_ds = TensorDataset(train_tensor, train_labels)

val_ds = TensorDataset(val_tensor, val_labels)

test_ds = TensorDataset(test_tensor, test_labels)



batch_size = 32

train_loader = DataLoader(train_ds, batch_size = batch_size)

val_loader = DataLoader(val_ds, batch_size = batch_size)

test_loader = DataLoader(test_ds, batch_size = batch_size)
import torch.nn as nn

import torch.nn.functional as F



class Network(nn.Module):

    def __init__(self):

        super(Network, self).__init__()

        self.embedding = nn.Embedding(10000, 500)

        self.rec = nn.LSTM(500, 1000, batch_first = True)

        self.classifier = nn.Sequential(

            nn.Linear(1000, 1),

            nn.Sigmoid(),

        )

        

        #self.conv = nn.Sequential(

        #    nn.Embedding(10000, 100),

        #    nn.Conv1d(in_channels=1, out_channels=100, kernel_size=4, padding=1), nn.ReLU(), 

        #    nn.Conv1d(in_channels=100, out_channels=100, kernel_size=4, padding=1), nn.ReLU(),nn.MaxPool1d(5),

        #    nn.Conv1d(in_channels=100, out_channels=120, kernel_size=4, padding=1), nn.MaxPool1d(5))

        #    

        #self.classifier1 = nn.Sequential(

        #    nn.Linear(360,1),nn.Sigmoid())

        

    def forward(self, x):       

        x = self.embedding(x)

        _, (x, __) = self.rec(x)

        x = x.view(x.size()[1::])

        res = self.classifier(x).view(-1)

        return res

    

        #x = torch.transpose(x,-1,-2)

        #print("bef",x.size())

        #y = self.conv(x)

        #print("aft",y.size())

        #y = y.view(x.size(0), -1)

        #return self.classifier1(y)

        
from sklearn import metrics

def train_model(epochs, model, optimizer, criterion, train_loader,val_loader, device, n_prints=1):

    print_every = len(train_loader) // n_prints

    for epoch in range(epochs):

        best_acc = 0

        model.train()

        model.to(device)

        running_train_loss = 0.0

        

        for iteration, (xx, yy) in enumerate(train_loader):

            optimizer.zero_grad()

            xx, yy = xx.to(device), yy.to(device)

            out = model(xx)

            loss = criterion(out, yy.type(torch.float32))

            running_train_loss += loss.item()

            loss.backward()

            optimizer.step()

            

            if(iteration % print_every == print_every - 1):

                running_train_loss /= print_every

                print(f"Epoch {epoch}, iteration {iteration} training_loss {running_train_loss}")

                running_train_loss = 0.0

            

        with torch.no_grad():

            model.eval()

            running_corrects = 0

            running_total = 0

            running_loss = 0.0

            all_preds = []

            correct_preds = []

            

            for xx, yy in val_loader:

                batch_size = xx.size(0)

                xx, yy = xx.to(device), yy.to(device)

                out = model(xx)

                

                loss = criterion(out, yy.type(torch.float32))

                running_loss += loss.item()

                for i in out:

                    if i >= 0.5:

                        all_preds.append(1)

                    else:

                        all_preds.append(0)

                correct_preds.extend(yy.cpu().tolist())

                

               # running_corrects += (predictions == yy).sum().item()

               # running_total += batch_size

            cur_acc = metrics.accuracy_score(correct_preds,all_preds)

            mean_val_loss = running_loss / len(val_loader)

            

            if cur_acc > best_acc:

                best_acc = cur_acc

                torch.save(model.state_dict(), "../best_model.pytorch")

            

            print(f"Epoch {epoch}, val_loss {mean_val_loss}, val_accuracy {cur_acc}")

            

    model.load_state_dict(torch.load("../best_model.pytorch"))

                
cuda_device = torch.device('cuda')

device = cuda_device



model = Network()

optimizer = torch.optim.Adam(model.parameters(), lr=0.0004)

optimizer1 = torch.optim.SGD(model.parameters(), lr=0.01,momentum=0.5)



criterion = nn.BCELoss()

train_model(10, model, optimizer, criterion,train_loader,val_loader, device, n_prints=1)
from sklearn.metrics import classification_report



model.load_state_dict(torch.load("../best_model.pytorch"))

all_preds = []

correct_preds = []

for xx,yy in test_loader:

    xx = xx.to(device)

    y_pred = model.forward(xx)

    for i in y_pred:

        if i >= 0.5:

            all_preds.append(1)

        else:

            all_preds.append(0)

    correct_preds.extend(yy.tolist())

print(metrics.accuracy_score(correct_preds,all_preds))

print(classification_report(correct_preds,all_preds))