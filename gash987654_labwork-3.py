import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import nltk

from collections import Counter

import itertools

import torch

from sklearn import model_selection

import os

import torch.nn as nn

import torch.nn.functional as F



print(os.listdir("../input"))
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



dict_size = 10000

max_seq_len=200

classes = {'neg': 0, 'pos' : 1}

text2id = SimpleTextTransformer(dict_size)



train_ids = text2id.fit_transform(train_df['review'])

val_ids = text2id.transform(val_df['review'])

test_ids = text2id.transform(test_df['review'])

train_features = [build_features(token_ids, label,max_seq_len, text2id.pad_index, classes) 

                  for token_ids, label in zip(train_ids, train_df['label'])]



val_features = [build_features(token_ids, label,max_seq_len, text2id.pad_index, classes) 

                  for token_ids, label in zip(val_ids, val_df['label'])]



test_features = [build_features(token_ids, label,max_seq_len, text2id.pad_index, classes) 

                  for token_ids, label in zip(test_ids, test_df['label'])]



from torch.utils.data import TensorDataset,DataLoader

from torch.utils import data



train_tensor, train_labels = features_to_tensor(train_features)

val_tensor,     val_labels = features_to_tensor(val_features)

test_tensor,   test_labels = features_to_tensor(test_features)



train_dataset = TensorDataset(train_tensor, train_labels)

val_dataset   = TensorDataset(val_tensor, val_labels)

test_dataset  = TensorDataset(test_tensor, test_labels)



batch_size = 64

train_loader = DataLoader(train_dataset, batch_size = batch_size)

val_loader   = DataLoader(val_dataset, batch_size = batch_size)

test_loader  = DataLoader(test_dataset, batch_size = batch_size)

class Classifier(nn.Module):

    def __init__(self):

        super().__init__()

        self.embed = nn.Embedding(dict_size, 70)

        self.lstm = nn.LSTM(70, 400, batch_first = True)

        

        self.convolution = nn.Sequential(

            nn.Conv1d(50,70,3),

            nn.ReLU(),

            nn.MaxPool1d(20),

        )

        

        self.classifier  = nn.Sequential(

            nn.Linear(400, 1),

            nn.Sigmoid(),

        )

        

        

    def forward(self, x):

        batchsize = x.size(0)

        x = self.embed(x)

        #y1 = self.convolution(x.transpose(2,1)).view(batchsize,-1)

        

        a, (y2, b) = self.lstm(x)

        y2 = y2.view(batchsize,-1)

        

        #print(y1.shape,y2.shape)

        #y = y2+y1



        return self.classifier(y2)
from sklearn.metrics import accuracy_score,classification_report

def train_model(model,optimizer, criterion, train_dl,val_dl,epochs):

    best_accuracy=0

    model.cuda()

    for epoch in range(epochs):

        model.train()

        running_train_loss = 0.0

        

        for xx, yy in train_dl:

            optimizer.zero_grad()

            xx, yy = xx.cuda(), yy.cuda()

            out = model.forward(xx)

            loss = criterion(out, yy.float())

            running_train_loss += loss.item()

            loss.backward()

            optimizer.step()

        train_loss = running_train_loss/len(train_loader)

        with torch.no_grad():

            model.eval()

            running_corrects = 0

            running_total = 0

            running_loss = 0.0

            y_pred = []

            y_true = []

            for xx, yy in val_dl:

                batch_size = xx.size(0)

                xx, yy = xx.cuda(), yy.cuda()

                out = model.forward(xx)

                for j in out:

                    if j<0.5:

                        y_pred.append(0)

                    else:

                        y_pred.append(1)

                y_true.extend(yy.tolist())

                loss = criterion(out, yy.float())

                running_loss += loss.item()

            accuracy = accuracy_score(y_pred,y_true)

            mean_val_loss = running_loss / len(val_dl)

            if accuracy > best_accuracy:

                best_accuracy = accuracy

                torch.save(model.state_dict(), "../classifier.py")

            

            print(f"Epoch {epoch},train_loss {train_loss}, val_loss {mean_val_loss}, accuracy = {accuracy}")
from torch.optim import Adam

classifier = Classifier()

criterion = nn.BCELoss()



optimizer = Adam(classifier.parameters(), lr=0.005)

train_model(classifier,optimizer,criterion,train_loader,val_loader,15)


classifier.load_state_dict(torch.load("../classifier.py"))

classifier.cuda()

all_preds = []

correct_preds = []

loss=0

for xx,yy in test_loader:

    xx = xx.cuda()

    yy = yy.cuda()

    res = classifier.forward(xx)

    for j in res:

        if j<0.5:

            all_preds.append(0)

        else:

            all_preds.append(1)

    correct_preds.extend(yy.tolist())

classifier.cpu()

print("Test accuracy = ",accuracy_score(all_preds,correct_preds))

print((classification_report(correct_preds,all_preds)))