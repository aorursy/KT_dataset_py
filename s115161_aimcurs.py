import numpy as np

import pandas as pd

import torch as tr

import os

print(os.listdir("../input"))

print("aclImdb:\n", os.listdir("../input/aclimdb/aclImdb"))

print("TRAIN:\n", os.listdir("../input/aclimdb/aclImdb/train"), "\nTEST:\n", os.listdir("../input/aclimdb/aclImdb/test"))
import torch 

np.random.seed(1234)

torch.manual_seed(1234)
path = './../input/aclimdb/'
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
def datasetLoader(path):

    imdb_path = os.path.join(path, 'aclImdb')

    

    trainTexts = []

    testTexts = []

    trainRating = []

    testRating = []

    

    for dir_ in ['train', 'test']:

        for rate in ['pos', 'neg']:

            directory = os.path.join(imdb_path, dir_, rate)



            for fileName in sorted(os.listdir(directory)):

                if fileName.endswith('.txt'):

                    with open(os.path.join(directory, fileName)) as file:

                        if dir_ == 'train':  

                            trainTexts.append(file.read())

                        else:

                            testTexts.append(file.read())

                    if rate == 'neg':

                        rating = 0

                    else:

                        rating = 1

                    if dir_ == 'train': 

                        trainRating.append(rating)

                    else: 

                        testRating.append(rating)

    

    validText = []

    validRating = []

    #train_texts, train_labels = shuffle(train_texts, train_labels, random_state = 0)

    trainTexts, trainRating, testTexts, testRating = shuffle(trainTexts, trainRating, testTexts, testRating, random_state = 0)

    

    for i in range(5000):

        validText.append(trainTexts.pop(i))

        validRating.append(trainRating.pop(i))

    

    return(trainTexts, trainRating, testTexts, testRating, validText, validRating)
trainTexts, trainRating, testTexts, testRating, validText, validRating = datasetLoader(path)
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

#classes = {'neg': 0, 'pos' : 1}



text2id = SimpleTextTransformer(10000)       

train_ids = text2id.fit_transform(trainTexts)

val_ids = text2id.transform(validText)

test_ids = text2id.transform(testTexts)
train_features = [build_features(token_ids, label,max_seq_len, text2id.pad_index, [0,1]) 

                  for token_ids, label in zip(train_ids, trainRating)]

val_features = [build_features(token_ids, label,max_seq_len, text2id.pad_index, [0,1]) 

                  for token_ids, label in zip(val_ids, validRating)]

test_features = [build_features(token_ids, label,max_seq_len, text2id.pad_index, [0,1]) 

                  for token_ids, label in zip(test_ids, testRating)]



train_tensor, train_labels = features_to_tensor(train_features)

val_tensor, val_labels = features_to_tensor(val_features)

test_tensor, test_labels = features_to_tensor(test_features)



print(len(text2id.vocab))
train_dataset = TensorDataset(train_tensor, train_labels)

val_dataset = TensorDataset(val_tensor, val_labels)

test_dataset = TensorDataset(test_tensor, test_labels)



print(train_dataset[0])
train_loader = DataLoader(train_dataset, batch_size = 100)

val_loader = DataLoader(val_dataset, batch_size = 100)

test_loader = DataLoader(test_dataset, batch_size = 100)
class BestModel(nn.Module):

    def __init__(self, vocab_size):

        super().__init__()

        self.embed = nn.Embedding(vocab_size + 1, 100, padding_idx = 0)

        self.lstm = nn.LSTM(100, 300, batch_first = True)

        self.lin  = nn.Linear(300, 1)

        

    def forward(self, x):

        x = self.embed(x)

        x, (h_t, h_c) = self.lstm(x)

        out = self.lin(h_t)

        sig = torch.sigmoid(out)

        sig = sig.view(-1)



        return sig
vocab_size = len(text2id.vocab)

model = BestModel(vocab_size).cuda()

optimizer = optim.Adam(model.parameters(), lr = 0.0005)

criterion = nn.BCELoss()
def fit(model, train_loader, val_loader, optimizer, criterion, epochs, tries):

    

    min_loss_v = 100

    counter = 0

    for epoch in range(epochs):

        model.train()

        loss_v = 0

        epoch_loss = 0

        for xx,yy in train_loader:

            xx = xx.cuda()

            yy = yy.cuda()

            optimizer.zero_grad()

            pred = model.forward(xx)

            loss = criterion(pred,yy.type(torch.float32))

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

                loss = criterion(pred,yy.type(torch.float32))

                loss_v += loss.item()

            loss_v /= len(val_loader)

            print("Epoch = ", epoch, ", Epoch_loss = ", epoch_loss, ", Val_loss = ", loss_v)

            if loss_v < min_loss_v:

                print("new min_loss_v")

                torch.save(model.state_dict(), "../best_model.md")

                min_loss_v = loss_v

            else:

                counter += 1

                print("counter = ", counter, "fail")

                if counter == tries:

                    break

            

    state = torch.load("../best_model.md")  

    model.load_state_dict(state)    

    model.eval()

    model.cpu()

fit(model, train_loader, val_loader, optimizer, criterion, 50, 20)
model.eval()

preds = []

true = []

for xx,yy in val_loader:

    xx = xx.cuda()

    model.cuda()

    pred = model.forward(xx)

    p = pred.tolist()

    

    for i in range(len(p)):

        p[i]=round(p[i])

    

    preds.extend(p)

    true.extend(yy.tolist())

print(classification_report(true, preds))