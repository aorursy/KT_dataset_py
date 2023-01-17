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
print(os.listdir("../input/aclimdb/aclImdb"))

print(os.listdir("../input/aclimdb/aclImdb/train"))

print(os.listdir("../input/aclimdb/aclImdb/test"))
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
seed = 9931

np.random.seed(seed)

torch.manual_seed(seed)



path = './../input/aclimdb/'



def imdb_dataset_loader(path):

    imdb_path = os.path.join(path, 'aclImdb')

    

    train_texts = []

    train_labels = []

    val_texts = []

    val_labels = []

    test_texts = []

    test_labels = []

    

    for drctr in ['train', 'test']:

        for fdbk in ['pos', 'neg']:

            

            directory = os.path.join(imdb_path, drctr, fdbk)

            count = 0

            

            for fname in sorted(os.listdir(directory)):

                

                if fname.endswith('.txt'):

                    with open(os.path.join(directory, fname)) as file:

                        

                        if drctr == 'train': 

                            count = count + 1

                            if count % 5 == 0:

                                val_texts.append(file.read().lower())

                            else: 

                                train_texts.append(file.read().lower())

                        else:

                            test_texts.append(file.read().lower())

                            

                    if fdbk == 'neg':

                        feedback = 0

                    else:

                        feedback = 1

                        

                    if drctr == 'train': 

                        if count % 5 == 0:

                            val_labels.append(feedback)

                        else: 

                            train_labels.append(feedback)

                    else: 

                        test_labels.append(feedback)

    

    train_texts = np.array(train_texts)

    #train_labels = np.array(train_labels)

    #val_texts = np.array(val_texts)

    #val_labels = np.array(val_labels)

    #test_texts = np.array(test_texts)

    #test_labels = np.array(test_labels)

    

    #train_texts, train_labels = shuffle(train_texts, train_labels, random_state = 0)

    #val_texts, val_labels = shuffle(val_texts, val_labels, random_state = 0)

    #test_texts, test_labels = shuffle(test_texts, test_labels, random_state = 0)



    return train_texts, train_labels, val_texts, val_labels, test_texts, test_labels



train_texts, train_labels, val_texts, val_labels, test_texts, test_labels = imdb_dataset_loader(path)



print(len(train_texts))
vocab = {}



with open(path + "aclImdb/" + "imdb.vocab") as file:

    i = 0

    for word in file.read().split():

        word = word.lower()

        vocab[word] = i

        i = i + 1



vocab_size = len(vocab)

print(vocab_size)
def to_tensor (something):

    tt_all = []

    for text in something:

        something_str = ''.join(text)

        something_split = something_str.split()

        tt = []

        for word in something_split:

            if word.isalpha():

                if word in vocab:

                    tt.append(vocab[word])

            else:

                word = word[:-1]

                if word in vocab:

                    tt.append(vocab[word])

        tt_all.append(tt)

    something_tenzor = torch.tensor(data = tt_all)

    

    return something_tenzor



#train_datasets = tdata.Subset(train_texts, train_labels)

#val_datasets = tdata.Subset(val_texts, val_labels)

#test_datasets = tdata.Subset(test_texts, test_labels)



#train_loader = DataLoader(train_datasets, batch_size = 20)

#val_loader = DataLoader(val_datasets, batch_size = 20)

#test_loader = DataLoader(test_datasets, batch_size = 20)
train_texts_tenzor = to_tensor(train_texts)

print(train_texts_tenzor)

print(train_texts_tenzor.size())
train_labels_tenzor = to_tensor(train_labels)

print(train_labels_tenzor)

print(train_labels_tenzor.size())
class BestModel(nn.Module):

    def __init__(self):

        super().__init__()

        self.embed = nn.Embedding(vocab_size, 50, padding_idx = 0)

        self.lstm = nn.LSTM(50, 500, batch_first = True)

        self.linear  = nn.Linear(500, 1)

        

    def forward(self, x):

        x = self.embed(x)

        x = self.lstm(x)

        x = torch.sigmoid(self.linear(x))

        

        return x
model = BestModel()

optimizer = optim.Adam(model.parameters(), lr = 0.0001)

criterion = nn.BCEWithLogitsLoss()
def train(model, train_loader, val_loader, optimizer, criterion, epochs):

    for epoche in range(epochs):

        model.train()

        val_loss = 0

        epoch_loss = 0

        for xx, yy in train_loader:

            xx = xx.cuda()

            yy = yy.cuda()

            optimizer.zero_grad()

            pred = model.forward(xx)

            loss = criterion(out, yy)

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

                loss = criterion(pred,yy)

                val_loss += loss.item()

            val_loss /= len(val_loader)

            print("Epoch = ", i, ", Epoch_loss = ", epoch_loss, ", Val_loss = ", val_loss)

    model.eval()

    model.cpu()
train(model, train_loader, val_loader, optimizer, criterion, epochs = 5)
model.eval()

preds = []

true = []

for xx,yy in test_loader:

    xx = xx.cuda()

    model.cuda()

    pred = model.forward(xx)

    preds.extend(pred.argmax(1).tolist())

    true.extend(yy.tolist())

print(classification_report(true, preds))