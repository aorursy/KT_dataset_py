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

import torch

import random

import pickle

from tqdm import tqdm

from collections import Counter

from torch.utils.data import DataLoader,Dataset

import torch.nn as nn

import torch.nn.functional as F
!unzip '/kaggle/input/sentiment-analysis-on-movie-reviews/train.tsv.zip'

!unzip '/kaggle/input/sentiment-analysis-on-movie-reviews/test.tsv.zip'
train = pd.read_csv('train.tsv', sep='\t')

test = pd.read_csv('test.tsv', sep='\t')
def Corpus_Extr(df):

    print('Construct Corpus...')

    corpus = []

    for i in tqdm(range(len(df))):

        corpus.append(df.Phrase[i].lower().split())

    corpus = Counter(np.hstack(corpus))

    corpus = corpus

    corpus2 = sorted(corpus,key=corpus.get,reverse=True)

    print('Convert Corpus to Integers')

    vocab_to_int = {word: idx for idx,word in enumerate(corpus2,1)}

    print('Convert Phrase to Integers')

    phrase_to_int = []

    for i in tqdm(range(len(df))):

        phrase_to_int.append([vocab_to_int[word] for word in df.Phrase.values[i].lower().split()])

    return corpus,vocab_to_int,phrase_to_int

corpus,vocab_to_int,phrase_to_int = Corpus_Extr(train)
def Pad_sequences(phrase_to_int,seq_length):

    pad_sequences = np.zeros((len(phrase_to_int), seq_length),dtype=int)

    for idx,row in tqdm(enumerate(phrase_to_int),total=len(phrase_to_int)):

        pad_sequences[idx, :len(row)] = np.array(row)[:seq_length]

    return pad_sequences
pad_sequences = Pad_sequences(phrase_to_int,30)
train.sample(50)
class PhraseDataset(Dataset):

    def __init__(self,df,pad_sequences):

        super().__init__()

        self.df = df

        self.pad_sequences = pad_sequences

        

    def __len__(self):

        return len(self.df)

    

    def __getitem__(self,idx):

        if 'Sentiment' in self.df.columns:

            label = self.df['Sentiment'].values[idx]

            item = self.pad_sequences[idx]

            return item,label

        else:

            item = self.pad_sequences[idx]

            return item
class SentimentRNN(nn.Module):

    

    def __init__(self,corpus_size,output_size,embedd_dim,hidden_dim,n_layers):

        super().__init__()

        self.output_size = output_size

        self.n_layers = n_layers

        self.hidden_dim = hidden_dim

        

        self.embedding = nn.Embedding(corpus_size,embedd_dim)

        self.lstm = nn.LSTM(embedd_dim, hidden_dim,n_layers,dropout=0.5, batch_first=True)

        self.dropout = nn.Dropout(0.3)

        self.fc = nn.Linear(hidden_dim,output_size)

        self.act = nn.Sigmoid()

        

    def forward(self,x,hidden):

        batch_size = x.size(0)

        embeds = self.embedding(x)

        lstm_out, hidden = self.lstm(embeds,hidden)

        lstm_out = lstm_out.contiguous().view(-1,self.hidden_dim)

        out = self.dropout(lstm_out)

        out = self.fc(out)

        out = self.act(out)

        out = out.view(batch_size,-1)

        out = out[:,-5:]

        return out, hidden

    def init_hidden(self,batch_size):

        

        weight = next(self.parameters()).data

        hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_(),

                   weight.new(self.n_layers, batch_size, self.hidden_dim).zero_())

        return hidden

        
vocab_size = len(vocab_to_int)

output_size = 5

embedding_dim = 400

hidden_dim = 256

n_layers = 2

net = SentimentRNN(vocab_size, output_size, embedding_dim, hidden_dim,n_layers)
net.train()

clip=5

epochs = 200

counter = 0

print_every = 100

lr=0.01



def criterion(input, target, size_average=True):

    """Categorical cross-entropy with logits input and one-hot target"""

    l = -(target * torch.log(F.softmax(input, dim=1) + 1e-10)).sum(1)

    if size_average:

        l = l.mean()

    else:

        l = l.sum()

    return l

#criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(net.parameters(), lr=lr)
import gc
batch_size=32

losses = []

accs=[]

for e in range(epochs):

    a = np.random.choice(len(train)-1, 1000)

    train_set = PhraseDataset(train.loc[train.index.isin(np.sort(a))],pad_sequences[a])

    train_loader = DataLoader(train_set,batch_size=32,shuffle=True)

    # initialize hidden state

    h = net.init_hidden(32)

    running_loss = 0.0

    running_acc = 0.0

    # batch loop

    for idx,(inputs, labels) in enumerate(train_loader):

        counter += 1

        gc.collect()

        # Creating new variables for the hidden state, otherwise

        # we'd backprop through the entire training history

        h = tuple([each.data for each in h])



        # zero accumulated gradients

        optimizer.zero_grad()

        if inputs.shape[0] != batch_size:

            break

        # get the output from the model

        output, h = net(inputs, h)

        labels=torch.nn.functional.one_hot(labels, num_classes=5)

        # calculate the loss and perform backprop

        loss = criterion(output, labels)

        loss.backward()

        running_loss += loss.cpu().detach().numpy()

        running_acc += (output.argmax(dim=1) == labels.argmax(dim=1)).float().mean()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.

        nn.utils.clip_grad_norm_(net.parameters(), clip)

        optimizer.step()

        if idx%20 == 0:

            print("Epoch: {}/{}...".format(e+1, epochs),

                  "Step: {}...".format(counter),

                  "Loss: {:.6f}...".format((running_loss/(idx+1))))

            losses.append(float(running_loss/(idx+1)))

            print(f'acc:{running_acc/(idx+1)}')

            accs.append(running_acc/(idx+1))
import matplotlib.pyplot as plt

plt.plot(losses)

plt.show()

plt.plot(accs)

plt.show()