import numpy as np

import pandas as pd

import random

import sys

import os



os.listdir('../input')
# Read the entire file containing song lyrics

path = "../input/songdata.csv"

df = pd.read_csv(path)

df.head()
# See all artist in this dataset

df['artist'].unique()
DP = df[df['artist']=='Deep Purple']
DP.head()
DP_text = DP['text'].str.cat(sep='\n').lower()



print(DP_text[:100])

print('corpus length:', len(DP_text))
chars = sorted(list(set(DP_text)))

print(chars)

print('total chars:', len(chars))
# Create a dictionary of characters, see the index of characters.

char_to_int = dict((c, i) for i, c in enumerate(chars))

int_to_char = dict((i, c) for i, c in enumerate(chars))



print(char_to_int)
seq_length = 50 # The sentence window size

step = 1 # The steps between the windows

sentences = []

next_chars = []



# Create Target and sentences window

for i in range(0, len(DP_text) - seq_length, step):

    sentences.append(DP_text[i: i + seq_length]) # range from current index to sequence length charaters 

    next_chars.append(DP_text[i + seq_length]) # the next character

    

sentences = np.array(sentences)

next_chars = np.array(next_chars)



#Print Sentence Window and next charaters

print('Sentence Window')

print (sentences[:5])

print('Target charaters')

print (next_chars[:5])

print('Number of sequences:', len(sentences))
def getdata(sentences, next_chars):

    X = np.zeros((len(sentences),seq_length))

    y = np.zeros((len(sentences)))

    length = len(sentences)

    index = 0

    for i in range(len(sentences)):

        sentence = sentences[i]

        for t, char in enumerate(sentence):

            X[i, t] = char_to_int[char]

        y[i] = char_to_int[next_chars[i]]

    return X, y
train_x,train_y = getdata(sentences, next_chars)

print('Shape of training_x:', train_x.shape)

print('Shape of training_y:', train_y.shape)
import torch

import torch.nn as nn

import torch.nn.functional as F

import torch.optim as optim

from torch.autograd import Variable



class Simple_LSTM(nn.Module):

    def __init__(self,n_vocab,hidden_dim, embedding_dim,dropout = 0.2):

        super(Simple_LSTM, self).__init__()

        

        self.hidden_dim = hidden_dim

        self.embedding_dim = embedding_dim

        self.lstm = nn.LSTM(embedding_dim, hidden_dim,dropout = dropout,num_layers = 2)

        self.embeddings = nn.Embedding(n_vocab, embedding_dim)

        self.fc = nn.Linear(hidden_dim, n_vocab)

    

    def forward(self, seq_in):

        # for LSTM, input should be (Sequnce_length,batchsize,hidden_layer), so we need to transpose the input

        embedded = self.embeddings(seq_in.t()) 

        lstm_out, _ = self.lstm(embedded)

        # Only need to keep the last character 

        ht=lstm_out[-1] 

        out = self.fc(ht)

        return out
X_train_tensor = torch.tensor(train_x, dtype=torch.long).cuda()

Y_train_tensor = torch.tensor(train_y, dtype=torch.long).cuda()
from torch.utils.data import Dataset, DataLoader

train = torch.utils.data.TensorDataset(X_train_tensor,Y_train_tensor)

train_loader = torch.utils.data.DataLoader(train, batch_size = 128)
model = Simple_LSTM(47,256,256)

model.cuda()

optimizer = torch.optim.Adam(model.parameters(), lr = 0.002) # Using Adam optimizer
import time # Add time counter

avg_losses_f = []

n_epochs=20



for epoch in range(n_epochs):

    start_time = time.time()

    model.train()

    loss_fn = torch.nn.CrossEntropyLoss()

    avg_loss = 0.

    for i, (x_batch, y_batch) in enumerate(train_loader):

        y_pred = model(x_batch)

        

        loss = loss_fn(y_pred, y_batch)

        optimizer.zero_grad()

        loss.backward()

        

        optimizer.step()

        avg_loss+= loss.item()/len(train_loader)

        

    elapsed_time = time.time() - start_time 

    print('Epoch {}/{} \t loss={:.4f} \t time={:.2f}s'.format(

        epoch + 1, n_epochs, avg_loss, elapsed_time))

    

    avg_losses_f.append(avg_loss)    

    

print('All \t loss={:.4f} \t '.format(np.average(avg_losses_f)))
import matplotlib.pyplot as plt



plt.plot(avg_losses_f)

plt.xlabel('Epoch')

plt.ylabel('Loss value')

plt.show()
def sample(preds, temperature=1.0):

    preds = np.asarray(preds).astype('float64')

    preds = np.log(preds) / temperature

    exp_preds = np.exp(preds)

    preds = exp_preds / np.sum(exp_preds)

    probas = np.random.multinomial(1, preds, 1)

    return np.argmax(probas)
# Define the start sentence

sentence = 'i read in the news\nthat the average man\nplease kis'



variance = 0.25

generated = ''

original = sentence

window = sentence



for i in range(400):

    x = np.zeros((1, seq_length))

    for t, char in enumerate(window):

        x[0, t] = char_to_int[char] # Change the sentence to index vector shape (1,50)

        

    x_in = Variable(torch.LongTensor(x).cuda())

    pred = model(x_in)

    pred = np.array(F.softmax(pred, dim=1).data[0].cpu())

    next_index = sample(pred, variance)

    next_char = int_to_char[next_index] # index to char



    generated += next_char

    window = window[1:] + next_char # Update Window for next char predict

    

print(original + generated)