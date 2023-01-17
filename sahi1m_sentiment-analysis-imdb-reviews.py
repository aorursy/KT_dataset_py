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
# Read Input

'''

reviews = pd.read_csv("../input/reviews.txt", header=None)

reviews = list(reviews.iloc[:].values)

label = pd.read_csv("../input/labels.txt", header=None)

labels = label.iloc[:].values

#print(reviews[:1])

encoded_label = np.array([1 if label == "positive" else 0 for label in labels])

'''
from collections import Counter

pos_counts = Counter()

neg_counts = Counter()

tot_counts = Counter()
with open('../input/reviews.txt', 'r') as f:

    reviews = f.read()

with open('../input/labels.txt', 'r') as f:

    labels = f.read()
# Preprocessing

from string import punctuation

reviews = reviews.lower()

corpus = ''.join([c for c in reviews if c not in punctuation])

reviews = corpus.split("\n")

corpus = ' '.join(reviews)
# labels

labels = labels.split("\n")

labels = np.array([1 if label == "positive" else 0 for label in labels])

# Different Vocabularies

for i in range(len(reviews)):

    if(labels[i]):

        for word in reviews[i].split(" "):

            pos_counts[word] += 1

            tot_counts[word] += 1

    else:

        for word in reviews[i].split(" "):

            neg_counts[word] += 1

            tot_counts[word] += 1
reviews_ints = []

for review in reviews:

    reviews_ints.append([tot_counts[word] for word in review.split()])
words = corpus.split()

counts = Counter(words)

vocab = sorted(counts, key=counts.get, reverse=True)

vocab_to_int = {word: ii for ii, word in enumerate(vocab, 1)}



reviews_ints = []

for review in reviews:

    reviews_ints.append([vocab_to_int[word] for word in review.split()])
non_zero_idx = [ii for ii, review in enumerate(reviews_ints) if len(review) != 0]



# remove 0-length reviews and their labels

reviews_ints = [reviews_ints[ii] for ii in non_zero_idx]

labels = np.array([labels[ii] for ii in non_zero_idx])
''' 

Return features of review_ints, where each review is padded with 0's if len less than seq_length

or truncated to the input seq_length.

'''

def pad_features(reviews_ints, seq_length):

    

    # getting the correct rows x cols shape

    features = np.zeros((len(reviews_ints), seq_length), dtype=int)



    # for each review, I grab that review and 

    for i, row in enumerate(reviews_ints):

        features[i, -len(row):] = np.array(row)[:seq_length]

    

    return features
# Features

features = pad_features(reviews_ints, seq_length = 200)
# Train, Test and Validation sets

from sklearn.model_selection import train_test_split

train_x, test_valid_x, train_y, test_valid_y = train_test_split(features, labels, test_size=0.2)



valid_x, test_x, valid_y, test_y = train_test_split(test_valid_x,test_valid_y, test_size=0.5)
import torch

from torch.utils.data import TensorDataset, DataLoader



# create Tensor datasets

train_data = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))

valid_data = TensorDataset(torch.from_numpy(valid_x), torch.from_numpy(valid_y))

test_data = TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_y))



# dataloaders

batch_size = 50



train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)

valid_loader = DataLoader(valid_data, shuffle=True, batch_size=batch_size)

test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size)
import torch.nn as nn



class SentimentRNN(nn.Module):

    def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim, n_layers, drop_prob=0.5):

        super(SentimentRNN, self).__init__()

        self.output_size = output_size

        self.hidden_dim = hidden_dim

        self.n_layers = n_layers

        

        # DEFINE LAYERS

        

        # embedding & LSTM

        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout = drop_prob, batch_first=True)

        

        # dropout layer

        self.dropout = nn.Dropout(0.3)

        

        # Output

        self.fc = nn.Linear(hidden_dim, output_size)

        self.sig = nn.Sigmoid()

        

    def forward(self,x,hidden):

        batch_size = x.size(0)

        

        x = x.long()

        embeds = self.embedding(x)

        lstm_out, hidden = self.lstm(embeds, hidden)

        

        # stack up lstm outputs

        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)

        

        out = self.dropout(lstm_out)

        out = self.fc(out)

        sig_out = self.sig(out)

        

        #reshape

        sig_out = sig_out.view(batch_size,-1)

        sig_out = sig_out[:,-1]

        return sig_out, hidden

    

    def init_hidden(self,batch_size):

        weight = next(self.parameters()).data

        hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_(),

                      weight.new(self.n_layers, batch_size, self.hidden_dim).zero_())

        

        return hidden

        

        
net = SentimentRNN(len(vocab_to_int)+1, 1, 400, 256, 2)



print(net)
# learning Param

lr=0.001

# BinaryCrossEntropyLoss

criterion = nn.BCELoss()

optimizer = torch.optim.Adam(net.parameters(), lr=lr)
epochs = 4 # 3-4 is approx where I noticed the validation loss stop decreasing



counter = 0

print_every = 50

clip=5 

net.train()
for e in range(epochs):

    h = net.init_hidden(batch_size)

    for inputs,labels in train_loader:

        counter+=1

        # Creating new variables for the hidden state, otherwise

        # we'd backprop through the entire training history

        h = tuple([each.data for each in h])

        

        net.zero_grad()

        output, h = net(inputs,h)

        

        loss = criterion(output.squeeze(), labels.float())

        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.

        nn.utils.clip_grad_norm_(net.parameters(), clip)

        optimizer.step()

        

        # loss stats

        if counter % print_every == 0:

            # Get validation loss

            val_h = net.init_hidden(batch_size)

            val_losses = []

            net.eval()

            for inputs, labels in valid_loader:

                val_h = tuple([each.data for each in val_h])

                output, val_h = net(inputs, val_h)

                val_loss = criterion(output.squeeze(), labels.float())



                val_losses.append(val_loss.item())



            net.train()

            print("Epoch: {}/{}...".format(e+1, epochs),

                  "Step: {}...".format(counter),

                  "Loss: {:.6f}...".format(loss.item()),

                  "Val Loss: {:.6f}".format(np.mean(val_losses)))
# Test

test_losses = []

num_correct = 0

h = net.init_hidden(batch_size)

net.eval()



for inputs,labels in test_loader:

    h = tuple([each.data for each in h ])

    out,h = net(inputs,h)

    

    test_loss = criterion(out.squeeze(),labels.float())

    test_losses.append(test_loss.item())

    # convert out prob to int

    pred = torch.round(out.squeeze())

    

    correct_tensor = pred.eq(labels.float().view_as(pred))

    correct = np.squeeze(correct_tensor.numpy())

    num_correct += np.sum(correct)



print("Test loss: {:.3f}".format(np.mean(test_losses)))

test_acc = num_correct/len(test_loader.dataset)

print("Test accuracy: {:.3f}".format(test_acc))

    

    