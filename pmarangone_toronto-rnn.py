import os

print(os.listdir("../input"))
import pandas as pd

dataset = pd.read_csv('../input/guide-to-text-classification/train_from_guide')
dataset.head()
import torch

import torch.optim

import torchtext

import torch.nn as nn

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, TensorDataset



# from pytorch_pretrained_bert import BertTokenizer

# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

glove = torchtext.vocab.Vectors('../input/glove-global-vectors-for-word-representation/glove.6B.200d.txt')

from tensorflow.keras.preprocessing.text import text_to_word_sequence



tokz = text_to_word_sequence
dataset.iloc[:,[0]].head() # there is difference between iloc[:, 0] and iloc[:, [0]]; the first one is a series, and the second a dataframe
len(dataset)
tokz(dataset.iloc[2,1])
def get_data(data):

    train = []

    valid = []

    test = []

    

    i = 0

    for sentence, label in zip(data.iloc[:,1], data.iloc[:,0]):

        idxs = [glove.stoi[w] for w in tokz(sentence) if w in glove.stoi]

        

        if not idxs:

            continue

        # convert lst to torch tensor

        idxs = torch.tensor(idxs)

        label = torch.tensor(int(label == 0)).long()

        if i % 5 < 3:

            train.append((idxs, label))

        elif i % 5 == 4:

            valid.append((idxs, label))

        else:

            test.append((idxs, label))

        i += 1

    return train, valid, test



train, valid, test = get_data(dataset)

        
len(train), len(valid), len(test)
vectors, label = train[0]
print(vectors)

print(vectors.shape)
for i in range(10):

    vectors, label = train[i]

    print(vectors.shape)
glove_emb = nn.Embedding.from_pretrained(glove.vectors)
# you have your vectors and your label, see how your features will comport when passe

# to embeddings



sample_embedding = glove_emb(vectors)

sample_embedding.shape
rnn_layer = nn.RNN(input_size=200,    # dimension of the input repr

                   hidden_size=200,   # dimension of the hidden units

                   batch_first=True) # input format is [batch_size, seq_len, repr_dim]





sample = sample_embedding.unsqueeze(0) # add the batch_size dimension

h0 = torch.zeros(1, 1, 200)           # initial hidden state

out, last_hidden = rnn_layer(sample, h0)
sample.shape
out.shape
''' also true '''

out, last_hidden = rnn_layer(sample) 
# vectors shape [1] after go through preprocessing

# sample_emb sha[1, 1]

# sample_with_batch [1,1,1] #### this is what RNN and LSTM expects
class RNN(nn.Module):

    def __init__(self, input_size, hidden_size, output):

        super(RNN,self).__init__()

        

        self.embedding = nn.Embedding.from_pretrained(glove.vectors)

        

        self.hidden_size = hidden_size

        

        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True) # batch_first

        

        self.fc = nn.Linear(hidden_size, output)

        

    def forward(self, x):

        

#         h0 = torch.zeros(1, x.size(0), self.hidden_size)

        emb = self.embedding(x)

#         out, _ = self.rnn(emb, h0)

        

        out, _ = self.rnn(emb)

        last_out = out[:, -1, :]

        out = self.fc(last_out)

        

        return out

    

model = RNN(200, 200, 2)
'''you can make use of it, i know!'''

samplebatch = train[199]

ds = DataLoader(samplebatch, shuffle=True)
# to use pad_packed_sequence

'''do you want pad_sequence?'''

from torch.nn.utils.rnn import pad_sequence



sample_padded = pad_sequence([sequences for sequences, label in train[:10]], batch_first=True)

sample_padded.shape
out = model(sample_padded)
for i in range(10):

    j = train[i]

    print(j[0].shape)
import random



class TweetBatcher:

    def __init__(self, tweets, batch_size=32, drop_last=False):

        # store tweets by length

        self.tweets_by_length = {}

        for words, label in tweets:

            # compute the length of the tweet

            wlen = words.shape[0]

            # put the tweet in the correct key inside self.tweet_by_length

            if wlen not in self.tweets_by_length:

                self.tweets_by_length[wlen] = []

            self.tweets_by_length[wlen].append((words, label),)

         

        #  create a DataLoader for each set of tweets of the same length

        self.loaders = {wlen : torch.utils.data.DataLoader(

                                    tweets,

                                    batch_size=batch_size,

                                    shuffle=True,

                                    drop_last=drop_last) # omit last batch if smaller than batch_size

            for wlen, tweets in self.tweets_by_length.items()}

        

    def __iter__(self): # called by Python to create an iterator

        # make an iterator for every tweet length

        iters = [iter(loader) for loader in self.loaders.values()]

        while iters:

            # pick an iterator (a length)

            im = random.choice(iters)

            try:

                yield next(im)

            except StopIteration:

                # no more elements in the iterator, remove it

                iters.remove(im)
for i, (vectors, labels) in enumerate(TweetBatcher(train, drop_last=True)):

    if i > 5: break

    print(vectors.shape, labels.shape)
def get_accuracy(model, data_loader):

    correct, total = 0, 0

    for tweets, labels in data_loader:

        output = model(tweets)

        pred = output.max(1, keepdim=True)[1]

        correct += pred.eq(labels.view_as(pred)).sum().item()

        total += labels.shape[0]

    return correct / total



test_loader = TweetBatcher(test, batch_size=64, drop_last=False)

get_accuracy(model, test_loader)
def train_rnn_network(model, train, valid, num_epochs=5, lr=0.001):

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    

    losses, train_acc, valid_acc = [], [], []

    epochs = []

    

    for epoch in range(num_epochs):

        for x, y in train:

            optimizer.zero_grad()

            pred = model(x)

            loss = criterion(pred, y)

            loss.backward()

            optimizer.step()

            

        losses.append(float(loss))

        

        epochs.append(epoch)

        train_acc.append(get_accuracy(model, train_loader))

        valid_acc.append(get_accuracy(model, valid_loader))

        print("Epoch %d; Loss %f; Train Acc %f; Val Acc %f" % (

              epoch+1, loss, train_acc[-1], valid_acc[-1]))

    # plotting

    plt.title("Training Curve")

    plt.plot(losses, label="Train")

    plt.xlabel("Epoch")

    plt.ylabel("Loss")

    plt.show()



    plt.title("Training Curve")

    plt.plot(epochs, train_acc, label="Train")

    plt.plot(epochs, valid_acc, label="Validation")

    plt.xlabel("Epoch")

    plt.ylabel("Accuracy")

    plt.legend(loc='best')

    plt.show()

model = RNN(200, 200, 2)
train_loader = TweetBatcher(train, batch_size=64, drop_last=True)

valid_loader = TweetBatcher(valid, batch_size=64, drop_last=False)

train_rnn_network(model, train_loader, valid_loader, num_epochs=20, lr=0.01)

get_accuracy(model, test_loader)
class TweetLSTM(nn.Module):

    def __init__(self, input_size, hidden_size, num_classes):

        super(TweetLSTM, self).__init__()

        self.emb = nn.Embedding.from_pretrained(glove.vectors)

        self.hidden_size = hidden_size

        self.rnn = nn.LSTM(input_size, hidden_size, batch_first=True)

        self.fc = nn.Linear(hidden_size, num_classes)

    

    def forward(self, x):

        # Look up the embedding

        x = self.emb(x)

        # Set an initial hidden state and cell state

#         h0 = torch.zeros(1, x.size(0), self.hidden_size)

#         c0 = torch.zeros(1, x.size(0), self.hidden_size)

        # Forward propagate the LSTM

        out, _ = self.rnn(x)

        # Pass the output of the last time step to the classifier

        out = self.fc(out[:, -1, :])

        return out



model = TweetLSTM(200, 50, 2)

train_rnn_network(model, train_loader, valid_loader, num_epochs=20, lr=0.001)

get_accuracy(model, test_loader)
class TweetGRU(nn.Module):

    def __init__(self, input_size, hidden_size, num_classes):

        super(TweetGRU, self).__init__()

        self.emb = nn.Embedding.from_pretrained(glove.vectors)

        self.hidden_size = hidden_size

        self.rnn = nn.GRU(input_size, hidden_size, batch_first=True)

        self.fc = nn.Linear(hidden_size, num_classes)

    

    def forward(self, x):

        # Look up the embedding

        x = self.emb(x)

        # Set an initial hidden state

        h0 = torch.zeros(1, x.size(0), self.hidden_size)

        # Forward propagate the GRU 

        out, _ = self.rnn(x, h0)

        # Pass the output of the last time step to the classifier

        out = self.fc(out[:, -1, :])

        return out



model = TweetGRU(200, 50, 2)

train_rnn_network(model, train_loader, valid_loader, num_epochs=20, lr=0.001)

get_accuracy(model, test_loader)