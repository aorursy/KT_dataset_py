# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os



import numpy as np

import torch

from torch import nn

import torch.nn.functional as F



# Any results you write to the current directory are saved as output.
os.listdir('../input/himu-book')
with open('../input/himu-book/himu.txt', 'r') as f:

    text = f.read()
text[:500]
chars = tuple(set(text))

int2char = dict(enumerate(chars))

char2int = {ch: ii for ii, ch in int2char.items()}



# encode the text

encoded = np.array([char2int[ch] for ch in text])



encoded[:100]
def one_hot_encode(arr, n_labels):

    one_hot = np.zeros((arr.size, n_labels), dtype=np.float32)

    

    one_hot[np.arange(one_hot.shape[0]), arr.flatten()] = 1.

    

    one_hot = one_hot.reshape((*arr.shape, n_labels))

    

    return one_hot
test_seq = np.array([[3, 5, 1]])

one_hot = one_hot_encode(test_seq, 8)



print(one_hot)
def get_batches(arr, batch_size, seq_length):

    

    batch_size_total = batch_size * seq_length

    

    n_batches = len(arr)//batch_size_total

    arr = arr[:n_batches * batch_size_total]

    arr = arr.reshape((batch_size, -1))

    

    for n in range(0, arr.shape[1], seq_length):

        x = arr[:, n:n+seq_length]

        y = np.zeros_like(x)

        try:

            y[:, :-1], y[:, -1] = x[:, 1:], arr[:, n+seq_length]

        except IndexError:

            y[:, :-1], y[:, -1] = x[:, 1:], arr[:, 0]

        yield x, y
batches = get_batches(encoded, 10, 50)

x, y = next(batches)
print('x\n', x[:10, :10])

print('\ny\n', y[:10, :10])
train_on_gpu = torch.cuda.is_available()
class CharRNN(nn.Module):

    

    def __init__(self, tokens, n_hidden=256, n_layers=2,

                               drop_prob=0.4, lr=0.001):

        super().__init__()

        self.drop_prob = drop_prob

        self.n_layers = n_layers

        self.n_hidden = n_hidden

        self.lr = lr

        

        self.chars = tokens

        self.int2char = dict(enumerate(self.chars))

        self.char2int = {ch: ii for ii, ch in self.int2char.items()}

        

        self.lstm1 = nn.LSTM(len(self.chars), n_hidden, n_layers, 

                            dropout=drop_prob, batch_first=True)

        

        self.dropout = nn.Dropout(drop_prob)

        

        self.fc1 = nn.Linear(n_hidden, int(n_hidden/2))

        self.fc2 = nn.Linear(int(n_hidden/2), int(n_hidden/2))

        self.fc3 = nn.Linear(int(n_hidden/2), len(self.chars))

      

    

    def forward(self, x, hidden):

        

        r_output, hidden = self.lstm1(x, hidden)

        

        out = self.dropout(r_output)

        

        out = out.contiguous().view(-1, self.n_hidden)

        

        out = self.fc1(out)

        out = self.fc2(out)

        out = self.fc3(out)

        

        return out, hidden

    

    

    def init_hidden(self, batch_size):

        weight = next(self.parameters()).data

        

        if (train_on_gpu):

            hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda(),

                  weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda())

        else:

            hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_(),

                      weight.new(self.n_layers, batch_size, self.n_hidden).zero_())

        

        return hidden
# define and print the net

n_hidden=1024

n_layers=5



net = CharRNN(chars, n_hidden, n_layers)

print(net)
def train(net, data, epochs=10, batch_size=10, seq_length=10, lr=0.001, clip=5, val_frac=0.15, print_every=10):

    

    net.train()

    

    opt = torch.optim.Adam(net.parameters(), lr=lr)

    criterion = nn.CrossEntropyLoss()

    

    # create training and validation data

    val_idx = int(len(data)*(1-val_frac))

    data, val_data = data[:val_idx], data[val_idx:]

    

    if(train_on_gpu):

        net.cuda()

    

    counter = 0

    n_chars = len(net.chars)

    for e in range(epochs):

        # initialize hidden state

        h = net.init_hidden(batch_size)

        

        for x, y in get_batches(data, batch_size, seq_length):

            counter += 1

            

            # One-hot encode our data and make them Torch tensors

            x = one_hot_encode(x, n_chars)

            inputs, targets = torch.from_numpy(x), torch.from_numpy(y)

            

            if(train_on_gpu):

                inputs, targets = inputs.cuda(), targets.cuda()



            # Creating new variables for the hidden state, otherwise

            # we'd backprop through the entire training history

            h = tuple([each.data for each in h])



            # zero accumulated gradients

            net.zero_grad()

            

            # get the output from the model

            output, h = net(inputs, h)

            

            # calculate the loss and perform backprop

            loss = criterion(output, targets.view(batch_size*seq_length).long())

            loss.backward()

            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.

            nn.utils.clip_grad_norm_(net.parameters(), clip)

            opt.step()

            

            # loss stats

            if counter % print_every == 0:

                # Get validation loss

                val_h = net.init_hidden(batch_size)

                val_losses = []

                net.eval()

                for x, y in get_batches(val_data, batch_size, seq_length):

                    # One-hot encode our data and make them Torch tensors

                    x = one_hot_encode(x, n_chars)

                    x, y = torch.from_numpy(x), torch.from_numpy(y)

                    

                    # Creating new variables for the hidden state, otherwise

                    # we'd backprop through the entire training history

                    val_h = tuple([each.data for each in val_h])

                    

                    inputs, targets = x, y

                    if(train_on_gpu):

                        inputs, targets = inputs.cuda(), targets.cuda()



                    output, val_h = net(inputs, val_h)

                    val_loss = criterion(output, targets.view(batch_size*seq_length).long())

                

                    val_losses.append(val_loss.item())

                

                net.train() # reset to train mode after iterationg through validation data

                

                print("Epoch: {}/{}...".format(e+1, epochs),

                      "Step: {}...".format(counter),

                      "Loss: {:.4f}...".format(loss.item()),

                      "Val Loss: {:.4f}".format(np.mean(val_losses)))
batch_size = 128

seq_length = 10

n_epochs = 30



# train the model

train(net, encoded, epochs=n_epochs, batch_size=batch_size, seq_length=seq_length, lr=0.001, print_every=50)
def predict(net, char, h=None, top_k=None):

        ''' Given a character, predict the next character.

            Returns the predicted character and the hidden state.

        '''

        

        # tensor inputs

        x = np.array([[net.char2int[char]]])

        x = one_hot_encode(x, len(net.chars))

        inputs = torch.from_numpy(x)

        

        if(train_on_gpu):

            inputs = inputs.cuda()

        

        # detach hidden state from history

        h = tuple([each.data for each in h])

        # get the output of the model

        out, h = net(inputs, h)



        # get the character probabilities

        p = F.softmax(out, dim=1).data

        if(train_on_gpu):

            p = p.cpu() # move to cpu

        

        # get top characters

        if top_k is None:

            top_ch = np.arange(len(net.chars))

        else:

            p, top_ch = p.topk(top_k)

            top_ch = top_ch.numpy().squeeze()

        

        # select the likely next character with some element of randomness

        p = p.numpy().squeeze()

        char = np.random.choice(top_ch, p=p/p.sum())

        

        # return the encoded value of the predicted char and the hidden state

        return net.int2char[char], h
def sample(net, size, prime='হিমু ', top_k=None):

        

    if(train_on_gpu):

        net.cuda()

    else:

        net.cpu()

    

    net.eval() # eval mode

    

    # First off, run through the prime characters

    chars = [ch for ch in prime]

    h = net.init_hidden(1)

    for ch in prime:

        char, h = predict(net, ch, h, top_k=top_k)



    chars.append(char)

    

    # Now pass in the previous character and get a new one

    for ii in range(size):

        char, h = predict(net, chars[-1], h, top_k=top_k)

        chars.append(char)



    return ''.join(chars)
print(sample(net, 1000, prime='হিমু ', top_k=5))