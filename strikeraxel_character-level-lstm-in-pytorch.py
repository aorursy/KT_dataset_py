# Import Libs

import numpy as np

import nltk

import torch

import torch.nn as nn

import torch.nn.functional as F
# Step 1: Load Data

with open('/kaggle/input/charrnn/data/data/anna.txt', 'r') as f:

    raw_txt = f.read()

# Step 2: Get int2char

def get_unique_chars_dict(txt):

    unique_chars = tuple(set(txt))

    result = dict(enumerate(unique_chars))

    return result



chars = tuple(set(raw_txt))

int2char = get_unique_chars_dict(raw_txt)

char2int = {k: i  for i, k in int2char.items()}



# Step 3: Replace encode all char to int

encoded_txt = np.array([char2int[ch] for ch in raw_txt])

# Step 4: one-hot encoding of words

def get_one_hot_encoding(txt_seq, num_char):

    one_hot_vec = np.zeros((txt_seq.size, num_char), dtype=np.float32)

    one_hot_vec[np.arange(one_hot_vec.shape[0]), txt_seq.flatten()] = 1.

    one_hot_vec = one_hot_vec.reshape((*txt_seq.shape, num_char))

    return one_hot_vec



# get_one_hot_encoding(np.array([55,60,30]), len(char2int)) # Test
def get_mini_batch(encoding_txt,batch_size ,seq_length):

    batch_size_total = batch_size * seq_length

    total_batches = encoding_txt.shape[0] // batch_size_total

    encoding_txt = encoding_txt[:total_batches * batch_size_total]

    encoding_txt = encoding_txt.reshape((batch_size,-1))

    for i in range(0, encoding_txt.shape[1], seq_length):

        x = encoding_txt[:,i:i+seq_length]

        y = np.zeros_like(x)

        try:

            y[:,:-1], y[:,-1] = x[:,1:], x[:,i+seq_length]

        except IndexError:

            y[:,:-1], y[:,-1] = x[:,1:], x[:,0]

        yield x,y



# Test mini-batch

# batches = get_mini_batch(encoded_txt, 7, 10)

# x,y = next(batches)
class CharRNN(nn.Module):

    def __init__(self,tokens, n_hidden=256, n_layers=2, dropout_prob=0.5, lr=0.0001):

        super().__init__()

        

        self.chars = tokens

        self.n_hidden = n_hidden

        self.n_layers = n_layers

        self.lr = lr

        self.int2char = dict(enumerate(tokens))

        self.char2int = { c:i for i, c in self.int2char.items() }

        

        # Define layers

        self.lstm = nn.LSTM(len(tokens), n_hidden, n_layers, dropout=dropout_prob, batch_first=True)

        self.dropout = nn.Dropout(dropout_prob)

        self.fc = nn.Linear(n_hidden, len(self.chars))

        

    

    def forward(self, x, hidden):

        out, hidden = self.lstm(x,hidden)

        out = self.dropout(out)

        out = out.contiguous().view(-1, self.n_hidden)

        out = self.fc(out)

        return out, hidden



    def init_hidden(self, batch_size):

        weight = next(self.parameters()).data        

        hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_(),

                  weight.new(self.n_layers, batch_size, self.n_hidden).zero_())        

        return hidden

        
def train(net,data,epochs=10,batch_size=10,seq_length=30,lr=0.0001, clip=5):

    n_chars = len(net.chars)

    

    # Set model to training mode

    net.train()

    

    # Define optimizer

    optim = torch.optim.Adam(net.parameters(),lr=lr)

    # Define loss function

    criterion = nn.CrossEntropyLoss()

    

    # Define traning and validation data

    # 90% of the data is used for traning set and 10% for validation set

    val_inx = int(len(data)*0.9)

    data, val_data = data[:val_inx], data[val_inx:]



    count = 0

    for e in range(epochs):

        h = net.init_hidden(batch_size)

        for x, y in get_mini_batch(data, batch_size, seq_length):

            count+=1

            x = get_one_hot_encoding(x, n_chars)

            x,y = torch.from_numpy(x), torch.from_numpy(y)

            h = tuple([w.data for w in h])

            net.zero_grad()

            # forward prop

            out, h = net(x,h)

            # get loss

            loss = criterion(out, y.view(seq_length*batch_size).long())

            loss.backward()

            # gradient cliping help from gradent exploding

            nn.utils.clip_grad_norm_(net.parameters(), clip)

            # optimize parameters

            optim.step()

            

            if count % 10 == 0 :

                net.eval()

                v_h = net.init_hidden(batch_size)

                val_losses = []

                for v_x, v_y in get_mini_batch(val_data, batch_size, seq_length):

                    v_x = get_one_hot_encoding(v_x, n_chars)

                    v_x,v_y = torch.from_numpy(v_x), torch.from_numpy(v_y)

                    v_h = tuple([w.data for w in v_h])

                    v_out, v_h = net(v_x,v_h)

                    # get loss

                    loss = criterion(v_out, y.view(seq_length*batch_size).long())

                    val_losses.append(loss.item())

                net.train()

                print("Epoch : {}".format(e))

                print("Steps : {}".format(count))

                print("Training Loss  : {:.4f}".format(loss.item()))

                print("Validation Loss : {:.4f}".format(np.mean(val_losses)))
n_hidden=512

n_layers=2

net = CharRNN(chars, n_hidden, n_layers)

net = CharRNN(chars)

print(net)
batch_size = 128

seq_length = 100

n_epochs = 2 



# train the model

train(net, encoded_txt, epochs=n_epochs, batch_size=batch_size, seq_length=seq_length, lr=0.01)