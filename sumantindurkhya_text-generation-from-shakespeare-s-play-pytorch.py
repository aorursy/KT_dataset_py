# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import torch
import torch.nn as nn
# import torch.nn.functional as F

from collections import Counter
import warnings

import string
import nltk

import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
warnings.filterwarnings('ignore')
torch.autograd.set_detect_anomaly(True)
batch_size = 16
seq_size = 32
embedding_size = 64
lstm_size = 64
gradients_norm = 5
# set device parameter
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# read document
with open ('../input/shakespeare-plays/alllines.txt', 'r') as f:
    doc = f.read()
# get list of words from document
def doc2words(doc):
    lines = doc.split('\n')
    lines = [line.strip(r'\"') for line in lines]
    words = ' '.join(lines).split()
    return words
def removepunct(words):
    punct = set(string.punctuation)
    words = [''.join([char for char in list(word) if char not in punct]) for word in words]
    return words
# get vocab from word list
def getvocab(words):
    wordfreq = Counter(words)
    sorted_wordfreq = sorted(wordfreq, key=wordfreq.get)
    return sorted_wordfreq
# get dictionary of int to words and word to int
def vocab_map(vocab):
    int_to_vocab = {k:w for k,w in enumerate(vocab)}
    vocab_to_int = {w:k for k,w in int_to_vocab.items()}
    return int_to_vocab, vocab_to_int
words = removepunct(doc2words(doc))
vocab = getvocab(words)
int_to_vocab, vocab_to_int = vocab_map(vocab)
def get_batches(words, vocab_to_int, batch_size, seq_size):
    # generate a Xs and Ys of shape (batchsize * num_batches) * seq_size
    word_ints = [vocab_to_int[word] for word in words]
    num_batches = int(len(word_ints) / (batch_size * seq_size))
    Xs = word_ints[:num_batches*batch_size*seq_size]
    Ys = np.zeros_like(Xs)
    Ys[:-1] = Xs[1:]
    Ys[-1] = Xs[0]
    Xs = np.reshape(Xs, (num_batches*batch_size, seq_size))
    Ys= np.reshape(Ys, (num_batches*batch_size, seq_size))
    
    # iterate over rows of Xs and Ys to generate batches
    for i in range(0, num_batches*batch_size, batch_size):
        yield Xs[i:i+batch_size, :], Ys[i:i+batch_size, :]
class RNNModule(nn.Module):
    # initialize RNN module
    def __init__(self, n_vocab, seq_size=32, embedding_size=64, lstm_size=64):
        super(RNNModule, self).__init__()
        self.seq_size = seq_size
        self.lstm_size = lstm_size
        self.embedding = nn.Embedding(n_vocab, embedding_size)
        self.lstm = nn.LSTM(embedding_size,
                            lstm_size,
                            batch_first=True)
        self.dense = nn.Linear(lstm_size, n_vocab)
        
    def forward(self, x, prev_state):
        embed = self.embedding(x)
        output, state = self.lstm(embed, prev_state)
        logits = self.dense(output)

        return logits, state
    
    def zero_state(self, batch_size):
        return (torch.zeros(1, batch_size, self.lstm_size),torch.zeros(1, batch_size, self.lstm_size))
 
def get_loss_and_train_op(net, lr=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    return criterion, optimizer
def generate_text(device, net, words, n_vocab, vocab_to_int, int_to_vocab, top_k=5):
    net.eval()

    state_h, state_c = net.zero_state(1)
    state_h = state_h.to(device)
    state_c = state_c.to(device)
    for w in words:
        ix = torch.tensor([[vocab_to_int[w]]]).to(device)
        output, (state_h, state_c) = net(ix, (state_h, state_c))
    
    _, top_ix = torch.topk(output[0], k=top_k)
    choices = top_ix.tolist()
    choice = np.random.choice(choices[0])

    words.append(int_to_vocab[choice])
    
    for _ in range(100):
        ix = torch.tensor([[choice]]).to(device)
        output, (state_h, state_c) = net(ix, (state_h, state_c))

        _, top_ix = torch.topk(output[0], k=top_k)
        choices = top_ix.tolist()
        choice = np.random.choice(choices[0])
        words.append(int_to_vocab[choice])

    print(' '.join(words))
def train_rnn(words, vocab_to_int, int_to_vocab, n_vocab):
    
    # RNN instance
    net = RNNModule(n_vocab, seq_size, embedding_size, lstm_size)
    net = net.to(device)
    criterion, optimizer = get_loss_and_train_op(net, 0.01)

    iteration = 0
    
    for e in range(50):
        batches = get_batches(words, vocab_to_int, batch_size, seq_size)
        state_h, state_c = net.zero_state(batch_size)

        # Transfer data to GPU
        state_h = state_h.to(device)
        state_c = state_c.to(device)
        for x, y in batches:
            iteration += 1

            # Tell it we are in training mode
            net.train()

            # Reset all gradients
            optimizer.zero_grad()

            # Transfer data to GPU
            x = torch.tensor(x).to(device)
            y = torch.tensor(y).to(device)

            logits, (state_h, state_c) = net(x, (state_h, state_c))
            loss = criterion(logits.transpose(1, 2), y)

            state_h = state_h.detach()
            state_c = state_c.detach()

            loss_value = loss.item()

            # Perform back-propagation
            loss.backward(retain_graph=True)

            _ = torch.nn.utils.clip_grad_norm_(net.parameters(), gradients_norm)
            
            # Update the network's parameters
            optimizer.step()

            if iteration % 100 == 0:
                print('Epoch: {}/{}'.format(e, 200),'Iteration: {}'.format(iteration),'Loss: {}'.format(loss_value))

            # if iteration % 1000 == 0:
                # predict(device, net, flags.initial_words, n_vocab,vocab_to_int, int_to_vocab, top_k=5)
                # torch.save(net.state_dict(),'checkpoint_pt/model-{}.pth'.format(iteration))
                
    return net
rnn_net = train_rnn(words, vocab_to_int, int_to_vocab, len(vocab))
generate_text(device, rnn_net, ['hey', 'you'], len(vocab), vocab_to_int, int_to_vocab)
