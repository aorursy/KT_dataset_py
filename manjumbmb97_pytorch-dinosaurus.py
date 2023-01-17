from __future__ import unicode_literals, print_function, division

import numpy as np 

import pandas as pd 

from io import open

import glob

import unicodedata

import string

import random

import torch

import torch.nn as nn

import torch.nn.functional as F

import torch.optim as optim

import matplotlib.pyplot as plt



import os

print(os.listdir("../input"))
data = open("../input/dinos.txt", "r").read()

data = data.lower()

chars = list(sorted(set(data)))

chars.pop(0) #remove \n character

data_size, chars_size = len(data), len(chars)



print(chars, "Data size: {}".format(data_size), "Chars length: {}".format(chars_size))
with open("../input/dinos.txt") as f:

    examples = f.readlines()

examples = [x.lower().strip() for x in examples]
char_to_ix = { ch:i for i, ch in enumerate(sorted(chars)) }

ix_to_char = { i:ch for i, ch in enumerate(sorted(chars)) }
class RNN(nn.Module):

    

    def __init__(self, input_size, hidden_size, output_size):

        super(RNN, self).__init__()

        

        self.hidden_size = hidden_size

        self.i2o = nn.Linear(input_size + hidden_size, output_size)

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)

        self.softmax = nn.LogSoftmax(dim=1)

    

    def forward(self, input, hidden):

        combined = torch.cat((input, hidden), 1)

        hidden = self.i2h(combined)

        output = self.i2o(combined)

        output = self.softmax(output)

        return output, hidden

    

    def initHidden(self):

        return torch.zeros(1, self.hidden_size)
def inputTensor(word):

    tensor = torch.zeros(len(word), 1, chars_size + 1)

    for i in range(len(word)):

        letter = word[i]

        tensor[i][0][char_to_ix[letter]] = 1

    return tensor



def targetTensor(word):

    letter_indexes = [char_to_ix[word[i]] for i in range(1, len(word))]

    letter_indexes.append(chars_size) # EOS

    return torch.LongTensor(letter_indexes)
rnn = RNN(chars_size + 1, 128 , chars_size + 1)



num_iterations = 35000

print_every = 2000

plot_every = 500

all_losses = []

total_loss = 0.0



criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(rnn.parameters(), lr=0.0005)

np.random.shuffle(examples)



def train(input_tensor, target_tensor):

    target_tensor.unsqueeze_(-1)

    hidden = rnn.initHidden()

    

    optimizer.zero_grad()

    

    loss = 0.0

    

    for i in range(input_tensor.size(0)):

        output, hidden = rnn(input_tensor[i], hidden)

        l = criterion(output, target_tensor[i])

        loss += l

        

    loss.backward()

    optimizer.step()

    

    return output, loss.item() / input_tensor.size(0)
def randomTrainingExample():

    word = examples[random.randint(0, len(examples) - 1)]

    input_tensor = inputTensor(word)

    target_tensor = targetTensor(word)

    return input_tensor, target_tensor
max_length = 20



# Sample from a category and starting letter

def sample(start_letter='A'):

    with torch.no_grad():  # no need to track history in sampling

        input = inputTensor(start_letter.lower())

        hidden = rnn.initHidden()



        output_name = start_letter



        for i in range(max_length):

            output, hidden = rnn(input[0], hidden)

            topv, topi = output.topk(1)

            topi = topi[0][0]

            if topi == chars_size:

                break

            else:

                letter = ix_to_char[topi.item()]

                output_name += letter

            input = inputTensor(letter)



        return output_name



def samples(start_letters='ABC'):

    for start_letter in start_letters:

        print(sample(start_letter))
for iter in range(1, num_iterations + 1):

    output, loss = train(*randomTrainingExample())

    total_loss += loss



    if iter % print_every == 0:

        print('(%d %d %%) %.4f' % (iter, iter / num_iterations * 100, loss))

        for name in range(5):

            print(sample(ix_to_char[random.randint(0, chars_size-1)]))



    if iter % plot_every == 0:

        all_losses.append(total_loss / plot_every)

        total_loss = 0.0
plt.figure()

plt.plot(all_losses)

plt.xlabel('Iteration')

plt.ylabel('Loss')
for i in range(len(ix_to_char)):

    print(sample(ix_to_char[i]))