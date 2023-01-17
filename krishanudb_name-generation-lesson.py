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
from __future__ import unicode_literals, print_function, division

from io import open

import glob

import os

import unicodedata

import string
all_letters = string.ascii_letters + " ,.;'"

n_letters = len(all_letters) + 1



def find_files(path): return glob.glob(path)
find_files("../input/data/data/names/*.txt")
def unicode_to_ascii(s):

    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != "Mn" and c in all_letters)
def read_lines(filename):

    lines = open(filename, encoding="utf-8").read().strip().split("\n")

    return [unicode_to_ascii(line) for line in lines]
category_lines = {}

all_categories = []



for filename in find_files("../input/data/data/names/*.txt"):

    category = os.path.splitext(os.path.basename(filename))[0]

    category_lines[category] = read_lines(filename)

    all_categories.append(category)

    

n_categories = len(all_categories)
print('# categories:', n_categories, all_categories)

print(unicode_to_ascii("O'Néàl"))
import torch

import torch.nn as nn



class RNN(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):

        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        self.input_to_hidden = nn.Linear(n_categories + input_size + hidden_size, hidden_size)

        self.input_to_output = nn.Linear(n_categories + input_size + hidden_size, output_size)

        self.output_to_output = nn.Linear(hidden_size + output_size, output_size)

        self.dropout = nn.Dropout(0.1)

        self.softmax = nn.LogSoftmax(dim = 1)

        

    def forward(self, category, input, hidden):

        input_combined = torch.cat((category, input, hidden), 1)

        hidden = self.input_to_hidden(input_combined)

        output = self.input_to_output(input_combined)

        output_combined = torch.cat((output, hidden), 1)

        output = self.output_to_output(output_combined)

        output = self.dropout(output)

        output = self.softmax(output)

        return output, hidden

    

    def initialize_hidden(self):

        return torch.zeros(1, self.hidden_size)
import random



def random_choice(l):

    return l[random.randint(0, len(l) - 1)]



def random_training_pair():

    category = random_choice(all_categories)

    line = random_choice(category_lines[category])

    return category, line
def category_tensor(category):

    li = all_categories.index(category)

    tensor = torch.zeros(1, n_categories)

    tensor[0][li] = 1.

    return tensor



def input_tensor(line):

    tensor = torch.zeros(len(line), 1, n_letters)

    for li in range(len(line)):

        tensor[li][0][all_letters.find(line[li])] = 1.

    return tensor



def target_tensor(line):

    letter_indices = [all_letters.find(line[li]) for li in range(1, len(line))]

    letter_indices.append(n_letters - 1)

    return torch.LongTensor(letter_indices)
def random_training_example():

    category, line = random_training_pair()

    category_tensor_ = category_tensor(category)

    input_tensor_ = input_tensor(line)

    target_tensor_ = target_tensor(line)

    return category_tensor_, input_tensor_, target_tensor_
criterion = nn.NLLLoss()

learning_rate = 0.0005



def train(category_tensor, input_line_tensor, target_line_tensor):

    target_line_tensor.unsqueeze_(-1)

    hidden = rnn.initialize_hidden()

    

    rnn.zero_grad()

    loss = 0

    

    for i in range(input_line_tensor.size()[0]):

        output, hidden = rnn(category_tensor, input_line_tensor[i], hidden)

        l = criterion(output, target_line_tensor[i])

        loss += l

    

    loss.backward()

    for p in rnn.parameters():

        p.data.add_(-learning_rate, p.grad.data)

        

    return output, loss.item()/input_line_tensor.size()[0]
import math

import time



def time_since(since):

    now = time.time()

    s = now - since

    m = math.floor(s / 60)

    s = s - m * 60

    return "%dm, %ds" %(m, s)
rnn = RNN(n_letters, 128, n_letters)



# rnn = rnn.cuda()



n_iters = 200000

print_every = 10000

plot_every = 500

all_losses = []

total_loss = 0



start = time.time()



for iter in range(1, n_iters + 1):

    output, loss = train(*random_training_example())

    total_loss += loss

    

    if iter % print_every == 0:

        print('%s (%d %d%%) %.4f' % (time_since(start), iter, iter / n_iters * 100, loss))

        

    if iter % plot_every == 0:

        all_losses.append(total_loss / plot_every)

        total_loss = 0
import matplotlib.pyplot as plt



plt.plot(all_losses)

plt.show()
max_length = 20



def sample(category, start_letter = 'A'):

    with torch.no_grad():

        category_tensor_ = category_tensor(category)

        input = input_tensor(start_letter)

        hidden = rnn.initialize_hidden()

        

        output_name = start_letter

        

        for i in range(max_length):

            output, hidden = rnn(category_tensor_, input[0], hidden)

            topv, topi = output.topk(1)

            topi = topi[0][0]

            if topi == n_letters - 1:

                break

            else:

                output_name += all_letters[topi]

            input = input_tensor(all_letters[topi])

    return output_name



def samples(category, start_letters = "ABC"):

    for letter in start_letters:

        print(sample(category, letter))

    
samples("German", "YZD")
samples("Russian", "KHF")