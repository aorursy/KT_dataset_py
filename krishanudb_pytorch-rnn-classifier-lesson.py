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
def FindFiles(path): return glob.glob(path)



print(FindFiles("../input/data/data/names/*.txt"))
import unicodedata

import string
all_letters = string.ascii_letters + " ,.;'"

n_letters = len(all_letters)

print(n_letters)

def UnicodetoASCII(s):

    return "".join(

    c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c)!= "Mn" and c in all_letters)



print(UnicodetoASCII("ŚlusàrskiMesut"))
def readlines(filename):

    lines = open(filename, encoding = "utf-8").read().strip().split("\n")

    return [UnicodetoASCII(line) for line in lines]



category_lines = {}

all_categories = []



for filename in FindFiles("../input/data/data/names/*.txt"):

#     category = os.path.basename(filename)

#     print(category)

#     category = os.path.splitext(category)

#     print(category)

#     category = category[0]

#     print(category)

    category = os.path.splitext(os.path.basename(filename))[0]

    all_categories.append(category)

    lines = readlines(filename)

    category_lines[category] = lines



n_categories = len(all_categories)
for key, value in category_lines.items():

    print(key,": ", value[0:5])
import torch



def letter_to_index(letter):

    return all_letters.find(letter)



def letter_to_tensor(letter):

    tensor = torch.zeros(1, n_letters)

    tensor[0][letter_to_index(letter)] = 1.

    return tensor



def line_to_tensor(line):

    tensor = torch.zeros(len(line), 1, n_letters)

    for li, letter in enumerate(line):

        tensor[li][0][letter_to_index(letter)] = 1.

    return tensor





print(line_to_tensor("Krishanu").size())

print(line_to_tensor("Krishanu"))

print(letter_to_tensor("u"))
import torch.nn as nn



class RNN(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):

        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        self.input_size = input_size

        self.output_size = output_size

        

        self.input_to_hidden = nn.Linear(input_size + hidden_size, hidden_size)

        self.input_to_output = nn.Linear(input_size + hidden_size, output_size)

        

        self.softmax = nn.LogSoftmax(dim = 1)

        

    def forward(self, input, hidden):

        combined = torch.cat((input, hidden), 1)

        hidden = self.input_to_hidden(combined)

        output = self.input_to_output(combined)

        

        output = self.softmax(output)

        return output, hidden

    

    def initialize_hidden(self):

        return torch.zeros(1, self.hidden_size)

    

n_hidden = 128

rnn = RNN(n_letters, n_hidden, n_categories)
input = letter_to_tensor("A")

hidden = torch.zeros(1, n_hidden)

output, next_hidden = rnn(input, hidden)

print(output)
input = line_to_tensor("Krishanu")

hidden = torch.zeros(1, n_hidden)

output, hidden = rnn(input[0], hidden)

print(output)
def category_from_top_k(output):

    top_n, top_i = output.topk(1)

    category_i = top_i[0].item()

    return all_categories[category_i], category_i
category_from_top_k(output)
import random



def random_choice(l):

    return l[random.randint(0, len(l) - 1)]



def random_training_example():

    category = random_choice(all_categories)

    line = random_choice(category_lines[category])

    category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long)

    line_tensor = line_to_tensor(line)

    return category, line, category_tensor, line_tensor



for i in range(10):

    category, line, category_tensor, line_tensor = random_training_example()

    print("Language: ", category, "; Name: ", line, "; Cat Tensor", category_tensor, ";\nName Tensor: ", line_tensor)
criterion = nn.NLLLoss()
learning_rate = 0.005



def train(category_tensor, line_tensor):

    hidden = rnn.initialize_hidden()

    

    rnn.zero_grad()

    

    for i in range(line_tensor.size()[0]):

        output, hidden = rnn(line_tensor[i], hidden)

        

    loss = criterion(output, category_tensor)

    

    loss.backward()

    

    for p in rnn.parameters():

        p.data.add_(-learning_rate, p.grad.data)

        

    return output, loss.item()
import time

import math



n_iters = 200000

print_every = 10000

plot_every = 1000



current_loss = 0

all_losses = []



def timeSince(since):

    now = time.time()

    s = now - since

    m = math.floor(s/60)

    s -=m * 60

    return "%dm, %ds" % (m, s)



start = time.time()



for iter in range(1, n_iters + 1):

    category, line, category_tensor, line_tensor = random_training_example()

    output, loss = train(category_tensor, line_tensor)

    current_loss += loss

    

    if iter % print_every == 0:

        guess, guess_i = category_from_top_k(output)

        correct = '✓' if guess == category else '✗ (%s)' % category

        print('%d %d%% (%s) %.4f %s / %s %s' % (iter, iter / n_iters * 100, timeSince(start), loss, line, guess, correct))

        

    if iter % plot_every == 0:

        all_losses.append(current_loss / plot_every)

        current_loss = 0
import matplotlib.pyplot as plt

# import matplotlib.ticker as ticker



# plt.figure()

plt.plot(all_losses)

plt.show()
def evaluate(line_tensor):

    hidden = rnn.initialize_hidden()

    for i in range(line_tensor.size()[0]):

        output, hidden = rnn(line_tensor[i], hidden)

    return output



def predict(name):

    line_tensor = line_to_tensor(name)

    with torch.no_grad():

        output = evaluate(line_tensor)

    

    top_n, top_i = output.topk(1)

    category_i = top_i[0].item()

    print(name, "; prediction: ", all_categories[category_i])
predict("Krishanu")