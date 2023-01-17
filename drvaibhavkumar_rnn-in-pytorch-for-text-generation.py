from __future__ import unicode_literals, print_function, division

from io import open

import glob

import os

import unicodedata

import string

import torch

import torch.nn as nn

import random

import time

import math

import matplotlib.pyplot as plt

import matplotlib.ticker as ticker
all_letters = string.ascii_letters + " .,;'-"

n_letters = len(all_letters) + 1
def findFiles(path):

  return glob.glob(path)
# Turn a Unicode string to plain ASCII, thanks to https://stackoverflow.com/a/518232/2809427

def unicodeToAscii(s):

    return ''.join(

        c for c in unicodedata.normalize('NFD', s)

        if unicodedata.category(c) != 'Mn'

        and c in all_letters

    )
# Read a file and split into lines

def readLines(filename):

    lines = open(filename, encoding='utf-8').read().strip().split('\n')

    return [unicodeToAscii(line) for line in lines]



# Build the category_lines dictionary, a list of lines per category

category_lines = {}

all_categories = []

for filename in findFiles('../input/name-languages/*.txt'):

    category = os.path.splitext(os.path.basename(filename))[0]

    all_categories.append(category)

    lines = readLines(filename)

    category_lines[category] = lines



n_categories = len(all_categories)
class NameGeneratorModule(nn.Module):

    def __init__(self, inp_size, hid_size, op_size):

        super(NameGeneratorModule, self).__init__()

        self.hid_size = hid_size



        self.i2h = nn.Linear(n_categories + inp_size + hid_size, hid_size)

        self.i2o = nn.Linear(n_categories + inp_size + hid_size, op_size)

        self.o2o = nn.Linear(hid_size + op_size, op_size)

        self.dropout = nn.Dropout(0.1)

        self.softmax = nn.LogSoftmax(dim=1)



    def forward(self, category, input, hidden):

        inp_comb = torch.cat((category, input, hidden), 1)

        hidden = self.i2h(inp_comb)

        output = self.i2o(inp_comb)

        op_comb = torch.cat((hidden, output), 1)

        output = self.o2o(op_comb)

        output = self.dropout(output)

        output = self.softmax(output)

        return output, hidden



    def initHidden(self):

        return torch.zeros(1, self.hid_size)
# Random item from a list

def randomChoice(l):

    return l[random.randint(0, len(l) - 1)]



# Get a random category and random line from that category

def randomTrainingPair():

    category = randomChoice(all_categories)

    line = randomChoice(category_lines[category])

    return category, line
# One-hot vector for category

def categoryTensor(category):

    li = all_categories.index(category)

    tensor = torch.zeros(1, n_categories)

    tensor[0][li] = 1

    return tensor



# One-hot matrix of first to last letters (not including EOS) for input

def inputTensor(line):

    tensor = torch.zeros(len(line), 1, n_letters)

    for li in range(len(line)):

        letter = line[li]

        tensor[li][0][all_letters.find(letter)] = 1

    return tensor



# LongTensor of second letter to end (EOS) for target

def targetTensor(line):

    letter_indexes = [all_letters.find(line[li]) for li in range(1, len(line))]

    letter_indexes.append(n_letters - 1) # EOS

    return torch.LongTensor(letter_indexes)
# Make category, input, and target tensors from a random category, line pair

def randomTrainingExample():

    category, line = randomTrainingPair()

    category_tensor = categoryTensor(category)

    input_line_tensor = inputTensor(line)

    target_line_tensor = targetTensor(line)

    return category_tensor, input_line_tensor, target_line_tensor
#Loss

criterion = nn.NLLLoss()

#Learning rate

learning_rate = 0.0005



def train(category_tensor, input_line_tensor, target_line_tensor):

    target_line_tensor.unsqueeze_(-1)

    hidden = model.initHidden()



    model.zero_grad()



    loss = 0



    for i in range(input_line_tensor.size(0)):

        output, hidden = model(category_tensor, input_line_tensor[i], hidden)

        l = criterion(output, target_line_tensor[i])

        loss += l



    loss.backward()



    for p in model.parameters():

        p.data.add_(p.grad.data, alpha=-learning_rate)



    return output, loss.item() / input_line_tensor.size(0)
def timeSince(since):

    now = time.time()

    s = now - since

    m = math.floor(s / 60)

    s -= m * 60

    return '%dm %ds' % (m, s)
model = NameGeneratorModule(n_letters, 128, n_letters)
print(model)
epochs = 100000

print_every = 5000

plot_every = 500

all_losses = []

total_loss = 0 # Reset every plot_every iters



start = time.time()



for iter in range(1, epochs + 1):

    output, loss = train(*randomTrainingExample())

    total_loss += loss



    if iter % print_every == 0:

        print('Time: %s, Epoch: (%d - Total Iterations: %d%%),  Loss: %.4f' % (timeSince(start), iter, iter / epochs * 100, loss))



    if iter % plot_every == 0:

        all_losses.append(total_loss / plot_every)

        total_loss = 0
plt.figure(figsize=(7,7))

plt.title("Loss")

plt.plot(all_losses)

plt.xlabel("Epochs")

plt.ylabel("Loss")

plt.show()
max_length = 20



# Sample from a category and starting letter

def sample_model(category, start_letter='A'):

    with torch.no_grad():  # no need to track history in sampling

        category_tensor = categoryTensor(category)

        input = inputTensor(start_letter)

        hidden = model.initHidden()



        output_name = start_letter



        for i in range(max_length):

            output, hidden = model(category_tensor, input[0], hidden)

            topv, topi = output.topk(1)

            topi = topi[0][0]

            if topi == n_letters - 1:

                break

            else:

                letter = all_letters[topi]

                output_name += letter

            input = inputTensor(letter)



        return output_name



# Get multiple samples from one category and multiple starting letters

def sample_names(category, start_letters='ABC'):

    for start_letter in start_letters:

        print(sample_model(category, start_letter))
print("Italian:-")

sample_names('Italian', 'BPRT')

print("\nKorean:-")

sample_names('Korean', 'CMRS')

print("\nRussian:-")

sample_names('Russian', 'AJLN')

print("\nVietnamese:-")

sample_names('Vietnamese', 'LMT')