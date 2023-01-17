from __future__ import unicode_literals, print_function, division

from io import open

import glob

import os



def printFiles(path):

  return glob.glob(path)



printFiles('../input/name-languages/*.txt')
import unicodedata

import string



all_let = string.ascii_letters + " .,;'"

n_let = len(all_let)



def unicodeToAscii(s):

    return ''.join(

        c for c in unicodedata.normalize('NFD', s)

        if unicodedata.category(c) != 'Mn'

        and c in all_let

    )
cat_line = {}

all_cats = []



# Read a file and split into lines

def readLines(filename):

    lines = open(filename, encoding='utf-8').read().strip().split('\n')

    return [unicodeToAscii(line) for line in lines]



for filename in printFiles('../input/name-languages/*.txt'):

    category = os.path.splitext(os.path.basename(filename))[0]

    all_cats.append(category)

    lines = readLines(filename)

    cat_line[category] = lines



n_categories = len(all_cats)
#Check names in a category

print(cat_line['Japanese'][:4])
import torch

# Find letter index from all_let, e.g. "a" = 0

def letterToIndex(letter):

    return all_let.find(letter)



# Turn a letter into a <1 x n_let> Tensor

def letterToTensor(letter):

    tensor = torch.zeros(1, n_let)

    tensor[0][letterToIndex(letter)] = 1

    return tensor



# Turn a line into a <line_length x 1 x n_let>,

# or an array of one-hot letter vectors

def lineToTensor(line):

    tensor = torch.zeros(len(line), 1, n_let)

    for li, letter in enumerate(line):

        tensor[li][0][letterToIndex(letter)] = 1

    return tensor
print(letterToTensor('K'))

print(lineToTensor('Kakinomoto').size())
import torch.nn as nn



class RNN(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):

        super(RNN, self).__init__()



        self.hidden_size = hidden_size



        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)

        self.i2o = nn.Linear(input_size + hidden_size, output_size)

        self.softmax = nn.LogSoftmax(dim=1)



    def forward(self, input, hidden):

        combined = torch.cat((input, hidden), 1)

        hidden = self.i2h(combined)

        output = self.i2o(combined)

        output = self.softmax(output)

        return output, hidden



    def initHidden(self):

        return torch.zeros(1, self.hidden_size)



n_hidden = 128

#Binding model

rnn = RNN(n_let, n_hidden, n_categories)
input = lineToTensor('Aalsburg')

hidden = torch.zeros(1, n_hidden)



output, next_hidden = rnn(input[0], hidden)

print(output)
import random



def randomChoice(l):

    return l[random.randint(0, len(l) - 1)]



def randomTrainingExample():

    category = randomChoice(all_cats)

    line = randomChoice(cat_line[category])

    category_tensor = torch.tensor([all_cats.index(category)], dtype=torch.long)

    line_tensor = lineToTensor(line)

    return category, line, category_tensor, line_tensor



#Check on a random sample

for i in range(10):

    category, line, category_tensor, line_tensor = randomTrainingExample()

    print('category =', category, '/ line =', line)
def categoryFromOutput(output):

    top_n, top_i = output.topk(1)

    category_i = top_i[0].item()

    return all_cats[category_i], category_i

#Check category for an output

print(categoryFromOutput(output))
criterion = nn.NLLLoss()



learning_rate = 0.005 



def train(category_tensor, line_tensor):

    hidden = rnn.initHidden()



    rnn.zero_grad()



    for i in range(line_tensor.size()[0]):

        output, hidden = rnn(line_tensor[i], hidden)



    loss = criterion(output, category_tensor)

    loss.backward()



    # Add parameters' gradients to their values, multiplied by learning rate

    for p in rnn.parameters():

        p.data.add_(p.grad.data, alpha=-learning_rate)



    return output, loss.item()



import time

import math



n_iters = 100000

print_every = 5000

plot_every = 1000



# Keep track of losses for plotting

current_loss = 0

all_losses = []



def timeSince(since):

    now = time.time()

    s = now - since

    m = math.floor(s / 60)

    s -= m * 60

    return '%dm %ds' % (m, s)



start = time.time()



for iter in range(1, n_iters + 1):

    category, line, category_tensor, line_tensor = randomTrainingExample()

    output, loss = train(category_tensor, line_tensor)

    current_loss += loss



    # Print iter number, loss, name and guess

    if iter % print_every == 0:

        guess, guess_i = categoryFromOutput(output)

        correct = '✓' if guess == category else '✗ (%s)' % category

        print('%d %d%% (%s) %.4f %s / %s %s' % (iter, iter / n_iters * 100, timeSince(start), loss, line, guess, correct))



    # Add current loss avg to list of losses

    if iter % plot_every == 0:

        all_losses.append(current_loss / plot_every)

        current_loss = 0
#Visualize Performance

import matplotlib.pyplot as plt

import matplotlib.ticker as ticker



plt.figure()

plt.plot(all_losses)
# Keep track of correct guesses in a confusion matrix

confusion = torch.zeros(n_categories, n_categories)

n_confusion = 10000



# Just return an output given a line

def evaluate(line_tensor):

    hidden = rnn.initHidden()



    for i in range(line_tensor.size()[0]):

        output, hidden = rnn(line_tensor[i], hidden)



    return output



# Go through a bunch of examples and record which are correctly guessed

for i in range(n_confusion):

    category, line, category_tensor, line_tensor = randomTrainingExample()

    output = evaluate(line_tensor)

    guess, guess_i = categoryFromOutput(output)

    category_i = all_cats.index(category)

    confusion[category_i][guess_i] += 1



# Normalize by dividing every row by its sum

for i in range(n_categories):

    confusion[i] = confusion[i] / confusion[i].sum()



# Set up plot

figsize = (10, 10)

fig = plt.figure(figsize=figsize)

ax = fig.add_subplot(111)

cax = ax.matshow(confusion.numpy())

fig.colorbar(cax)



# Set up axes

ax.set_xticklabels([''] + all_cats, rotation=90)

ax.set_yticklabels([''] + all_cats)



# Force label at every tick

ax.xaxis.set_major_locator(ticker.MultipleLocator(1))

ax.yaxis.set_major_locator(ticker.MultipleLocator(1))



# sphinx_gallery_thumbnail_number = 2

plt.show()
def predict(input_line, n_predictions=3):

    print('\n> %s' % input_line)

    with torch.no_grad():

        output = evaluate(lineToTensor(input_line))



        # Get top N categories

        topv, topi = output.topk(n_predictions, 1, True)

        predictions = []



        for i in range(n_predictions):

            value = topv[0][i].item()

            category_index = topi[0][i].item()

            print('(%.2f) %s' % (value, all_cats[category_index]))

            predictions.append([value, all_cats[category_index]])
predict('Aggelen')

predict('Accardo')

predict('Ferreiro')