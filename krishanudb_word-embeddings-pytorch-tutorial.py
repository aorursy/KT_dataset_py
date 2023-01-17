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
import torch

import torch.nn as nn

import torch.nn.functional as F

import torch.optim as optim
torch.manual_seed(1)
words_to_ix = {"hello": 0, "world": 1}

embeds = nn.Embedding(2, 5)

lookup_tensor = torch.tensor(words_to_ix["hello"], dtype=torch.long)

hello_embed = embeds(lookup_tensor)

print(hello_embed)
context_size = 4

embedding_dimension = 10



test_sentence = """When forty winters shall besiege thy brow,

And dig deep trenches in thy beauty's field,

Thy youth's proud livery so gazed on now,

Will be a totter'd weed of small worth held:

Then being asked, where all thy beauty lies,

Where all the treasure of thy lusty days;

To say, within thine own deep sunken eyes,

Were an all-eating shame, and thriftless praise.

How much more praise deserv'd thy beauty's use,

If thou couldst answer 'This fair child of mine

Shall sum my count, and make my old excuse,'

Proving his beauty by succession thine!

This were to be new made when thou art old,

And see thy blood warm when thou feel'st it cold.""".strip().split()
vocab = list(set(test_sentence))
print(vocab[:10])
trigrams = [(test_sentence[i], [test_sentence[i+1], test_sentence[i+2], test_sentence[i-1], test_sentence[i-2]]) for i in range(2, len(test_sentence) - 2)]
trigrams[:10]
words_to_ix = {word: i for i, word in enumerate(vocab)}

list(words_to_ix.items())[:10]
class NGramLanguageModeller(nn.Module):

    def __init__(self, vocab_size, embedding_dimension, context_size):

        super(NGramLanguageModeller, self).__init__()

        self.embeddings = nn.Embedding(vocab_size, embedding_dimension)

        self.linear1 = nn.Linear(context_size * embedding_dimension, 128)

        self.linear2 = nn.Linear(128, vocab_size)

        

    def forward(self, inputs):

        embeds = self.embeddings(inputs).view(1, -1)

        out = F.relu(self.linear1(embeds))

        out = self.linear2(out)

        log_probs = F.log_softmax(out, dim=1)

        return log_probs
losses = []

loss_function = nn.NLLLoss()

model = NGramLanguageModeller(len(vocab), embedding_dimension, context_size)

optimizer = optim.SGD(model.parameters(), lr=0.0005)
for epoch in range(5000):

    total_loss = 0

    for target, context in trigrams:

        context_ids = torch.tensor([words_to_ix[context_word] for context_word in context], dtype=torch.long)

        model.zero_grad()

        log_probs = model(context_ids)

        loss = loss_function(log_probs, torch.tensor([words_to_ix[target]], dtype=torch.long))

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    losses.append(total_loss)

    if epoch % 100 == 0:

        print("Epoch: {}; Loss: {}".format(epoch, total_loss))        
import matplotlib.pyplot as plt

plt.plot(losses)

plt.show()
model.embeddings(torch.tensor([words_to_ix["thy"]], dtype=torch.long))
test_sentence = """When forty winters shall besiege thy brow,

And dig deep trenches in thy beauty's field,

Thy youth's proud livery so gazed on now,

Will be a totter'd weed of small worth held:

Then being asked, where all thy beauty lies,

Where all the treasure of thy lusty days;

To say, within thine own deep sunken eyes,

Were an all-eating shame, and thriftless praise.

How much more praise deserv'd thy beauty's use,

If thou couldst answer 'This fair child of mine

Shall sum my count, and make my old excuse,'

Proving his beauty by succession thine!

This were to be new made when thou art old,

And see thy blood warm when thou feel'st it cold.""".strip().split()



vocab = list(set(test_sentence))

len_vocab = len(vocab)
import random

pairs = []

for i in range(20000):

    j = random.randint(2, len(test_sentence) - 3)

    pairs.append(((test_sentence[j], test_sentence[j + random.randint(-2, 2)]), 1.))

print(pairs[:10])
random_pairs = []

for i in range(100000):

    j = random.randint(0, len(test_sentence) - 1)

    k = random.randint(0, len(test_sentence) - 1)

    random_pairs.append(((test_sentence[j], test_sentence[k]), 0.))

print(random_pairs[:10])