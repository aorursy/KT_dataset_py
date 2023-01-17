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
# Generate text which is similar to the writing style of William Shakespeare



#  link to understand problem statement https://blog.owulveryck.info/2017/10/29/about-recurrent-neural-network-shakespeare-and-go.html



# The dataset used in this experiment has partial content of different plays of Shakespeare concatenated into a single plain text file.

# Shakespeare is a famous English poet , play writer and actor. He is regarded as the greatest writer in the English language and the world's greatest dramatist. He is often called a England's national poet and the Bard of Avon.



# We have chosen plays of Shakespeare as our dataset mainly for two reasons :



# His work is widely recognized as standard for poetry and language.

# The result of combining of his work provides a sizeable corpus for our model to learn.



## Importing required packages



import unidecode

import string

import random

import re

import torch

import torch.nn as nn

from torch.autograd import Variable



all_characters = string.printable

## code to find length of all_characters and storing the value in n_characters

n_characters = len(all_characters)

## code to convert unicode characters into plain ASCII.

file = unidecode.unidecode(open('/kaggle/input/shakespeare.txt').read())

## code to find length of the file

file_len = len(file)

## printing the length of the file

print('file_len =', file_len)
# The variable 'file' is a string with 1115393 characters. This is the raw content of the Shakespeare text file (dataset file), including many details like white spaces, line breaks etc.



# Now to get the sense of the data we print first 1000 characters in the string:



file[:1000]
# As the string is large, we will split it into chunks to provide inputs to the RNN using the function random_chunk()



## Initializing the length of chunk

chunk_len = 200

## Function to split the string into chunks

def random_chunk():

    ## Initializing the starting index value of the big string 

    start_index = random.randint(0, file_len - chunk_len)

    ## Initializing the ending index of the string 

    end_index = start_index + chunk_len + 1

    ## returning the chunk

    return file[start_index:end_index]



print((random_chunk()))
# Building the Model

# This model will take input as the character for step  t−1 , and is expected to give the output  t , which is the next character. There are three layers:



# Linear layer that encodes the input character into an internal state

# GRU layer (which may itself have multiple layers) that operates on that internal and hidden state

# Decoder layer that outputs the probability distribution



### Creating recurrent neural network



### Creating recurrent neural network

class RNN(nn.Module):

    

    def __init__(self, input_size, hidden_size, output_size, n_layers=1):

        super(RNN, self).__init__()

        self.input_size = input_size

        self.hidden_size = hidden_size

        self.output_size = output_size

        self.n_layers = n_layers

        self.encoder = nn.Embedding(input_size, hidden_size)

        self.gru = nn.GRU(hidden_size, hidden_size, n_layers)

        self.decoder = nn.Linear(hidden_size, output_size)

    

    def forward(self, input, hidden):

        input = self.encoder(input.view(1, -1))

        output, hidden = self.gru(input.view(1, 1, -1), hidden)

        output = self.decoder(output.view(1, -1))

        return output, hidden



    def init_hidden(self):

        return Variable(torch.zeros(self.n_layers, 1, self.hidden_size))

    

    
# Defining the Helper Functions

# Let us define some helper functions to:

# A helper function is a function that performs part of the computation of another function. Helper functions are used to make your programs easier to read by giving descriptive names to computations. They also let you reuse computations, just as with functions in general.

# A "helper function" is a function you write because you need that particular functionality in multiple places, and because it makes the code more readable. A good example is an average function. You'd write a function named avg or similar, that takes in a list of numbers, and returns the average value from that list.

# You can then use that function in your main function, or in other, more complex helper functions, wherever you need it. Basically any block of code that you use multiple times would be a good candidate to be made into a helper function.

# Another reason for helper functions is to make the code easier to read. For instance, I might be able to write a really clever line of code to take the average of a list of numbers, and it only takes a single line, but it's complicated and hard to read. I could make a helper function and replace my complicated line with one that's much easier to read.







# Convert the input string chunks into the character tensors

# Each chunk will be turned into a tensor, specifically a LongTensor (used for integer values), by looping through the characters of the string and looking up the index of each character in all_characters.







# Turn string into list of longs

def char_tensor(string):

    ## tensor is a array

    tensor = torch.zeros(len(string)).long()

    for c in range(len(string)):

        tensor[c] = all_characters.index(string[c])

    return Variable(tensor)



print(char_tensor('abcDEF'))

    
# Finally we assemble a pair of input and target tensors for training, from a random chunk. The input will be all characters up to the end, and the target will be all characters from the first. So if our chunk is "abc" the input will correspond to "ab" while the target is "bc".



def random_training_set():    

    chunk = random_chunk()

    inp = char_tensor(chunk[:-1])

    target = char_tensor(chunk[1:])

    return inp, target

# To evaluate the network we will feed one character at a time, use the outputs of the network as a probability distribution for the next character, and repeat. To start generation we pass a priming string to start building up the hidden state, from which we then generate one character at a time.



def evaluate(prime_str='A', predict_len=100, temperature=0.8):

    hidden = decoder.init_hidden()

    prime_input = char_tensor(prime_str)

    predicted = prime_str



    # Use priming string to "build up" hidden state

    for p in range(len(prime_str) - 1):

        _, hidden = decoder(prime_input[p], hidden)

    inp = prime_input[-1]

    

    for p in range(predict_len):

        output, hidden = decoder(inp, hidden)

        

        # Sample from the network as a multinomial distribution

        output_dist = output.data.view(-1).div(temperature).exp()

        top_i = torch.multinomial(output_dist, 1)[0]

        

        # Add predicted character to string and use as next input

        predicted_char = all_characters[top_i]

        predicted += predicted_char

        inp = char_tensor(predicted_char)



    return predicted





# To keep track of how long training takes, we have added a time_since(timestamp) function which returns a human readable string:



## Importing required packages

import time, math

## function to print amount of time passed

def time_since(since):

    s = time.time() - since

    m = math.floor(s / 60)

    s -= m * 60

    return '%dm %ds' % (m, s)
# The main training function



def train(inp, target):

    hidden = decoder.init_hidden()

    decoder.zero_grad()

    loss = 0



    for c in range(chunk_len):

        output, hidden = decoder(inp[c], hidden)

        '''unsqueeze() is used to add dimension to the tensor'''

        loss += criterion(output, target[c].unsqueeze(dim=0))

    # Back propagation

    loss.backward()

    decoder_optimizer.step()



    return loss.item() / chunk_len
#  Then we define the training parameters, instantiate the model, and start training. In the below cell we are trying to print the chunk, loss and time taken for every 50th iteration and for every 20th iteration we are trying to plot the loss vs epochs(iterations).



n_epochs = 1300 #Number of epochs

print_every = 60

plot_every = 25

hidden_size = 139

n_layers = 6

lr = 0.0005



decoder = RNN(n_characters, hidden_size, n_characters, n_layers)

## Optimizer

decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=lr)

## Loss function

criterion = nn.CrossEntropyLoss()



start = time.time()

all_losses = []

loss_avg = 0



for epoch in range(1, n_epochs + 1):

    loss = train(*random_training_set())       

    loss_avg += loss



    if epoch % print_every == 0:

        print('[%s (%d %d%%) %.4f]' % (time_since(start), epoch, epoch / n_epochs * 100, loss))

        print(evaluate('Wh', 100), '\n')



    if epoch % plot_every == 0:

        all_losses.append(loss_avg / plot_every)

        loss_avg = 0
#  Plotting the historical loss from all_losses shows the network learning:



import matplotlib.pyplot as plt

import matplotlib.ticker as ticker

%matplotlib inline



plt.figure()

plt.plot(all_losses)

plt.xlabel("epoch")

plt.ylabel("loss")
# Adapting or Tuning for Text Generation



# In the evaluate function above, every time a prediction is made, the outputs are divided by the "temperature" argument passed. Using a higher number makes all actions more equally likely, and thus gives us "more random" outputs. Using a lower value (less than 1) makes high probabilities contribute more. As we turn the temperature towards zero we are choosing only the most likely outputs.



# We can see the effects of this by adjusting the temperature argument:
print(evaluate('u', 200, temperature=0.8))
#  Lower temperatures are less varied, choosing only the more probable outputs:



print(evaluate('Th', 200, temperature=0.2))
# Higher temperatures more varied, choosing less probable outputs:



print(evaluate('how', 200, temperature=1.4))
# Higher temperatures more varied, choosing less probable outputs:



print(evaluate('how', 40, temperature=1.9))
# Higher temperatures more varied, choosing less probable outputs:



print(evaluate('how', 49, temperature=10))