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
#reading file

f = open('../input/dinos.txt','r')

data = f.read().lower()

chars = list(set(data))

data_size, vocab_size = len(data), len(chars)
#creating index dictornaries for tracing at endd



i_to_c = {i:ch for i,ch in enumerate(chars)}

c_to_i = {ch:i for i,ch in enumerate(chars)}
#gradient cliping to prevent exploding gradient



def clip(gradients,maxValue):

    for gradient in [gradients['dWaa'], gradients['dWax'], gradients['dWya'], gradients['db'], gradients['dby']]:

        gradient[gradient >maxValue] = maxValue

        gradient[gradient < -maxValue] = -maxValue

    

    return gradients
#sampling



def sample(parameters, char_to_ix):

    

    Waa, Wax, Wya, by, b = parameters['Waa'], parameters['Wax'], parameters['Wya'], parameters['by'], parameters['b']

    vocab_size = by.shape[0]

    n_a = Waa.shape[1]

    

    x = np.zeros((vocab_size,1))

    a_prev = np.zeros((n_a,1))

    indices = []

    idx = -1 

    

    counter = 0

    newline_character = char_to_ix['\n']

    

    while (idx != newline_character and counter != 50):

        a = np.tanh(np.dot(Wax, x) + np.dot(Waa, a_prev) + b)

        z = np.dot(Wya, a) + by

        y = np.exp(z)/sum(np.exp(z))

        idx = np.random.choice(list(range(vocab_size)),p = y.ravel())

        indices.append(idx)

        

        x = np.zeros((vocab_size,1))

        x[idx] = 1 

        a_prev = a

        

        counter +=1

        

    if (counter == 50):

        indices.append(char_to_ix['\n'])

    

    return indices
#simple recurrent net functions



def softmax(x):

    return np.exp(x)/sum(np.exp(x))

def rnn_step_forward(parameters, a_prev, x):

    

    Waa, Wax, Wya, by, b = parameters['Waa'], parameters['Wax'], parameters['Wya'], parameters['by'], parameters['b']

    a_next = np.tanh(np.dot(Wax, x) + np.dot(Waa, a_prev) + b) # hidden state

    p_t = softmax(np.dot(Wya, a_next) + by) 

    

    return a_next, p_t





def rnn_step_backward(dy, gradients, parameters, x, a, a_prev):

    

    gradients['dWya'] += np.dot(dy, a.T)

    gradients['dby'] += dy

    da = np.dot(parameters['Wya'].T, dy) + gradients['da_next'] 

    daraw = (1 - a * a) * da 

    gradients['db'] += daraw

    gradients['dWax'] += np.dot(daraw, x.T)

    gradients['dWaa'] += np.dot(daraw, a_prev.T)

    gradients['da_next'] = np.dot(parameters['Waa'].T, daraw)

    

    return gradients



def update_parameters(parameters, gradients, lr):



    parameters['Wax'] += -lr * gradients['dWax']

    parameters['Waa'] += -lr * gradients['dWaa']

    parameters['Wya'] += -lr * gradients['dWya']

    parameters['b']  += -lr * gradients['db']

    parameters['by']  += -lr * gradients['dby']

    return parameters



def rnn_forward(X, Y, a0, parameters, vocab_size = 27):

    x, a, y_hat = {}, {}, {}    

    a[-1] = np.copy(a0)

    

    loss = 0

    

    for t in range(len(X)):

        x[t] = np.zeros((vocab_size,1)) 

        if (X[t] != None):

            x[t][X[t]] = 1

        

        a[t], y_hat[t] = rnn_step_forward(parameters, a[t-1], x[t])

        

        loss -= np.log(y_hat[t][Y[t],0])

        

    cache = (y_hat, a, x)

        

    return loss, cache



def rnn_backward(X, Y, parameters, cache):

    gradients = {}

    

    (y_hat, a, x) = cache

    Waa, Wax, Wya, by, b = parameters['Waa'], parameters['Wax'], parameters['Wya'], parameters['by'], parameters['b']

    

    gradients['dWax'], gradients['dWaa'], gradients['dWya'] = np.zeros_like(Wax), np.zeros_like(Waa), np.zeros_like(Wya)

    gradients['db'], gradients['dby'] = np.zeros_like(b), np.zeros_like(by)

    gradients['da_next'] = np.zeros_like(a[0])

    

    for t in reversed(range(len(X))):

        dy = np.copy(y_hat[t])

        dy[Y[t]] -= 1

        gradients = rnn_step_backward(dy, gradients, parameters, x[t], a[t], a[t-1])

    

    return gradients, a



#one training step



def optimize(X, Y, a_prev, parameters, learning_rate = 0.01):

    loss, cache = rnn_forward(X, Y, a_prev, parameters)

    gradients, a = rnn_backward(X, Y, parameters, cache)

    

    #clipping

    gradients = clip(gradients,5)

    

    parameters = update_parameters(parameters, gradients, learning_rate)

    

    return loss, gradients, a[len(X)-1]
def smooth(loss, cur_loss):

    return loss * 0.999 + cur_loss * 0.001



def print_sample(sample_ix, ix_to_char):

    txt = ''.join(ix_to_char[ix] for ix in sample_ix)

    txt = txt[0].upper() + txt[1:]  

    print ('%s' % (txt, ), end='')



def get_initial_loss(vocab_size, seq_length):

    return -np.log(1.0/vocab_size)*seq_length



def initialize_parameters(n_a, n_x, n_y):

    np.random.seed(1)

    Wax = np.random.randn(n_a, n_x)*0.01 # input to hidden

    Waa = np.random.randn(n_a, n_a)*0.01 # hidden to hidden

    Wya = np.random.randn(n_y, n_a)*0.01 # hidden to output

    b = np.zeros((n_a, 1)) # hidden bias

    by = np.zeros((n_y, 1)) # output bias

    

    parameters = {"Wax": Wax, "Waa": Waa, "Wya": Wya, "b": b,"by": by}

    

    return parameters

#training model



def model(data, ix_to_char, char_to_ix, num_iterations = 35000, n_a = 50, dino_names = 7, vocab_size = 27):

    

    n_x, n_y = vocab_size, vocab_size

    

    parameters = initialize_parameters(n_a, n_x, n_y)

    

    loss = get_initial_loss(vocab_size, dino_names)

    with open("../input/dinos.txt") as f:

        examples = f.readlines()

    examples = [x.lower().strip() for x in examples]

    np.random.seed(0)

    np.random.shuffle(examples)

    

    a_prev = np.zeros((n_a, 1))

    

    for j in range(num_iterations):

        

        

        index = j % len(examples)

        X = [None] + [c_to_i[ch] for ch in examples[index]] 

        Y = X[1:] + [c_to_i["\n"]]

        

        curr_loss, gradients, a_prev = optimize(X, Y, a_prev, parameters)

        

       

        

        loss = smooth(loss, curr_loss)



        if j % 2000 == 0:

            

            print('Iteration: %d, Loss: %f' % (j, loss) + '\n')

            

            for name in range(dino_names):

                sampled_indices = sample(parameters, c_to_i)

                print_sample(sampled_indices, i_to_c)

            print('\n')

        

    return parameters
parameters = model(data, i_to_c, c_to_i)