# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
!pip install -U scipy==1.5.0
# Simple training document

dataset = pd.DataFrame(

    [

        ['this is a book'],

        ['this is a cat'],

        ['how are you'],

        ['cat like milk'],

        ['a book about cat'],

    ], 

    columns=['doc'])



# Tokenization

tokens = [x.split(' ') for x in dataset['doc']]

tokens = [sentence for sentence in tokens if len(sentence) >= 2]  # Only keep sentence with at least 2 words

tokens
# Construct the vocabulary and token ids

list_of_tokens = []

dict_of_tokens = {}

for row in tokens:

    for t in row:

        if t not in dict_of_tokens:

            dict_of_tokens[t] = len(list_of_tokens)

            list_of_tokens.append(t)

dict_of_tokens
# Encode the document

encoded_tokens = [[dict_of_tokens[x] for x in row] for row in tokens]

encoded_tokens
n_tokens = len(list_of_tokens)

n_tokens
n_dim = 6  # Enter the dimension of the word embedding
window_size = 5  # need to be an odd number

assert window_size % 2 == 1, 'Window size needs to be an odd number'

assert window_size >= 3, 'Window size needs to be greater or equal to 3'

max_j = (window_size - 1) // 2
# Word Embedding Initialization

W_i = np.random.rand(n_tokens, n_dim)  # Input words weight matrix, this will be the word embedding after training

W_o = np.random.rand(n_tokens, n_dim)  # Ouput words weight metrix
# Continuous Bag of Words (CBOW)

# Context words as inputs, output center word prediction



from scipy.special import log_softmax, softmax

def cbow_predict_prob(input_tokens, W_i, W_o):

    """ Predict the probability of each words being the center word given a list of intput context words in CBOW """

    return softmax(np.dot(W_o, W_i[input_tokens].mean(axis=0)))

def cbow_predict(input_tokens, W_i, W_o):

    """ Predict the most likely center word given a list of input context words in CBOW """

    return np.argmax(np.dot(W_o, W_i[input_tokens].mean(axis=0)))

def cbow_prob(input_tokens, output_token, W_i, W_o):

    """ Probability of getting a output word given the input words in CBOW """

    return softmax(np.dot(W_o, W_i[input_tokens].mean(axis=0)))[output_token]

def cbow_loss(input_tokens, output_token, W_i, W_o):

    """ Loss (-log(probability)) between the input words and a true output word in CBOW """

    return -log_softmax(np.dot(W_o, W_i[input_tokens].mean(axis=0)))[output_token]
x_tokens = np.array([0, 1, 4])

print('CBOW predict most likely center word to be:', cbow_predict(x_tokens, W_i, W_o), list_of_tokens[cbow_predict(x_tokens, W_i, W_o)])

print('CBOW probability of each words as center word:', cbow_predict_prob(x_tokens, W_i, W_o))
x_tokens = np.array([0, 1, 4]); y_token = 3

print('CBOW probability and loss of center word being 3 given 0, 1, 4 as context words:',

      cbow_prob(x_tokens, y_token, W_i, W_o), cbow_loss(x_tokens, y_token, W_i, W_o))
def cbow_sentence_loss(sentence_tokens, W_i, W_o, max_j):

    if not isinstance(sentence_tokens, np.ndarray):

        sentence_tokens = np.array(sentence_tokens)    

    return np.sum([

        cbow_loss(

            sentence_tokens[[j for j in range(

                max(i - max_j, 0), min(i + max_j + 1, len(sentence_tokens))

            ) if j != i]], 

            sentence_tokens[i], W_i, W_o,

        ) 

        for i in range(len(sentence_tokens))

    ])
def cbow_doc_loss(doc_tokens, W_i, W_o, max_j):

    return np.sum([cbow_sentence_loss(sentence_tokens, W_i, W_o, max_j) for sentence_tokens in doc_tokens])        
cbow_sentence_loss(encoded_tokens[3], W_i, W_o, max_j)
cbow_doc_loss(encoded_tokens, W_i, W_o, max_j)
from itertools import chain



def doc_split_context_and_center(doc_tokens, max_j):

    """ Flatten the doc with multiple sentences into list of context words and center words 

    """

    # if isinstance(doc_tokens[0], list):

    flatten_context_words = [

        sentence_tokens[max(i - max_j, 0): i] + sentence_tokens[i + 1: i + max_j + 1]

        for sentence_tokens in doc_tokens

        for i in range(len(sentence_tokens))        

    ]

    # elif isinstance(doc_tokens[0], np.ndarray):

    flatten_center_words = list(chain.from_iterable(doc_tokens))

    return flatten_context_words, flatten_center_words
# Prepare our dataset into a better format

flatten_context_words, flatten_center_words = doc_split_context_and_center(encoded_tokens, max_j)
def cbow_flatten_data_loss(flatten_context_words, flatten_center_words, W_i, W_o):

    return np.sum([

        cbow_loss(context_tokens, center_token, W_i, W_o) 

        for context_tokens, center_token in zip(flatten_context_words, flatten_center_words)

    ])
cbow_flatten_data_loss(flatten_context_words, flatten_center_words, W_i, W_o)
# gradient

def cbow_gradient(flatten_context_words, flatten_center_words, W_i, W_o):

    context_vectors = np.array([W_i[context_words].mean(axis=0) for context_words in flatten_context_words])

    scalar_products = np.matmul(context_vectors, W_o.T)

    softmax_factors = softmax(scalar_products, axis=1)

    len_context_words = np.array([len(x) for x in flatten_context_words])

    A_tj = (W_o[flatten_center_words] - np.matmul(softmax_factors, W_o)) / len_context_words.reshape(-1, 1)

    dLdW_i = np.zeros_like(W_i)

    for context_words, a_t in zip(flatten_context_words, A_tj):

        dLdW_i[context_words] += a_t

    dLdW_i = -dLdW_i / len(flatten_context_words)

    B_ti = -softmax_factors

    for t, center_word in enumerate(flatten_center_words):

        B_ti[t, center_word] += 1

    dLdW_o = -np.matmul(B_ti.T, context_vectors) / len(flatten_context_words)

    return dLdW_i, dLdW_o
dLdW_i, dLdW_o = cbow_gradient(flatten_context_words, flatten_center_words, W_i, W_o)
plt.subplot(1,2,1); plt.imshow(dLdW_i)

plt.subplot(1,2,2); plt.imshow(dLdW_o)

plt.show()
def train(flatten_context_words, flatten_center_words, W_i, W_o,

          gradient=cbow_gradient, loss_function=cbow_flatten_data_loss,

          num_iterations=100, learning_rate=0.1, hist=[], verbose=1):

    loss = loss_function(flatten_context_words, flatten_center_words, W_i, W_o)

    print('Inital loss:', loss)

    hist.append(loss)

    for i in range(num_iterations):

        dLdW_i, dLdW_o = gradient(flatten_context_words, flatten_center_words, W_i, W_o)

        W_i -= learning_rate * dLdW_i

        W_o -= learning_rate * dLdW_o

        loss = loss_function(flatten_context_words, flatten_center_words, W_i, W_o)

        if verbose and i%verbose == 0:

            print(f'[{i+1} / {num_iterations}]', loss)

        hist.append(loss)

    print('Final loss:', loss)

    return W_i, W_o
%%time

hist = []

W_i, W_o = train(flatten_context_words, flatten_center_words, W_i, W_o,

                 gradient=cbow_gradient, loss_function=cbow_flatten_data_loss,

                 num_iterations=1000, learning_rate=0.2, hist=hist, verbose=20)
plt.plot(hist)

plt.show()
W_i  # This is our word embedding for CBOW
x_tokens = np.array([0, 1, 4])

print('CBOW predict most likely center word to be:', cbow_predict(x_tokens, W_i, W_o), list_of_tokens[cbow_predict(x_tokens, W_i, W_o)])

print('CBOW probability of each words as center word:', cbow_predict_prob(x_tokens, W_i, W_o))
x_tokens = np.array([1, 2, 3])

print('CBOW predict most likely center word to be:', cbow_predict(x_tokens, W_i, W_o), list_of_tokens[cbow_predict(x_tokens, W_i, W_o)])

print('CBOW probability of each words as center word:', cbow_predict_prob(x_tokens, W_i, W_o))
# Another example

# Word Embedding Initialization

W_i = np.random.rand(n_tokens, n_dim)  # Input words weight matrix, this will be the word embedding after training

W_o = np.random.rand(n_tokens, n_dim)  # Ouput words weight metrix
# before training

x_tokens = np.array([0, 1, 4])

y_pred = cbow_predict(x_tokens, W_i, W_o)

y_prod_pred = cbow_predict_prob(x_tokens, W_i, W_o)

print('CBOW predict most likely center word to be:', y_pred, list_of_tokens[y_pred], 'with probability =', y_prod_pred[y_pred])

print('CBOW probability of each words as center word:', y_prod_pred)
%%time

hist = []

W_i, W_o = train(flatten_context_words, flatten_center_words, W_i, W_o,

                 gradient=cbow_gradient, loss_function=cbow_flatten_data_loss,

                 num_iterations=1000, learning_rate=0.2, hist=hist, verbose=50)

plt.plot(hist); plt.show()
# after training

x_tokens = np.array([0, 1, 4])

y_pred = cbow_predict(x_tokens, W_i, W_o)

y_prod_pred = cbow_predict_prob(x_tokens, W_i, W_o)

print('CBOW predict most likely center word to be:', y_pred, list_of_tokens[y_pred], 'with probability =', y_prod_pred[y_pred])

print('CBOW probability of each words as center word:', y_prod_pred)
W_i  # This is our word embedding for CBOW