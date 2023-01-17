# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#I will try this from my coursera assignment of the course "Sequence Model"-Week 2 Assignment- "Emojify", Let's see how far can I go :)

import matplotlib.pyplot as plt

import tensorflow as tf

import re

import nltk



%matplotlib inline



train_df=pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')

test_df=pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')
#reading from the file to turn the words to word embedding vector

def read_glove_vecs(glove_file):

    with open(glove_file, 'r') as f:

        words = set()

        word_to_vec_map = {}

        for line in f:

            line = line.strip().split()

            curr_word = line[0]

            words.add(curr_word)

            word_to_vec_map[curr_word] = np.array(line[1:], dtype=np.float64)

        

        i = 1

        words_to_index = {}

        index_to_words = {}

        for w in sorted(words):

            words_to_index[w] = i

            index_to_words[i] = w

            i = i + 1

    return words_to_index, index_to_words, word_to_vec_map
#reading from the file to learn the word embedding into the list word_to_vec_map

word_to_index, index_to_word, word_to_vec_map = read_glove_vecs('/kaggle/input/wordembed/glove.6B.50d.txt')
print(train_df["text"])
def clean(text):

    regex = re.compile('([^\s\w]|_)+')

    sentence = regex.sub('', text).lower()

    sentence = sentence.split(" ")

    

    for word in list(sentence):

        if word not in word_to_vec_map:

            sentence.remove(word)  

            

    sentence = " ".join(sentence)

    return sentence
for i in range (train_df.shape[0]):

    train_df.at[i,'text']=clean(train_df.loc[i,'text'])

    

for i in range(test_df.shape[0]):

    test_df.at[i,'text']=clean(test_df.loc[i,'text'])
#determining the max length of a text in training set

maxLen = len(max(train_df["text"], key=len).split())
#trying the length of the text of id=1

length=len(str(train_df[train_df['id']==1]["text"]).split())
train_df.head()
#One hot encoding of the target to 2 dimensional vector

Y_oh_train = tf.one_hot(train_df["target"],2,dtype='int32')



Y_oh_train[0]

train_df["text"].values[1]
import string

def sentence_to_avg(sentence, word_to_vec_map):

    """

    Converts a sentence (string) into a list of words (strings). Extracts the GloVe representation of each word

    and averages its value into a single vector encoding the meaning of the sentence.

    

    Arguments:

    sentence -- string, one training example from X

    word_to_vec_map -- dictionary mapping every word in a vocabulary into its 50-dimensional vector representation

    

    Returns:

    avg -- average vector encoding information about the sentence, numpy-array of shape (50,)

    """

    

    # Step 1: Split sentence into list of lower case words (≈ 1 line)

    #sentence=str(sentence.translate(str.maketrans('', '', string.punctuation)))

    words = (sentence.lower()).split()

    # Initialize the average word vector, should have the same shape as your word vectors.

    avg = np.zeros((50,))

    

    # Step 2: average the word vectors. You can loop over the words in the list "words".

    #using try except disregard some words which doesn't exist in the glove Vector such as 'dtype'

    total = 0

    for w in words:

        total += word_to_vec_map[w]

    if len(words):

        avg = total/len(words)

    

    return avg
def softmax(x):

    """Compute softmax values for each sets of scores in x."""

    e_x = np.exp(x - np.max(x))

    return e_x / e_x.sum()
def predict(X, Y, W, b, word_to_vec_map):

    """

    Given X (sentences) and Y (emoji indices), predict emojis and compute the accuracy of your model over the given set.

    

    Arguments:

    X -- input data containing sentences, numpy array of shape (m, None)

    Y -- labels, containing index of the label emoji, numpy array of shape (m, 1)

    

    Returns:

    pred -- numpy array of shape (m, 1) with your predictions

    """

    m = Y.shape[0]

    pred = np.zeros((m, 1))

    Y_oh=tf.one_hot(Y,n_y,dtype='int32')

    Y=np.zeros((m,1))

    

    

    for j in range(m):                       # Loop over training examples

        

        # Split jth test example (sentence) into list of lower case words

        if Y_oh[j][0]==1:

            Y[j]=0

        else:

            Y[j]=1

        words = X[j].lower().split()

        

        avg = np.zeros((50,))

    

        total = 0

        for w in words:

            total += word_to_vec_map[w]

        if len(words):

            avg = total/len(words)

        



        # Forward propagation

        Z = np.dot(W, avg) + b

        A = softmax(Z)

        pred[j] = np.argmax(A)

        

    print("Accuracy: "  + str(np.mean((pred[:] == np.reshape(Y,(Y.shape[0],1)[:])))))

    return pred
def model(X, Y, word_to_vec_map, learning_rate = 0.01, num_iterations = 400):

    """

    Model to train word vector representations in numpy.

    

    Arguments:

    X -- input data, numpy array of sentences as strings, of shape (m, 1)

    Y -- labels, numpy array of integers between 0 and 7, numpy-array of shape (m, 1)

    word_to_vec_map -- dictionary mapping every word in a vocabulary into its 50-dimensional vector representation

    learning_rate -- learning_rate for the stochastic gradient descent algorithm

    num_iterations -- number of iterations

    

    Returns:

    pred -- vector of predictions, numpy-array of shape (m, 1)

    W -- weight matrix of the softmax layer, of shape (n_y, n_h)

    b -- bias of the softmax layer, of shape (n_y,)

    """

    

    np.random.seed(1)



    # Define number of training examples

    m = Y.shape[0]                          # number of training examples

    n_y = 2                                # number of classes  

    n_h = 50                                # dimensions of the GloVe vectors 

    

    # Initialize parameters using Xavier initialization

    W = np.random.randn(n_y, n_h) / np.sqrt(n_h)

    b = np.zeros((n_y,1))

    pred=np.zeros((m,1))

    

    # Convert Y to Y_onehot with n_y classes

    Y_oh=tf.one_hot(Y,n_y,dtype='int32')

    

    # Optimization loop

    for t in range(num_iterations):# Loop over the number of iterations

        print("Number of iterations",t)

        for i in range(m):          # Loop over the training examples

            

            ### START CODE HERE ### (≈ 4 lines of code)

            # Average the word vectors of the words from the i'th training example

            avg = sentence_to_avg(X[i], word_to_vec_map)

            avg=np.reshape(avg,(n_h,1))



            # Forward propagate the avg through the softmax layer

            z = np.dot(W,avg)+b

            a = softmax(z)



            # Compute cost using the i'th training label's one hot representation and "A" (the output of the softmax)

            cost =-np.dot(np.transpose(Y_oh[i]),np.log(a))

            ### END CODE HERE ###

            

            # Compute gradients 

            Y_oh_try=np.reshape(Y_oh[i],(n_y,1))

            dz = a - Y_oh_try

            dz=np.reshape(dz,(n_y,1))

            avg=np.reshape(avg,(1, n_h))

            dW = np.dot(dz,avg)

            db = dz



            # Update parameters with Stochastic Gradient Descent

            W = W - learning_rate * dW

            b = b - learning_rate * db

        

        if t % 100 == 0:

            print("Epoch: " + str(t) + " --- cost = " + str(cost))

            print(Y.shape)

            pred = predict(X, Y, W, b, word_to_vec_map) #predict is defined in emo_utils.py

        

    return pred, W, b
print(train_df["text"].shape)

print(Y_oh_train[0].shape)

X=train_df["text"]

n_y=2

n_h=50

W = np.random.randn(n_y, n_h) / np.sqrt(n_h)

b = np.zeros((n_y,1))

avg = sentence_to_avg(X[0], word_to_vec_map)

avg=np.reshape(avg,(n_h,1))

# Forward propagate the avg through the softmax layer

z = np.dot(W,avg)+b

a = softmax(z)

print("shape of b",b.shape)

print("shape of W",W.shape)

print("shape of avg",avg.shape)

print("z shape",z.shape)

print()

cost =-np.dot(np.transpose(Y_oh_train[0]),np.log(a))

dz = a - Y_oh_train[0]

print("a shape",a.shape)

print("y_oh shape",train_df["target"].shape)

print("X_shape",train_df["text"].shape)

print("shape of dz",dz.shape)
pred, W, b = model(train_df["text"], train_df["target"], word_to_vec_map)

print(pred)
def pre(X, W, b, word_to_vec_map):

    

    print(type(X))

    m=X.shape[0]

    

    pred=np.zeros((m,1))

    

    

    for j in range(m):                       # Loop over training examples

        

        # Split jth test example (sentence) into list of lower case words

        words = X[j].lower().split()

        

        avg = np.zeros((50,))

    

        total = 0

        for w in words:

            total += word_to_vec_map[w]

        if len(words):

            avg = total/len(words)

        



        # Forward propagation

        Z = np.dot(W, avg) + b

        A = softmax(Z)

        pred[j] = np.argmax(A)

    return pred
test_df['text']
# %% [code]

sample_sub=pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv")

sample_sub.head()

# %% [code]

sample_sub["target"]= pre(test_df["text"], W, b, word_to_vec_map)

sample_sub["target"]=sample_sub["target"].astype(int)

sample_sub.head()
sample_sub.to_csv("submission.csv",index=False)