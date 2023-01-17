# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from keras.models import Model

from keras.layers import Dense, Input, Dropout, LSTM, Activation

from keras.layers.embeddings import Embedding

from keras.preprocessing import sequence

from keras.initializers import glorot_uniform



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

import tensorflow as tf

import re

import nltk



%matplotlib inline



train_df=pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')

test_df=pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')
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
def clean(text):

    regex = re.compile('([^\s\w]|_)+')

    sentence = regex.sub('', text).lower()

    '''sentence = sentence.split(" ")

    

    for word in list(sentence):

        if word not in word_to_vec_map:

            sentence.remove(word)  

            

    sentence = " ".join(sentence)'''

    return sentence
for i in range (train_df.shape[0]):

    train_df.at[i,'text']=clean(train_df.loc[i,'text'])

    

for i in range(test_df.shape[0]):

    test_df.at[i,'text']=clean(test_df.loc[i,'text'])
def convert_to_one_hot(Y, C):

    Y = np.eye(C)[np.reshape(Y,-1)]

    return Y
Y_oh_train = convert_to_one_hot(train_df["target"], C = 2)
def sentences_to_indices(X, word_to_index, max_len):

    """

    Converts an array of sentences (strings) into an array of indices corresponding to words in the sentences.

    The output shape should be such that it can be given to `Embedding()` (described in Figure 4). 

    

    Arguments:

    X -- array of sentences (strings), of shape (m, 1)

    word_to_index -- a dictionary containing the each word mapped to its index

    max_len -- maximum number of words in a sentence. You can assume every sentence in X is no longer than this. 

    

    Returns:

    X_indices -- array of indices corresponding to words in the sentences from X, of shape (m, max_len)

    """

    

    m = X.shape[0]                                   # number of training examples

    vocab_len = len(word_to_index) + 1    

    ### START CODE HERE ###

    # Initialize X_indices as a numpy matrix of zeros and the correct shape (≈ 1 line)

    X_indices = np.zeros((m,max_len))

    

    for i in range(m):                               # loop over training examples

        

        # Convert the ith training sentence in lower case and split is into words. You should get a list of words.

        sentence_words =X[i].split()

        #print(i,sentence_words)

        

        # Initialize j to 0

        j = 0

        

        # Loop over the words of sentence_words

        for w in sentence_words:

            

            # Set the (i,j)th entry of X_indices to the index of the correct word.

            if w not in word_to_index:

                X_indices[i,j]=vocab_len-1

            else:

                X_indices[i, j] = word_to_index[w]

            # Increment j to j + 1

            j = j+1

            

    ### END CODE HERE ###

    

    return X_indices
train_df["text"].shape[0]
def pretrained_embedding_layer(word_to_vec_map, word_to_index):

    """

    Creates a Keras Embedding() layer and loads in pre-trained GloVe 50-dimensional vectors.

    

    Arguments:

    word_to_vec_map -- dictionary mapping words to their GloVe vector representation.

    word_to_index -- dictionary mapping from words to their indices in the vocabulary (400,001 words)



    Returns:

    embedding_layer -- pretrained layer Keras instance

    """

    

    vocab_len = len(word_to_index) + 1                  # adding 1 to fit Keras embedding (requirement)

    emb_dim = word_to_vec_map["cucumber"].shape[0]      # define dimensionality of your GloVe word vectors (= 50)

    

    ### START CODE HERE ###

    # Step 1

    # Initialize the embedding matrix as a numpy array of zeros.

    # See instructions above to choose the correct shape.

    emb_matrix = np.zeros((vocab_len,emb_dim))

    

    # Step 2

    # Set each row "idx" of the embedding matrix to be 

    # the word vector representation of the idx'th word of the vocabulary

    for word, idx in word_to_index.items():

        emb_matrix[idx, :] = word_to_vec_map[word]



    # Step 3

    # Define Keras embedding layer with the correct input and output sizes

    # Make it non-trainable.

    embedding_layer = Embedding(vocab_len,emb_dim,trainable=False)

    ### END CODE HERE ###



    # Step 4 (already done for you; please do not modify)

    # Build the embedding layer, it is required before setting the weights of the embedding layer. 

    embedding_layer.build((None,)) # Do not modify the "None".  This line of code is complete as-is.

    

    # Set the weights of the embedding layer to the embedding matrix. Your layer is now pretrained.

    embedding_layer.set_weights([emb_matrix])

    

    return embedding_layer
embedding_layer = pretrained_embedding_layer(word_to_vec_map, word_to_index)

print("weights[0][1][3] =", embedding_layer.get_weights()[0][1][3])
def Emojify_V2(input_shape, word_to_vec_map, word_to_index):

    """

    Function creating the Emojify-v2 model's graph.

    

    Arguments:

    input_shape -- shape of the input, usually (max_len,)

    word_to_vec_map -- dictionary mapping every word in a vocabulary into its 50-dimensional vector representation

    word_to_index -- dictionary mapping from words to their indices in the vocabulary (400,001 words)



    Returns:

    model -- a model instance in Keras

    """

    

    ### START CODE HERE ###

    # Define sentence_indices as the input of the graph.

    # It should be of shape input_shape and dtype 'int32' (as it contains indices, which are integers).

    sentence_indices = Input(shape=input_shape,dtype='int32')

    

    # Create the embedding layer pretrained with GloVe Vectors (≈1 line)

    embedding_layer = pretrained_embedding_layer(word_to_vec_map, word_to_index)

    

    # Propagate sentence_indices through your embedding layer

    # (See additional hints in the instructions).

    embeddings =  embedding_layer(sentence_indices)   

    

    # Propagate the embeddings through an LSTM layer with 128-dimensional hidden state

    # The returned output should be a batch of sequences.

    X = LSTM(units=128,return_sequences=True)(embeddings)

    # Add dropout with a probability of 0.5

    X = Dropout(rate = 0.5 )(X)

    # Propagate X trough another LSTM layer with 128-dimensional hidden state

    # The returned output should be a single hidden state, not a batch of sequences.

    X = LSTM(units=128,return_sequences=False)(X)

    # Add dropout with a probability of 0.5

    X = Dropout(rate = 0.5 )(X)

    # Propagate X through a Dense layer with 5 units

    X = Dense(units=2)(X)

    # Add a softmax activation

    X = Activation('softmax')(X)

    

    # Create Model instance which converts sentence_indices into X.

    model  = Model(inputs=sentence_indices, outputs=X)

    

    ### END CODE HERE ###

    

    return model
maxLen=0

for i in range(train_df["text"].shape[0]):

    length=len(str(train_df["text"].values[i]).split())

    if length>maxLen:

        maxLen=length

        

#maxLen = len(max(train_df["text"].values, key=len).split())

print(maxLen)
model = Emojify_V2((maxLen,), word_to_vec_map, word_to_index)

model.summary()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
X_train_indices = sentences_to_indices(train_df["text"], word_to_index, maxLen)
model.fit(X_train_indices, Y_oh_train, epochs = 50, batch_size = 32, shuffle=True)
X_test_indices = sentences_to_indices(test_df["text"], word_to_index, max_len = maxLen)

# %% [code]

sample_sub=pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv")

pred = model.predict(X_test_indices)



for i in range(test_df["text"].shape[0]):

    sample_sub["target"].values[i]=np.argmax(pred[i])



sample_sub.head()
sample_sub.to_csv("submission.csv",index=False)