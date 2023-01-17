# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import emoji

import matplotlib.pyplot as plt



from keras.models import Model

from keras.layers import Dense, Input, Dropout, LSTM, Activation

from keras.layers.embeddings import Embedding

from keras.callbacks import ReduceLROnPlateau



%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/emojify/train_emoji.csv', header=None, usecols=[0,1])

test = pd.read_csv('../input/emojify/test_emoji.csv', header=None, usecols=[0,1])

train.head()
X_train, Y_train = train[0], train[1]

X_test, Y_test = test[0], test[1]

print(f'Shape of X is: {X_train.shape}')

print(f'Shape of Y is: {Y_train.shape}')
emoji_dictionary = {"0": "\u2764\uFE0F",    # :heart: prints a black instead of red heart depending on the font

                    "1": ":baseball:",

                    "2": ":smile:",

                    "3": ":disappointed:",

                    "4": ":fork_and_knife:"}



def label_to_emoji(label):

    return emoji.emojize(emoji_dictionary[str(label)], use_aliases=True)



print(X_train[20], label_to_emoji(Y_train[20]))



maxWords = len(max(X_train, key=len).split())

print('Maximum words in sentence are:',maxWords)
# Convert Y to one-hot vectors

Y_train_oh = pd.get_dummies(Y_train)

print(Y_train_oh.shape)
def read_glove_vecs(glove_file):

    with open(glove_file, 'r') as f:

        words = set()         # ensures unique values

        word_to_vec_map = {}  # this will be a dictionary mapping words to their vectors

        for line in f:

            line = line.strip().split()

            curr_word = line[0]

            words.add(curr_word)

            word_to_vec_map[curr_word] = np.array(line[1:], dtype=np.float64)

        

        i = 1

        words_to_index = {}   # dictionary mapping words to their index in the dictionary

        index_to_words = {}   # dictionary mapping index to the word in the dictionary

        for w in sorted(words):

            words_to_index[w] = i

            index_to_words[i] = w

            i = i + 1

    return words_to_index, index_to_words, word_to_vec_map



word_to_index, index_to_word, word_to_vec_map = read_glove_vecs('../input/glove-global-vectors-for-word-representation/glove.6B.100d.txt')
def pretrained_embedding_layer(word_to_vec_map, word_to_index):

    vocab_len = len(word_to_index) + 1               # +1 for Keras  

    emb_dim = word_to_vec_map["happy"].shape[0]      # dimensionality of your GloVe word vectors

    

    emb_matrix = np.zeros((vocab_len, emb_dim))      # Initialization with zeros

    

    # Set each row "index" of the embedding matrix to be the word vector representation of the "index"th word of the vocabulary

    for word, index in word_to_index.items():

        emb_matrix[index, :] = word_to_vec_map[word]



    # Define Keras embedding layer with the correct output/input sizes

    embedding_layer = Embedding(vocab_len, emb_dim, trainable=False)

    

    # Build the embedding layer

    embedding_layer.build((None,))

    

    # Set the weights of the embedding layer to the embedding matrix. Your layer is now pretrained.

    embedding_layer.set_weights([emb_matrix])

    

    return embedding_layer
def sentences_to_indices(X, word_to_index, max_len):

    m = X.shape[0]                               # number of training examples

    X_indices = np.zeros((m, max_len))           # Initialize with zeros

    for i in range(m):

        sentence_words = (X[i].lower()).split()  # split each sentence into words

        j = 0

        for w in sentence_words:

            X_indices[i, j] = word_to_index[w]   # lookup index of word from vocabulary

            j = j + 1

            

    return X_indices



X_train_indices = sentences_to_indices(X_train, word_to_index, maxWords)
def Emojify(input_shape, word_to_vec_map, word_to_index):

    sentence_indices = Input(shape=input_shape, dtype='int32')

    embedding_layer = pretrained_embedding_layer(word_to_vec_map, word_to_index)

    embeddings = embedding_layer(sentence_indices)

    

    X = LSTM(128, return_sequences=True)(embeddings)

    X = Dropout(0.5)(X)

    X = LSTM(128, return_sequences=False)(X)

    X = Dropout(0.5)(X)

    X = Dense(5, activation='softmax')(X)

    X = Activation('softmax')(X)    

    

    model = Model(inputs=sentence_indices, outputs=X)

    

    return model



emojifier = Emojify((maxWords,), word_to_vec_map, word_to_index)

emojifier.summary()
reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=3, min_lr=0.00001, verbose=1)

emojifier.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

emojifier.fit(X_train_indices, Y_train_oh, epochs = 100, batch_size = 16, shuffle=True, 

                               callbacks=[reduce_lr])

X_test_indices = sentences_to_indices(X_test, word_to_index, max_len = maxWords)

Y_test_oh = pd.get_dummies(Y_test)

loss, acc = emojifier.evaluate(X_test_indices, Y_test_oh)

print()

print("Test accuracy = ", acc)
Y_test_oh = pd.get_dummies(Y_test)

X_test_indices = sentences_to_indices(X_test, word_to_index, maxWords)

pred = emojifier.predict(X_test_indices)

for i in range(len(X_test)):

    x = X_test_indices

    num = np.argmax(pred[i])

    if(num != Y_test[i]):

        print('Expected emoji:'+ label_to_emoji(Y_test[i]) + ' prediction: '+ X_test[i] + label_to_emoji(num).strip())