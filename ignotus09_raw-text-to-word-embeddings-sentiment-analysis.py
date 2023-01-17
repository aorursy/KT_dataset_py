# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os



# Any results you write to the current directory are saved as output.
imdb_dir = '/kaggle/input/imdb-raw/imdb/IMDB/aclImdb'

train_dir = os.path.join(imdb_dir, 'train')



# To store the review lables

labels = []

# To store the actual text of the review

texts = []
print(train_dir)
# Pre - Processing the lables and text data

# In the code block below we are making a list of all the comments and their correponding lables in the training data



for label_type in ['neg','pos']:

    dir_name = os.path.join(train_dir, label_type)

    for fname in os.listdir(dir_name):

        if fname[-4:] == '.txt':

            f = open(os.path.join(dir_name, fname))

            texts.append(f.read())

            f.close()

            if label_type == 'neg':

                labels.append(0)

            else:

                labels.append(1)
# Viewing the first 5 entries in the texts list and their corresponding lables

print(texts[:5])

print(labels[:5])
# The preprocessing code begins here



from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences



#Defining the number of words to read from each review

max_len = 100



#Defining the number of training samples to be considered

training_samples = 200



#vaidation samples

validation_samples = 5000



#Defining vocabulary size

max_words = 10000



tokenizer = Tokenizer(num_words = max_words) # Creates a tokenizer object configured to only take into account only "max_words" commonly occuring words 

tokenizer.fit_on_texts(texts) # Builds the word index

sequences = tokenizer.texts_to_sequences(texts) # Turns strings into list of integer indices

word_index = tokenizer.word_index # Way of recovering the word index that was computed

data = pad_sequences(sequences, maxlen=max_len) # Making sure that all the input vectors are of the same size
len(word_index)
print('Number of unique tokens:', len(word_index))
labels = np.asarray(labels)

print('Shape of data tensor:', data.shape)

print('Shape of label tensor:', labels.shape)
indices = np.arange(data.shape[0])

print(indices)
# Preparing the training and validation sets

np.random.shuffle(indices)

data = data[indices]

labels = labels[indices]

x_train = data[:training_samples]

y_train = labels[:training_samples]

x_val = data[training_samples:training_samples+validation_samples]

y_val = labels[training_samples:training_samples+validation_samples]
print(x_train.shape)

print(y_train.shape)

print(x_val.shape)

print(y_val.shape)
glove_dir = '/kaggle/input/glove6b'

embeddings_index = {}



f = open(os.path.join(glove_dir, 'glove.6B.100d.txt'))



for line in f:

    values = line.split()

    word = values[0]

    coefs = np.asarray(values[1:],dtype='float32')

    embeddings_index[word] = coefs

f.close()

print("Number of word vectors", len(embeddings_index))
# viewing the vector corresponding to the word 'rights"



embeddings_index['rights']
# Preparing the GloVe word-embeddings matrix

embedding_dim = 100

embedding_matrix = np.zeros((max_words, embedding_dim))



for word, i in word_index.items():

    if i < max_words:

        embedding_vector = embeddings_index.get(word)

        if embedding_vector is not None:

            embedding_matrix[i] = embedding_vector #words not found in the embedding index will be all zero
embedding_matrix.shape
from keras.models import Sequential

from keras.layers import Embedding, Flatten, Dense



model = Sequential()

model.add(Embedding(max_words, embedding_dim, input_length = max_len))

model.add(Flatten())

model.add(Dense(32, activation = 'relu'))

model.add(Dense(1, activation = 'sigmoid'))

model.summary()
# Loading the pretrained word embeddings into the embedding layer and freezing the embedding layer

model.layers[0].set_weights([embedding_matrix])

model.layers[0].trainable = False
model.compile(optimizer = 'rmsprop',

             loss = 'binary_crossentropy',

             metrics = ['acc'])



history = model.fit(x_train,y_train,

                   epochs=10,

                   batch_size=32,

                   validation_data=(x_val,y_val))



model.save_weights('pre_trained_glove_model.h5')
# Plotting the result



import matplotlib.pyplot as plt

acc = history.history['acc']

val_acc = history.history['val_acc']

loss = history.history['loss']

val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training acc')

plt.plot(epochs, val_acc, 'b', label='Validation acc')

plt.title('Training and validation accuracy')

plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')

plt.plot(epochs, val_loss, 'b', label='Validation loss')

plt.title('Training and validation loss')

plt.legend()

plt.show()
model = Sequential()

model.add(Embedding(max_words, embedding_dim, input_length=max_len))

model.add(Flatten())

model.add(Dense(32, activation='relu'))

model.add(Dense(1, activation='sigmoid'))

model.summary()

model.compile(optimizer='rmsprop',

loss='binary_crossentropy',

metrics=['acc'])

history = model.fit(x_train, y_train,

epochs=10,

batch_size=32,

validation_data=(x_val, y_val))
# Plotting the model output



acc = history.history['acc']

val_acc = history.history['val_acc']

loss = history.history['loss']

val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training acc')

plt.plot(epochs, val_acc, 'b', label='Validation acc')

plt.title('Training and validation accuracy')

plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')

plt.plot(epochs, val_loss, 'b', label='Validation loss')

plt.title('Training and validation loss')

plt.legend()

plt.show()