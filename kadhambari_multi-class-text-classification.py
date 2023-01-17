# to be imported

from keras.preprocessing.text import text_to_word_sequence

import pandas as pd

from keras.preprocessing.text import Tokenizer

import numpy as np

from __future__ import print_function



from keras.preprocessing import sequence

from keras.models import Sequential

from keras.layers import Dense, Dropout, Activation

from keras.layers import Embedding

from keras.layers import Conv1D, GlobalMaxPooling1D


# Read the input dataset 

d = pd.read_csv("../input/consumer_complaints.csv", 

                usecols=('product','consumer_complaint_narrative'),

                dtype={'consumer_complaint_narrative': object})

# Only interested in data with consumer complaints

d=d[d['consumer_complaint_narrative'].notnull()]



d=d[d['product'].notnull()]

d.reset_index(drop=True,inplace=True)

x = d.iloc[:, 1].values

y = d.iloc[:, 0].values

print(y)



#there are 11 unique classes for classification

print(np.unique(y, return_counts=True))



 # encode the text with word sequences - Preprocessing step 1

tk = Tokenizer(num_words= 200, filters = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',lower=True, split=" ")

tk.fit_on_texts(x)

x = tk.texts_to_sequences(x)

x = sequence.pad_sequences(x, maxlen=200)



print(x)
 # Label Encoding categorical data for the classification category

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder_Y = LabelEncoder()

y = labelencoder_Y.fit_transform(y)

print(y)

print(np.unique(y, return_counts=True))
# Perform one hot encoding 

from keras import utils as np_utils

y = np_utils.to_categorical(y, num_classes= 11)



print(y)
# Seeding

np.random.seed(200)

indices = np.arange(len(x))

np.random.shuffle(indices)

x = x[indices]

y = y[indices]
index_from=3

start_char = 1

if start_char is not None:

        x = [[start_char] + [w + index_from for w in x1] for x1 in x]

elif index_from:

        x = [[w + index_from for w in x1] for x1 in x]




num_words = None

if not num_words:

        num_words = max([max(x1) for x1 in x])

        

oov_char = 2

skip_top = 0

# by convention, use 2 as OOV word

# reserve 'index_from' (=3 by default) characters:

# 0 (padding), 1 (start), 2 (OOV)

if oov_char is not None:

        x = [[w if (skip_top <= w < num_words) else oov_char for w in x1] for x1 in x]

else:

        x = [[w for w in x1 if (skip_top <= w < num_words)] for x1 in x]

        

# split test and train data

test_split = 0.2

idx = int(len(x) * (1 - test_split))

x_train, y_train = np.array(x[:idx]), np.array(y[:idx])

x_test, y_test = np.array(x[idx:]), np.array(y[idx:])



print(x_train.shape)

print(y_train.shape)

print(x_test.shape)

print(y_test.shape)

print(y)
x_train = sequence.pad_sequences(x_train, maxlen=201)

x_test = sequence.pad_sequences(x_test, maxlen=201)

print('x_train shape:', x_train.shape)

print('x_test shape:', x_test.shape)


max_features = 1000

maxlen = 201

embedding_dims = 50

filters = 250

kernel_size = 3

hidden_dims = 250





# CNN with max pooling imeplementation 

print('Build model...')

model = Sequential()

# we start off with an efficient embedding layer which maps

# our vocab indices into embedding_dims dimensions

model.add(Embedding(max_features,

                    embedding_dims,

                    input_length=maxlen))

model.add(Dropout(0.2))



# we add a Convolution1D, which will learn filters

# word group filters of size filter_length:

model.add(Conv1D(filters,

                 kernel_size,

                 padding='valid',

                 activation='relu',

                 strides=1))

# we use max pooling:

model.add(GlobalMaxPooling1D())



# We add a vanilla hidden layer:

model.add(Dense(hidden_dims))

model.add(Dropout(0.2))

model.add(Activation('relu'))



# We project onto a single unit output layer, and squash it with a sigmoid:

model.add(Dense(11))

model.add(Activation('softmax'))



model.compile(loss='categorical_crossentropy',

              optimizer='rmsprop',

              metrics=['accuracy'])



model.fit(x_train, y_train,

          batch_size=32,

          epochs=50,

          validation_data=(x_test, y_test))