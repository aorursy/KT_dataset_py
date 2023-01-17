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
# %load ../input/language/keras_model.py

import tensorflow as tf





def create_model(total_words, hidden_size, num_steps, optimizer='adam'):

    model = tf.keras.models.Sequential()



    # Embedding layer / Input layer

    model.add(tf.keras.layers.Embedding(

        total_words, hidden_size, input_length=num_steps))



    # 4 LSTM layers

    model.add(tf.keras.layers.LSTM(units=hidden_size, return_sequences=True))

    model.add(tf.keras.layers.LSTM(units=hidden_size, return_sequences=True))

    model.add(tf.keras.layers.LSTM(units=hidden_size, return_sequences=True))

    model.add(tf.keras.layers.LSTM(units=hidden_size, return_sequences=True))



    # Fully Connected layer

    model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1024)))

    model.add(tf.keras.layers.Activation('relu'))

    model.add(tf.keras.layers.Dropout(0.3, seed=0.2))

    model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(512)))

    model.add(tf.keras.layers.Activation('relu'))



    # Output Layer

    model.add(tf.keras.layers.TimeDistributed(

        tf.keras.layers.Dense(total_words)))

    model.add(tf.keras.layers.Activation('softmax'))



    model.compile(loss='categorical_crossentropy', optimizer=optimizer,

                  metrics=[tf.keras.metrics.categorical_accuracy])

    return model

# %load ../input/language/utils

import collections

import json

import os



import numpy as np

import tensorflow as tf



data_path = os.path.join("../input","language1")

data_path1 = os.path.join("../input","tf-tutorial-ptb-dataset")







def load_dictionary(path):

    return json.loads(open(path).read())





def read_words(filename):

    with tf.gfile.GFile(filename, 'r') as f:

        return f.read().replace('\n', '<eos>').split()





def build_vocab(filename):

    data = read_words(filename)



    counter = collections.Counter(data)

    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))



    words, _ = list(zip(*count_pairs))

    word_to_id = dict(zip(words, range(len(words))))



    return word_to_id





def file_to_word_ids(filename, word_to_id):

    data = read_words(filename)

    return [word_to_id[word] for word in data if word in word_to_id]





def load_data():

    train_path = os.path.join(data_path, 'ptb.train.txt')

    valid_path = os.path.join(data_path1, 'ptb.valid.txt')



    word_to_id = build_vocab(train_path)

    train_data = file_to_word_ids(train_path, word_to_id)

    valid_data = file_to_word_ids(valid_path, word_to_id)

    total_words = len(word_to_id)

    reversed_dictionary = dict(zip(word_to_id.values(), word_to_id.keys()))

    dictionary = {value: key for key, value in reversed_dictionary.items()}



    print('\ntotalwords : ', total_words, '\n')

    return train_data, valid_data, total_words, reversed_dictionary, dictionary





def save_json(dictionary, filename):

    with open(filename, 'w') as fp:

        json.dump(dictionary, fp)





class BatchGenerator(object):



    def __init__(self, data, num_steps, batch_size, total_words, skip_step=5):

        self.data = data

        self.num_steps = num_steps

        self.batch_size = batch_size

        self.total_words = total_words

        self.current_idx = 0

        self.skip_step = skip_step



    def generate(self):

        x = np.zeros((self.batch_size, self.num_steps))

        y = np.zeros((self.batch_size, self.num_steps, self.total_words))

        while True:

            for i in range(self.batch_size):

                if self.current_idx + self.num_steps >= len(self.data):

                    self.current_idx = 0

                x[i, :] = self.data[self.current_idx:self.current_idx + self.num_steps]

                temp_y = self.data[self.current_idx +

                                   1:self.current_idx + self.num_steps + 1]

                y[i, :, :] = tf.keras.utils.to_categorical(

                    temp_y, num_classes=self.total_words)

                self.current_idx += self.skip_step

            yield x, y

import os



import tensorflow as tf



number_of_words = 3

batch_size = 200

hidden_size = 1500

num_epochs = 80

learning_rate = 0.001

learning_rate_decay = 0



# from keras_model import create_model

# from utils import BatchGenerator, load_data, save_json



train_data, valid_data, total_words, indexToString, stringToIndex = load_data()

train_data = train_data[0:len(train_data)//5]



print(len(train_data))

train_data_generator = BatchGenerator(

    train_data, number_of_words, batch_size, total_words, skip_step=number_of_words)

valid_data_generator = BatchGenerator(

    valid_data, number_of_words, batch_size, total_words, skip_step=number_of_words)



optimizer = tf.keras.optimizers.Adam(

    lr=learning_rate, decay=learning_rate_decay)



model = create_model(total_words=total_words, hidden_size=hidden_size,

                     num_steps=number_of_words, optimizer=optimizer)



print(model.summary())



# checkpointer = tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(

#     os.getcwd(),'model-{epoch:02d}.h5'), verbose=1)



# save_json(stringToIndex,"stringtoind")



# save_json(indexToString, "indtostring")



model.fit_generator(

    generator=train_data_generator.generate(),

    steps_per_epoch=len(train_data)//(batch_size *

                                      number_of_words),

    epochs=5,

    validation_data=valid_data_generator.generate(),

    validation_steps=len(valid_data) //

    (batch_size*number_of_words),

)



model.save("model.h5")
