from __future__ import absolute_import, division, print_function



import tensorflow as tf

from tensorflow import keras



import numpy as np



print(tf.__version__)
imdb = keras.datasets.imdb



(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
print("Training entries: {}, labels: {}".format(len(train_data), len(train_labels)))
print(train_data[0])
len(train_data[0]), len(train_data[1])
# A dictionary mapping words to an integer index

word_index = imdb.get_word_index()



# The first indices are reserved

word_index = {k:(v+3) for k,v in word_index.items()} 

word_index["<PAD>"] = 0

word_index["<START>"] = 1

word_index["<UNK>"] = 2  # unknown

word_index["<UNUSED>"] = 3



reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])



def decode_review(text):

    return ' '.join([reverse_word_index.get(i, '?') for i in text])
decode_review(train_data[0])
train_data = keras.preprocessing.sequence.pad_sequences(train_data,

                                                        value=word_index["<PAD>"],

                                                        padding='post',

                                                        maxlen=256)



test_data = keras.preprocessing.sequence.pad_sequences(test_data,

                                                       value=word_index["<PAD>"],

                                                       padding='post',

                                                       maxlen=256)
len(train_data[0]), len(train_data[1])
print(train_data[0])
# input shape is the vocabulary count used for the movie reviews (10,000 words)

vocab_size = 10000



model = keras.Sequential()

model.add(keras.layers.Embedding(vocab_size, 16))

model.add(keras.layers.GlobalAveragePooling1D())

model.add(keras.layers.Dense(16, activation=tf.nn.relu))

model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))



model.summary()
model.compile(optimizer='adam',

              loss='binary_crossentropy',

              metrics=['acc'])
x_val = train_data[:10000]

partial_x_train = train_data[10000:]



y_val = train_labels[:10000]

partial_y_train = train_labels[10000:]
history = model.fit(partial_x_train,

                    partial_y_train,

                    epochs=40,

                    batch_size=512,

                    validation_data=(x_val, y_val),

                    verbose=1)
results = model.evaluate(test_data, test_labels)



print(results)
history_dict = history.history

history_dict.keys()
import matplotlib.pyplot as plt



acc = history_dict['acc']

val_acc = history_dict['val_acc']

loss = history_dict['loss']

val_loss = history_dict['val_loss']



epochs = range(1, len(acc) + 1)



# "bo" is for "blue dot"

plt.plot(epochs, loss, 'bo', label='Training loss')

# b is for "solid blue line"

plt.plot(epochs, val_loss, 'b', label='Validation loss')

plt.title('Training and validation loss')

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.legend()



plt.show()
plt.clf()   # clear figure



plt.plot(epochs, acc, 'bo', label='Training acc')

plt.plot(epochs, val_acc, 'b', label='Validation acc')

plt.title('Training and validation accuracy')

plt.xlabel('Epochs')

plt.ylabel('Accuracy')

plt.legend()



plt.show()
#@title Licensed under the Apache License, Version 2.0 (the "License");

# you may not use this file except in compliance with the License.

# You may obtain a copy of the License at

#

# https://www.apache.org/licenses/LICENSE-2.0

#

# Unless required by applicable law or agreed to in writing, software

# distributed under the License is distributed on an "AS IS" BASIS,

# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

# See the License for the specific language governing permissions and

# limitations under the License.
#@title MIT License

#

# Copyright (c) 2017 François Chollet

#

# Permission is hereby granted, free of charge, to any person obtaining a

# copy of this software and associated documentation files (the "Software"),

# to deal in the Software without restriction, including without limitation

# the rights to use, copy, modify, merge, publish, distribute, sublicense,

# and/or sell copies of the Software, and to permit persons to whom the

# Software is furnished to do so, subject to the following conditions:

#

# The above copyright notice and this permission notice shall be included in

# all copies or substantial portions of the Software.

#

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR

# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,

# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL

# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER

# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING

# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER

# DEALINGS IN THE SOFTWARE.