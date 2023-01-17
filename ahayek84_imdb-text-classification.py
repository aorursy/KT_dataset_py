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
# keras.datasets.imdb is broken in 1.13 and 1.14, by np 1.16.3

#!pip install tf_nightly
#!pip install tensorflow==2.0.0-beta1
from __future__ import absolute_import, division, print_function, unicode_literals



import tensorflow as tf

from tensorflow import keras



import numpy as np



print(tf.__version__)
imdb = keras.datasets.imdb



(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
print("Training entries: {}, labels: {}".format(len(train_data), len(train_labels)))
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
decode_review(train_data[24999])
def one_hot_word_embedding():

    # switch data back to text 

    txt_train_data = [decode_review(txt) for txt in train_data]

    txt_test_data = [decode_review(txt) for txt in train_data]

    

    # integer encode the documents

    vocab_size = 50

    encoded_txt_train_data = [keras.preprocessing.text.one_hot(d, vocab_size) for d in txt_train_data]

    encoded_txt_test_data = [keras.preprocessing.text.one_hot(d, vocab_size) for d in txt_test_data]

    #print(encoded_txt_train_data)



    ptxt_train_data = keras.preprocessing.sequence.pad_sequences(train_data,

                                                            padding='post',

                                                            maxlen=256)



    ptxt_test_data = keras.preprocessing.sequence.pad_sequences(test_data,

                                                           padding='post',

                                                           maxlen=256)

    x_val = ptxt_train_data[:10000]

    partial_x_train = ptxt_train_data[10000:]



    y_val = train_labels[:10000]

    partial_y_train = train_labels[10000:]

    return (x_val,partial_x_train,y_val,partial_y_train,ptxt_test_data)
def normal_embedding(vtrain_data,vtest_data):

    train_data = keras.preprocessing.sequence.pad_sequences(vtrain_data,

                                                        value=word_index["<PAD>"],

                                                        padding='post',

                                                        maxlen=256)



    test_data = keras.preprocessing.sequence.pad_sequences(vtest_data,

                                                         value=word_index["<PAD>"],

                                                         padding='post',

                                                         maxlen=256)

    x_val = train_data[:10000]

    partial_x_train = train_data[10000:]



    y_val = train_labels[:10000]

    partial_y_train = train_labels[10000:]

    return (x_val,partial_x_train,y_val,partial_y_train,test_data)
def model_without_emb_acc(vtrain_data,vtest_data):

    model1 = keras.Sequential()

    model1.add(keras.layers.Dense(512, input_dim=256, kernel_initializer='normal', activation='relu'))

    model1.add(keras.layers.Dense(256, activation=tf.nn.relu))

    #model.add(keras.layers.Dense(16, activation=tf.nn.relu,activity_regularizer=keras.regularizers.l1(0.001)))

    #model.add(keras.layers.Dropout(0.2))

    model1.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))

    model1.compile(optimizer='adam',

              loss='binary_crossentropy',

              metrics=['acc'])

    x_val,partial_x_train,y_val,partial_y_train,test_data = normal_embedding(vtrain_data,vtest_data)

    earlystopper = keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, verbose=1)

    history = model1.fit(x_val,

                     y_val,

                     epochs=40,

                     callbacks=[earlystopper],

                     batch_size=512,

                     validation_data=(x_val, y_val),

                     verbose=0)

    results1 = model1.evaluate(test_data, test_labels)

    return (results1,history)



def model_with_emb_acc(vtrain_data,vtest_data,vocab_size = 10000):

    model1 = keras.Sequential()

    model1.add(keras.layers.Embedding(vocab_size, 16))

    model1.add(keras.layers.GlobalAveragePooling1D())

    model1.add(keras.layers.Dense(512, activation=tf.nn.relu))

    #model.add(keras.layers.Dense(16, activation=tf.nn.relu,activity_regularizer=keras.regularizers.l1(0.001)))

    #model.add(keras.layers.Dropout(0.2))

    model1.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))

    model1.compile(optimizer='adam',

              loss='binary_crossentropy',

              metrics=['acc'])

    x_val,partial_x_train,y_val,partial_y_train,test_data = normal_embedding(vtrain_data,vtest_data)

    earlystopper = keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, verbose=1)

    history = model1.fit(x_val,

                     y_val,

                     epochs=40,

                     callbacks=[earlystopper],

                     batch_size=512,

                     validation_data=(x_val, y_val),

                     verbose=0)

    results1 = model1.evaluate(test_data, test_labels)

    return (results1,history)
res1,his1 = model_without_emb_acc(train_data,test_data)

res1
vocab_sizes = range(10000,100001,10000)

res = pd.DataFrame({})

vs = []

loss = []

acc = []

his = []

for vocab_size in vocab_sizes:

    tmp,his1 = model_with_emb_acc(train_data,test_data,vocab_size)

    vs.append(vocab_size)

    loss.append(tmp[0])

    acc.append(tmp[1])

    his.append(his1)

res['vsize'] = vs

res['loss'] = loss

res['acc'] = acc

print(res)
test_loss, test_acc = model.evaluate(test_images, test_labels)
import seaborn as sns; sns.set()

import matplotlib.pyplot as plt
def plot_tra_val_acc(histo):

    history_dict = his1.history

    history_dict.keys()

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
#### Plotssss 
ax = sns.lineplot(x="vsize", y="acc", data=res)
ax = sns.lineplot(x="vsize", y="loss", data=res)
ax = sns.lineplot(x="loss", y="acc", data=res)

plot_tra_val_acc(his1) ## Plot for no embedding 
his1.history
print('-------------------- Hist 0 --------------------------')

plot_tra_val_acc(his[0]) ## Plot for embedding 

print('----------------------------------------------')