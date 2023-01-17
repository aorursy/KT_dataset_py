# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import random

from copy import deepcopy



import tensorflow as tf

import tensorflow.keras as keras

import tensorflow.keras.layers as layers



from tensorflow.keras.preprocessing import sequence

import tensorflow.keras.backend as K



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

import math



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
total_data = pd.read_csv("../input/rugty8fmubjefxn/train.csv")

label_psv = total_data[['label']].values.tolist()

label_data = np.array(label_psv)



tot_train_x1 = np.load('../input/scatter-tfidf/tfidf/train_s1.npy')

tot_train_x2 = np.load('../input/scatter-tfidf/tfidf/train_s2.npy')

test_x1 = np.load('../input/scatter-tfidf/tfidf/test_s1.npy')

test_x2 = np.load('../input/scatter-tfidf/tfidf/test_s2.npy')



train_x1 = tot_train_x1[:, :20]

train_x2 = tot_train_x2[:, :20]

test_x1 = test_x1[:, :20]

test_x2 = test_x2[:, :20]



train_x1_tf = tot_train_x1[:, 20:]

train_x2_tf = tot_train_x2[:, 20:]



# masked_x1 = []

# masked_x2 = []

# masked_label = []

# for i in np.random.choice(len(train_x1), int(len(train_x1) * 0.15)):

#     tmp_1 = deepcopy(train_x1[i])

#     start_sentence_1 = np.where(train_x1[i] > 0)[0][0]

#     tmp_2 = deepcopy(train_x2[i])

#     start_sentence_2 = np.where(train_x2[i] > 0)[0][0]

    

#     if start_sentence_1 < 19 and start_sentence_2 < 19:

#         tmp_1[start_sentence_1 + np.argmin(train_x1_tf[i, start_sentence_1:])] = 0

#         masked_x1.append(tmp_1)

        

#         tmp_2[start_sentence_2 + np.argmin(train_x2_tf[i, start_sentence_2:])] = 0

#         masked_x2.append(tmp_2)



#         masked_label.append(label_data[i])



# masked_x1 = np.array(masked_x1)

# masked_x2 = np.array(masked_x2)

# masked_label = np.array(masked_label)



# train_x1 = np.concatenate([train_x1, masked_x1])

# train_x2 = np.concatenate([train_x2, masked_x2])

# label_data = np.concatenate([label_data, masked_label])



print(label_data.shape)

print(train_x1.shape)

print(train_x2.shape)

print(test_x1.shape)

print(test_x2.shape)
def ma_dist(x, y):

     return tf.math.exp(-1 * tf.math.reduce_sum(tf.math.abs(tf.math.subtract(x, y)), axis=1, keepdims=True))



def build_model(embed_dim=128):

    embed_layer = layers.Embedding(30006, embed_dim, trainable=True)



    input_1 = keras.Input(shape=(20,))

    x1 = embed_layer(input_1)



    input_2 = keras.Input(shape=(20,))

    x2 = embed_layer(input_2)



    siam_lstm = layers.LSTM(256)



    x1 = siam_lstm(x1)

    x2 = siam_lstm(x2)

    

    x = ma_dist(x1, x2)

    return keras.Model(inputs=[input_1, input_2], outputs=x)



model = build_model()

model.summary()
save_best = tf.keras.callbacks.ModelCheckpoint('Oct15_best.h5', monitor='val_loss', save_best_only=True,)

early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode=min, patience=10)

# class_weights = {0:2.47, 1:1.52}

# class_weights = {0:1, 1:1}



model.compile(optimizer='adam', loss='BinaryCrossentropy', metrics=['accuracy'])

model.fit(x=[train_x1, train_x2], y=label_data, validation_split=0.1, epochs=100, callbacks=[save_best, early_stop])
del model

model = tf.keras.models.load_model('Oct15_best.h5')

pred = model.predict([tot_train_x1[:, :20], tot_train_x2[:, :20]])



accr = 0

for id, i in enumerate(pred):

    tmp_res = 0

    if i >= 0.5:

        tmp_res = 1



    if tmp_res == label_data[id]:

        accr += 1



print(accr / 40000)
import csv 



test_output = model.predict([test_x1, test_x2]).flatten()

print(test_output.shape)



with open('output.csv', 'w', encoding='utf-8') as f:

    wr = csv.writer(f)



    wr.writerow(['id', "label"])

    for id, i in enumerate(test_output):

        if i >= 0.5:

            wr.writerow([id + 40001, 1])

        else:

            wr.writerow([id + 40001, 0])
output = pd.read_csv("./output.csv")

print(output)

output = pd.read_csv("../input/rugty8fmubjefxn/sample_submission.csv")

print(output)