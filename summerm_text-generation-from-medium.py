import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import keras

from tensorflow.contrib.learn import preprocessing

from keras.callbacks import ModelCheckpoint

import re

from pickle import dump

import tensorflow as tf

import os

from sklearn.utils import shuffle





print(os.listdir("../input"))



print(tf.test.gpu_device_name())

# See https://www.tensorflow.org/tutorials/using_gpu#allowing_gpu_memory_growth

config = tf.ConfigProto()

config.gpu_options.allow_growth = True
# PREPROCESSING



train_df = pd.read_csv('../input/Medium_AggregatedData.csv', skipinitialspace=True, usecols = ['text', 'totalClapCount', 'language']).rename(columns=lambda x: x.strip())

train_df = train_df.drop_duplicates()

train_df = shuffle(train_df)



columns = ["text", "totalClapCount", "language"]

train_df = train_df.reindex(columns=columns)



train_df = train_df[train_df['totalClapCount'] > 1000][train_df['language'] == 'en']["text"].apply(lambda x: x.lower().replace('\n', ' ').replace(',', '').replace('.', ''))



print("PREPROCESSING complete with quantity {}".format(len(train_df)))
train_df[:5]

train_df.tail()
# TOKENIZATION



max_words = 1250 #1250 # 1000 # 5000 # Max size of the dictionary

tok = keras.preprocessing.text.Tokenizer(num_words=max_words)

tok.fit_on_texts(train_df.values)

sequences = tok.texts_to_sequences(train_df.values)

#sequences = [ np.int16(x) for x in sequences ]

print(sequences[:5])
# Flatten sequence list.

text = [item for sublist in sequences for item in sublist]

len(text)
sentence_len = 14 #7 #20

pred_len = 1

train_len = sentence_len - pred_len

seq = []

# Sliding window to generate test and train data

for i in range(len(text)-sentence_len):

    seq.append(text[i:i+sentence_len])

# Reverse dictionary so as to decode tokenized sequences back to words and sentences

reverse_word_map = dict(map(reversed, tok.word_index.items()))

dump(tok, open('tokenizer.pkl', 'wb'))
trainX = []

trainy = []

for i in seq:

    trainX.append(i[:train_len])

    trainy.append(i[-1])
model = keras.Sequential()

model.add(keras.layers.Embedding(max_words,100,input_length=train_len)) #100

model.add(keras.layers.LSTM(200, dropout=0.6, recurrent_dropout=0.2)) #128 #256

model.add(keras.layers.Dense(300,activation="relu")) #128 #1024

model.add(keras.layers.Dense(max_words-1,activation="softmax")) #4999

model.summary()
#filepath = "./weight_tr5.hdf5"

checkpoint_path = "training/cp-{epoch:04d}.ckpt"

checkpoint_dir = os.path.dirname(checkpoint_path)



if os.path.isdir(checkpoint_dir):

  print("Reloading checkpointed model from {}".format(checkpoint_dir))

  latest = tf.train.latest_checkpoint(checkpoint_dir)

  print(latest)

  model.load_weights(latest)

  #model.load_weights(filepath)

  #print(model.summary())
print(os.listdir(checkpoint_dir))

print(os.listdir("training/"))
model.compile(optimizer=keras.optimizers.Adam(lr=0.001),

              loss='categorical_crossentropy',

              metrics=['accuracy'])



checkpoint = ModelCheckpoint(checkpoint_path, #filepath, 

                             monitor='loss', 

                             verbose=1,  

                             save_weights_only=True,

                             #save_best_only=True, 

                             period=1, # Save weights, every 5-epochs.

                             mode='min') 

callbacks_list = [checkpoint]

    

# Fit model using the gpu.

with tf.device('/gpu:0'):

  history = model.fit(np.asarray(trainX).astype(np.int16), #(np.int8),

           np.asarray(pd.get_dummies(np.asarray(trainy),sparse=True)).astype(np.int16), #(np.int8),

           epochs = 500,

           batch_size = 2000, #200, #10240,

           callbacks = callbacks_list,

           verbose = 2)
def gen(seq,max_len = 7): #19

    sent = tok.texts_to_sequences([seq])

    #print(sent)

    while len(sent[0]) < max_len:

        sent2 = keras.preprocessing.sequence.pad_sequences(sent[-(sentence_len-1):],maxlen=(sentence_len-1))

        op = model.predict(np.asarray(sent2).reshape(1,-1))

        sent[0].append(op.argmax()+1)

    return " ".join(map(lambda x : reverse_word_map[x],sent[0]))
start = [("this generation is remarkably ",20),("seriously. every single time ",32),

         ("he couldn't stand such ",24),("and in that moment, he felt ",20),

        ("the last day they could ever enjoy ",50),("",600)]



for i in range(len(start)):

    print("<<-- Sentence %d -->>\n"%(i), gen(start[i][0],start[i][1]))
