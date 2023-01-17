import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras
from tensorflow.contrib.learn import preprocessing
from keras.callbacks import ModelCheckpoint
import re
from pickle import dump
train_df = pd.read_csv('../input/train.csv')
author = train_df[train_df['author'] == 'EAP']["text"]
author[:5]
max_words = 5000 # Max size of the dictionary
tok = keras.preprocessing.text.Tokenizer(num_words=max_words)
tok.fit_on_texts(author.values)
sequences = tok.texts_to_sequences(author.values)
print(sequences[:5])
text = [item for sublist in sequences for item in sublist]
len(text)
sentence_len = 20
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
#print("Training on : "," ".join(map(lambda x: reverse_word_map[x], trainX[0])),"\nTo predict : "," ".join(map(lambda x: reverse_word_map[x], trainy[0])))
model = keras.Sequential()
model.add(keras.layers.Embedding(max_words,100,input_length=train_len))
model.add(keras.layers.LSTM(256, dropout=0.6, recurrent_dropout=0.2))
model.add(keras.layers.Dense(1024,activation="relu"))
model.add(keras.layers.Dense(4999,activation="softmax"))
model.summary()
model.compile(optimizer=keras.optimizers.Adam(lr=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
filepath = "./weight_tr5.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]
history = model.fit(np.asarray(trainX),
         pd.get_dummies(np.asarray(trainy)),
         epochs = 500,
         batch_size = 10240,
         callbacks = callbacks_list,
         verbose = 2)
def gen(seq,max_len = 20):
    sent = tok.texts_to_sequences([seq])
    #print(sent)
    while len(sent[0]) < max_len:
        sent2 = keras.preprocessing.sequence.pad_sequences(sent[-19:],maxlen=19)
        op = model.predict(np.asarray(sent2).reshape(1,-1))
        sent[0].append(op.argmax()+1)
    return " ".join(map(lambda x : reverse_word_map[x],sent[0]))
start = [("i am curious of",26),("is this why he was ",32),
         ("he was scared of such ",24),("sea was blue like nothing else ",20),
        ("the last day i colud ever enjoy",50),("could you stop doing all this you trouble me a lot",600)]
# Last one was Describe in 600 words
for i in range(len(start)):
    print("<<-- Sentence %d -->>\n"%(i),gen(start[i][0],start[i][1]))