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
#!pip install tensorflow==2.0.0-beta1
import pandas as pd, numpy as np

import tensorflow as tf

from tensorflow import keras
train = pd.read_csv('../input/jigsaw-toxic-comment-classification-challenge/train.csv')

test = pd.read_csv('../input/jigsaw-toxic-comment-classification-challenge/test.csv')

##test_labels = pd.read_csv('../input/jigsaw-toxic-comment-classification-challenge/test_labels.csv') }Can not be loaded 

subm = pd.read_csv('../input/jigsaw-toxic-comment-classification-challenge/sample_submission.csv')

test.head()
subm.head()
train.head()
text = train['comment_text']
text[0]
train['comment_text'][0]
# for train

lens = train.comment_text.str.len()

lens.mean(), lens.std(), lens.max()
# for test

lens = test.comment_text.str.len()

lens.mean(), lens.std(), lens.max()
lens.hist();
label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

train['none'] = 1-train[label_cols].max(axis=1) ## each colum may have the value of one ( Labled ) . 1- calc the max # if has no lable max = 0 then col = 1 -0 = 0

train.describe()
len(train),len(test)
## deal with nulls 

COMMENT = 'comment_text'

train[COMMENT].fillna("unknown", inplace=True)

test[COMMENT].fillna("unknown", inplace=True)
import re, string

re_tok = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])')

def tokenize(s): return re_tok.sub(r' \1 ', s).split()

def clean(s): return re_tok.sub(r' \1 ', s)
## decide vocab size 

words = []

for t in text:

    words.extend(tokenize(t))

print(words[:100])

vocab = list(set(words))

print(len(words), len(vocab))
train['comment_text'][0]
clean(train['comment_text'][0])
def one_hot_word_embedding(vtrain_data,vtest_data):

    # switch data back to text 

    train_labels = vtrain_data[label_cols]

    txt_train_data = [clean(txt) for txt in train['comment_text']]

    txt_test_data = [clean(txt) for txt in test['comment_text']]

    

    # integer encode the documents

    vocab_size = 10000

    encoded_txt_train_data = [keras.preprocessing.text.one_hot(d, vocab_size) for d in txt_train_data]

    encoded_txt_test_data = [keras.preprocessing.text.one_hot(d, vocab_size) for d in txt_test_data]

    #print(encoded_txt_train_data)



    ptxt_train_data = keras.preprocessing.sequence.pad_sequences(encoded_txt_train_data,

                                                            padding='post',

                                                            maxlen=5000)



    ptxt_test_data = keras.preprocessing.sequence.pad_sequences(encoded_txt_test_data,

                                                           padding='post',

                                                           maxlen=5000)

    x_val = ptxt_train_data[:100000] 

    partial_x_train = ptxt_train_data[100000:]



    y_val = train_labels[:100000]

    partial_y_train = train_labels[100000:]

    return (x_val,partial_x_train,y_val,partial_y_train,ptxt_test_data)



def full_one_hot_word_embedding(vtrain_data,vtest_data):

    # switch data back to text 

    train_labels = vtrain_data[label_cols]

    txt_train_data = [clean(txt) for txt in train['comment_text']]

    txt_test_data = [clean(txt) for txt in test['comment_text']]

    

    # integer encode the documents

    vocab_size = 10000

    encoded_txt_train_data = [keras.preprocessing.text.one_hot(d, vocab_size) for d in txt_train_data]

    encoded_txt_test_data = [keras.preprocessing.text.one_hot(d, vocab_size) for d in txt_test_data]

    #print(encoded_txt_train_data)



    ptxt_train_data = keras.preprocessing.sequence.pad_sequences(encoded_txt_train_data,

                                                            padding='post',

                                                            maxlen=5000)



    ptxt_test_data = keras.preprocessing.sequence.pad_sequences(encoded_txt_test_data,

                                                           padding='post',

                                                           maxlen=5000)

    partial_x_train = ptxt_train_data

    partial_y_train = train_labels

    return (partial_x_train,partial_y_train,ptxt_test_data)
def model_with_emb_acc(vtrain_data,vtest_data,vocab_size = 10000):

    model1 = keras.Sequential()

    model1.add(keras.layers.Embedding(vocab_size, 16))

    model1.add(keras.layers.GlobalAveragePooling1D())

    model1.add(keras.layers.Dense(512, activation=tf.nn.relu))

    #model.add(keras.layers.Dense(16, activation=tf.nn.relu,activity_regularizer=keras.regularizers.l1(0.001)))

    #model.add(keras.layers.Dropout(0.2))

    model1.add(keras.layers.Dense(6, activation=tf.nn.sigmoid))

    model1.compile(optimizer='adam',

              loss='binary_crossentropy',

              metrics=['acc'])

    x_val,partial_x_train,y_val,partial_y_train,test_data = one_hot_word_embedding(vtrain_data,vtest_data)

    earlystopper = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=1)

    history = model1.fit(x_val,

                     y_val,

                     epochs=20,

                     callbacks=[earlystopper],

                     batch_size=512,

                     validation_data=(x_val, y_val),

                     verbose=1)

#     results1 = model1.evaluate(x_val, y_val)

    return (model1,test_data,history)



def full_model_with_emb_acc(vtrain_data,vtest_data,vocab_size = 10000):

    model1 = keras.Sequential()

    model1.add(keras.layers.Embedding(vocab_size, 16))

    model1.add(keras.layers.GlobalAveragePooling1D())

    model1.add(keras.layers.Dense(512, activation=tf.nn.relu))

    #model.add(keras.layers.Dense(16, activation=tf.nn.relu,activity_regularizer=keras.regularizers.l1(0.001)))

    #model.add(keras.layers.Dropout(0.2))

    model1.add(keras.layers.Dense(6, activation=tf.nn.sigmoid))

    model1.compile(optimizer='adam',

              loss='binary_crossentropy',

              metrics=['acc'])

    partial_x_train,partial_y_train,test_data = full_one_hot_word_embedding(vtrain_data,vtest_data)

    earlystopper = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=1)

    history = model1.fit(partial_x_train,

                     partial_y_train,

                     epochs=20,

                     callbacks=[earlystopper],

                     batch_size=1024,

                     verbose=1)

#     results1 = model1.evaluate(x_val, y_val)

    return (model1,test_data,history)
model1,test_data,his1 = full_model_with_emb_acc(train,test)
# serialize model to JSON

model_json = model1.to_json()

with open("my_model1.json", "w") as json_file:

    json_file.write(model_json)

    

model1.save_weights('my_model1_weights.h5')
### load the model 1 

# load json and create model

json_file = open('my_model1.json', 'r')

loaded_model_json = json_file.read()

json_file.close()

model1 = tf.keras.models.model_from_json(loaded_model_json)

# load weights into new model

model1.load_weights("my_model1_weights.h5")

print("Loaded model from disk")
def column(matrix, i):

    return [row[i] for row in matrix]
y_pred = model1.predict(test_data, batch_size=1024)
submission = pd.DataFrame()

submission['id'] = test['id']

submission['toxic'] = column(y_pred, 0)

submission['severe_toxic'] = column(y_pred, 1)

submission['obscene'] = column(y_pred, 2)

submission['threat'] = column(y_pred, 3)

submission['insult'] = column(y_pred, 4)

submission['identity_hate'] = column(y_pred, 5)
submission.to_csv('submission.csv', index=False)
submission