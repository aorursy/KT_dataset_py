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
!ls
!pip install keras-bert

!pip install keras-rectified-adam



!wget -q https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip

!unzip -o uncased_L-12_H-768_A-12.zip
import codecs

import tensorflow as tf

from tqdm import tqdm

from chardet import detect

import keras

from keras_radam import RAdam

from keras import backend as K

from keras_bert import load_trained_model_from_checkpoint

# from google.colab import drive





SEQ_LEN = 128

BATCH_SIZE = 50

EPOCHS = 7

LR = 1e-4




pretrained_path = 'uncased_L-12_H-768_A-12'

config_path = os.path.join(pretrained_path, 'bert_config.json')

checkpoint_path = os.path.join(pretrained_path, 'bert_model.ckpt')

vocab_path = os.path.join(pretrained_path, 'vocab.txt')

model = load_trained_model_from_checkpoint(

      config_path,

      checkpoint_path,

      training=True,

      trainable=True,

      seq_len=SEQ_LEN,

  )
model.summary()
import codecs

from keras_bert import Tokenizer

token_dict = {}

with codecs.open(vocab_path, 'r', 'utf8') as reader:

    for line in reader:

        token = line.strip()

        token_dict[token] = len(token_dict)

        

# print(token_dict)
# @title Download I20Newsgroup dataset

import tensorflow as tf



dataset = tf.keras.utils.get_file(

    fname="20news-18828.tar.gz", 

    origin="http://qwone.com/~jason/20Newsgroups/20news-18828.tar.gz", 

    extract=True,

)
tokenizer = Tokenizer(token_dict)

datapath = ".".join(dataset.split(".")[:-2])

txtfiles = os.listdir(datapath)

labels = [(x, i) for i,x in enumerate(txtfiles)]

def get_label(index):

    for each in labels:

        if index == each[1]:

            return each[0]

    

    
def load_data(path, labels):

    global tokenizer

    indices, sentiments = [], []

    for folder, sentiment in labels:

        folder = os.path.join(path, folder)

        for name in tqdm(os.listdir(folder)):

            with open(os.path.join(folder, name), 'r', encoding="utf-8", errors='ignore') as reader:

                  text = reader.read()

            ids, segments = tokenizer.encode(text, max_len=SEQ_LEN)

            indices.append(ids)

            sentiments.append(sentiment)

    items = list(zip(indices, sentiments))

    

    np.random.shuffle(items)

    test_items = items[int(0.8*len(items)):]

    train_items = items[:int(0.8*len(items))]

    indices_test, sentiments_test = zip(*test_items)

    indices_train, sentiments_train = zip(*train_items)

    indices_train = np.array(indices_train)

    indices_test = np.array(indices_test)

    mod_train = indices_train.shape[0] % BATCH_SIZE

    mod_test = indices_test.shape[0] % BATCH_SIZE

    if mod_train > 0:

        indices_train, sentiments_train = indices_train[:-mod_train], sentiments_train[:-mod_train]

    if mod_test > 0:

      indices_test, sentiments_test = indices_test[:-mod_test], sentiments_test[:-mod_test]



    return [indices_train, np.zeros_like(indices_train)], np.array(sentiments_train),[indices_test, np.zeros_like(indices_test)], np.array(sentiments_test)

  

train_path = os.path.join(os.path.dirname(dataset), '20news-18828')

train_x, train_y, test_x, test_y = load_data(train_path, labels)
pd.Series(train_y).value_counts().plot(kind = 'bar')
pd.Series(test_y).value_counts().plot(kind = 'bar')
inputs = model.inputs[:2]

dense = model.get_layer('NSP-Dense').output

outputs = keras.layers.Dense(units=20, activation='softmax')(dense)



model = keras.models.Model(inputs, outputs)

model.compile(

  RAdam(learning_rate =LR),

  loss='sparse_categorical_crossentropy',

  metrics=['sparse_categorical_accuracy'],

)

sess = K.get_session()

uninitialized_variables = set([i.decode('ascii') for i in sess.run(tf.report_uninitialized_variables())])

init_op = tf.variables_initializer(

    [v for v in tf.global_variables() if v.name.split(':')[0] in uninitialized_variables]

)

sess.run(init_op)

# @title Fit



model.fit(

    train_x,

    train_y,

    epochs=EPOCHS,

    batch_size=BATCH_SIZE,

)




predicts = model.predict(test_x, verbose=True).argmax(axis=-1)


print(np.sum(test_y == predicts) / test_y.shape[0])


test_text = """The Mumbai batsman is set to replace underperforming KL Rahul as an opener in upcoming Test series against South Africa in home conditions.India’s newly-appointed batting coach Vikram Rathour feels opener Rohit Sharma is “too good a player” to not be playing in all three formats. Rathour, like many former cricketers, backed Rohit to open for India in Test cricket.“He is too good a player to not be playing in any game. That is what is everyone is thinking. He has done so well in white-ball cricket as an opener so there is no reason why he can’t succeed as a Test opener provided he gets enough opportunities,” Rathour believes Rohit can be an asset to his team if he does good against South Africa in Tests."""

test = """Senate Democrats are planning to hold the floor on Tuesday evening for an hours-long talk-a-thon on the issue of gun violence.The floor marathon comes as the White House is struggling to find a place to land in the weeks-long debate over potential gun-law reforms.“Many of my colleagues have seen their communities torn apart by gun violence; some by horrific mass shootings, others by a relentless, daily stream. Many of them have worked for years to bring commonsense gun safety measures before the Senate,” Senate Minority Leader Charles Schumer (D-N.Y.) said Tuesday, in announcing the plan from the Senate floor."""

ids, segments = tokenizer.encode(test, max_len=SEQ_LEN)
inpu = np.array(ids).reshape([1, SEQ_LEN])

get_label(model.predict([inpu,np.zeros_like(inpu)]).argmax(axis=-1)[0])

ids, segments = tokenizer.encode(test_text, max_len=SEQ_LEN)

inpu = np.array(ids).reshape([1, SEQ_LEN])

get_label(model.predict([inpu,np.zeros_like(inpu)]).argmax(axis=-1)[0])