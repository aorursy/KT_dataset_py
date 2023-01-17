# Parameters

TRAIN_INPUT = 'twitgen_train_201906011956.csv'

VALID_INPUT = 'twitgen_valid_201906011956.csv'

TEST_INPUT = 'twitgen_test_201906011956.csv'

EMBEDDING_DIM = 200

MAXLEN = 50  # Maximum number of words per tweet that will be processed



DEBUG = False
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt # plotting

import numpy as np # linear algebra

import tensorflow as tf

import pandas as pd

import os

import re

import keras

from keras import backend as K

import keras.layers as layers

from keras.models import Model, load_model

from keras.engine import Layer

from keras.optimizers import Adam, Adagrad

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

import numpy as np

from datetime import datetime

import string

import numpy as np

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score

from sklearn.metrics import f1_score

from sklearn.metrics import roc_curve, auc

from statsmodels.stats.proportion import proportion_confint

import matplotlib.pyplot as plt



keras.__version__
print(os.listdir('../input'))
basepath = '/kaggle/input/tweet-files-for-gender-guessing/'

glovefile = 'glove.twitter.27B.200d.txt'

glovebase = '/kaggle/input/glovetwitter27b100dtxt/'

glovepath = glovebase + glovefile
!ls $glovepath
# Read in the data

df_train = pd.read_csv(basepath+TRAIN_INPUT, index_col=['id','time'], parse_dates=['time'])

df_valid = pd.read_csv(basepath+VALID_INPUT, index_col=['id','time'], parse_dates=['time'])

df_test = pd.read_csv(basepath+TEST_INPUT, index_col=['id','time'], parse_dates=['time'])

df_train.head()
# Maximum number of words per tweet in each data set

(df_train.text.str.split().apply(len).max(), 

 df_valid.text.str.split().apply(len).max(),

 df_test.text.str.split().apply(len).max())
# Text Normalization function



# Taken from 

# https://medium.com/@sabber/classifying-yelp-review-comments-using-lstm-and-word-embeddings-part-1-eb2275e4066b

# which was taken from https://www.kaggle.com/lystdo/lstm-with-word2vec-embeddings

# but this version no longer does stemming or stop word elmination



# This is for general text, not Twitter-specific.

# Maybe would get a better classifier if we used a Python transaltion of this:

# https://nlp.stanford.edu/projects/glove/preprocess-twitter.rb

# but that is arguably outside the scope of this project

# and my initial attempts to use Twitter-specific preprocessing have been unsuccessful





def clean_text(text):

    

    ## Remove puncuation

    text = text.translate(string.punctuation)

    

    ## Convert words to lower case and split them

    text = text.lower().split()

    

    text = " ".join(text)

    ## Clean the text

    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)

    text = re.sub(r"what's", "what is ", text)

    text = re.sub(r"\'s", " ", text)

    text = re.sub(r"\'ve", " have ", text)

    text = re.sub(r"n't", " not ", text)

    text = re.sub(r"i'm", "i am ", text)

    text = re.sub(r"\'re", " are ", text)

    text = re.sub(r"\'d", " would ", text)

    text = re.sub(r"\'ll", " will ", text)

    text = re.sub(r",", " ", text)

    text = re.sub(r"\.", " ", text)

    text = re.sub(r"!", " ! ", text)

    text = re.sub(r"\/", " ", text)

    text = re.sub(r"\^", " ^ ", text)

    text = re.sub(r"\+", " + ", text)

    text = re.sub(r"\-", " - ", text)

    text = re.sub(r"\=", " = ", text)

    text = re.sub(r"'", " ", text)

    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)

    text = re.sub(r":", " : ", text)

    text = re.sub(r" e g ", " eg ", text)

    text = re.sub(r" b g ", " bg ", text)

    text = re.sub(r" u s ", " american ", text)

    text = re.sub(r"\0s", "0", text)

    text = re.sub(r" 9 11 ", "911", text)

    text = re.sub(r"e - mail", "email", text)

    text = re.sub(r"j k", "jk", text)

    text = re.sub(r"\s{2,}", " ", text)



    return text

# Process the data for model input

def get_texts_and_labels(df):

  texts = df['text'].map(lambda x: clean_text(x)).tolist()

  texts = [t.split()[0:MAXLEN] for t in texts]

  labels = df['male'].tolist()

  return texts, labels



train_text, train_label = get_texts_and_labels(df_train)

valid_text, valid_label = get_texts_and_labels(df_valid)

test_text, test_label = get_texts_and_labels(df_test)



max([len(x) for x in train_text]), max([len(x) for x in valid_text]), max([len(x) for x in test_text])
# Fit tokenizer on training data

tok = Tokenizer()

tok.fit_on_texts(train_text)

vocab_size = len(tok.word_index) + 1



# Tokenize the data

def get_tokenized_texts(texts):

  encoded_docs = tok.texts_to_sequences(texts)

  padded_docs = pad_sequences(encoded_docs, maxlen=MAXLEN, padding='post')

  return padded_docs



docs_train = get_tokenized_texts(train_text)

docs_valid = get_tokenized_texts(valid_text)

docs_test = get_tokenized_texts(test_text)



print(type(docs_train), len(docs_train), len(docs_valid), len(docs_test))

docs_train[0][:10]
# Load the whole embedding into memory

embeddings_index = dict()

with open(glovepath) as f:

    for line in f:

      values = line.split()

      word = values[0]

      coefs = np.asarray(values[1:], dtype='float32')

      embeddings_index[word] = coefs

print('Loaded %s word vectors.' % len(embeddings_index))
# Create a weight matrix for words in training docs

embedding_matrix = np.zeros((vocab_size, EMBEDDING_DIM))

for word, i in tok.word_index.items():

    embedding_vector = embeddings_index.get(word)

    if embedding_vector is not None:

        embedding_matrix[i] = embedding_vector
# NERUAL NETWORK MODEL





# PARAMETERS

batchsize = 128



lstm_dim = 80

residual_connection_width = 40



dropout_factor = 1.07

spatiotemporal_dropout = 0.25 * dropout_factor

lstm_dropout = 0.3 * dropout_factor

residual_connection_dropout = 0.6 * dropout_factor

final_dropout = 0.7 * dropout_factor



base_frozen_lr = 1e-3

base_frozen_decay = 1e-4

frozen_epochs = 35

frozen_batchsize = batchsize



base_unfrozen_lr = 1.3e-4

base_unfrozen_decay = 2.5e-4

unfrozen_epochs = 29

unfrozen_batchsize = batchsize



if DEBUG:

    frozen_batchsize = 512

    unfrozen_batchsize = 512

    frozen_epochs = 2

    unfrozen_epochs = 2



    

base_batchsize = 512



frozen_lr_factor = frozen_batchsize / base_batchsize

unfrozen_lr_factor = unfrozen_batchsize / base_batchsize



frozen_lr = base_frozen_lr * frozen_lr_factor

frozen_decay = base_frozen_decay * frozen_lr_factor



unfrozen_lr = base_unfrozen_lr * unfrozen_lr_factor

unfrozen_decay = base_unfrozen_decay * unfrozen_lr_factor





inputs = layers.Input((MAXLEN,), dtype="int32")





# EMBEDDING BLOCK

raw_embed = layers.Embedding(vocab_size, 

                           EMBEDDING_DIM, 

                           weights=[embedding_matrix], 

                           input_length=MAXLEN, 

                           trainable=False)(inputs)

embed_random_drop = layers.Dropout(rate=spatiotemporal_dropout)(raw_embed)

embed_time_drop = layers.Dropout(rate=spatiotemporal_dropout, 

                       noise_shape=(None, MAXLEN, 1))(embed_random_drop)





# LEFT LSTM BLOCK



# Backward LSTM layer

lstm_bottom_left = layers.LSTM(lstm_dim, return_sequences=True, 

                               go_backwards=True, dropout=lstm_dropout, 

                               recurrent_dropout=lstm_dropout)(embed_time_drop)

lstm_random_drop_left = layers.Dropout(rate=spatiotemporal_dropout)(lstm_bottom_left)

lstm_time_drop_left = layers.Dropout(rate=spatiotemporal_dropout, 

                            noise_shape=(None,MAXLEN,1))(lstm_random_drop_left)

# Forward LSTM layer

lstm_top_left = layers.LSTM(lstm_dim, return_sequences=False, dropout=lstm_dropout, 

                            recurrent_dropout=lstm_dropout)(lstm_time_drop_left)





# RIGHT LSTM BLOCK



# Forward LSTM layer

lstm_bottom_right = layers.LSTM(lstm_dim, return_sequences=True, dropout=lstm_dropout, 

                                recurrent_dropout=lstm_dropout)(embed_time_drop)

lstm_random_drop_right = layers.Dropout(rate=spatiotemporal_dropout)(lstm_bottom_right)

lstm_time_drop_right = layers.Dropout(rate=spatiotemporal_dropout, 

                            noise_shape=(None,MAXLEN,1))(lstm_random_drop_right)

# Backward LSTM layer

lstm_top_right = layers.LSTM(80, return_sequences=False, 

                             go_backwards=True, dropout=lstm_dropout, 

                             recurrent_dropout=lstm_dropout)(lstm_time_drop_right)





# MERGE LEFT AND RIGHT BLOCK

merged_lstm = layers.merge.concatenate([lstm_top_left, lstm_top_right])





# LEFT RESIDUAL BRANCH

dropout_resid = layers.Dropout(rate=residual_connection_dropout)(merged_lstm)

dense_resid = layers.Dense(residual_connection_width, activation='relu')(dropout_resid)



# RIGHT RESIDUAL BRANCH

dropout_resid2 = layers.Dropout(rate=residual_connection_dropout)(merged_lstm)

dense_resid2 = layers.Dense(residual_connection_width, activation='relu')(dropout_resid2)





# FINAL DENSE BLOCK

merged_resid = layers.merge.concatenate([merged_lstm, dense_resid, dense_resid2])

dropout = layers.Dropout(rate=final_dropout)(merged_resid)

pred = layers.Dense(1, activation='sigmoid')(dropout)





# FINAL MODEL

model = Model(inputs=[inputs], outputs=pred)

model.compile(optimizer=Adam(frozen_lr, decay=frozen_decay), 

              loss='binary_crossentropy', metrics=['acc'])

model.summary()
# Fit the frozen model

model.fit(docs_train, train_label, epochs=frozen_epochs, 

          validation_data=(docs_valid, valid_label), batch_size=batchsize)
[(i, l.name, l.trainable) for i,l in enumerate(model.layers)]
# Unfreeze the embeddings

model.layers[1].trainable = True

model.compile(optimizer=Adam(unfrozen_lr, decay=unfrozen_decay), 

              loss='binary_crossentropy', metrics=['acc'])

model.summary()
# Fit the unfrozen model

model.fit(docs_train, train_label, epochs=unfrozen_epochs, 

          validation_data=(docs_valid, valid_label), batch_size=unfrozen_batchsize)
y_test_pred = model.predict(docs_test)

print( confusion_matrix(test_label, (y_test_pred>.5)) )

f1_score(test_label, y_test_pred>.5)
accuracy_score(test_label, y_test_pred>.5)
fpr, tpr, _ = roc_curve(test_label, y_test_pred)

roc_auc = auc(fpr, tpr)



plt.figure()

plt.plot(fpr, tpr, color='darkorange',

         lw=1, label='ROC curve (area = %0.3f)' % roc_auc)

plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Receiver operating characteristic curve')

plt.legend(loc="lower right")

plt.show()
df_acc = pd.DataFrame(columns=['minprob','maxprob','count','accuracy'])

y_test = np.array(test_label)

y_test_pred = y_test_pred.reshape(-1)

for pbot in np.linspace(0,.9,10):

    ptop = pbot+.1

    mask = (y_test_pred>=pbot)&(y_test_pred<ptop)

    count = int(mask.sum())

    if count>0:

        actual = pd.Series(y_test)[mask].values

        pred_prob = pd.Series(y_test_pred)[mask].values

        pred_bin = pred_prob>.5

        acc = accuracy_score(actual, pred_bin)

        nsucc = sum(actual==pred_bin)

        confint = proportion_confint(nsucc, count)

        minconf = confint[0]

        maxconf = confint[1]

    else:

        acc = np.nan

        minconf = np.nan

        maxconf = np.nan

    row = pd.DataFrame({'minprob':[pbot], 'maxprob':[ptop], 'count':[count], 

                        'accuracy':[acc], 'lconf95':[minconf], 'hconf95':[maxconf]})

    df_acc = pd.concat([df_acc, row], sort=False)

df_acc.set_index(['minprob','maxprob'])
df_acc['avgprob'] = .5*(df_acc.minprob+df_acc.maxprob)

ax = df_acc.drop(['count','minprob','maxprob'],axis=1).set_index('avgprob').plot(

        title='Accuracy of Predictions by Range')

ax.legend(labels=['accuracy', '95% conf, lower', '95% conf, upper'])

ax.set(xlabel="center of probability bin", ylabel="fraction of correct predicitons")

plt.show()