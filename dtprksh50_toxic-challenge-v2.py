# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

import sys, os, re, csv, codecs

import matplotlib.pyplot as plt

%matplotlib inline

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.layers import Input, Dense, Activation, Dropout, CuDNNLSTM, GlobalMaxPool1D,CuDNNGRU

from keras.layers import Embedding, Bidirectional, Concatenate, SpatialDropout1D

from keras.models import Model

from keras import regularizers, initializers, constraints, layers, optimizers

import keras

# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/train.csv')

test = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/test.csv')
train['comment_text'].isnull().any()
train.head()
y = np.where(train['target'].values >= 0.5, 1, 0)
def preprocess(data):

    '''

    Credit goes to https://www.kaggle.com/gpreda/jigsaw-fast-compact-solution

    '''

    punct = "/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~`" + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&'

    def clean_special_chars(text, punct):

        for p in punct:

            text = text.replace(p, ' ')

        return text



    data = data.astype(str).apply(lambda x: clean_special_chars(x, punct))

    return data

train_list = preprocess(train['comment_text'])
tokenizer = Tokenizer()

tokenizer.fit_on_texts(list(train_list))

train_tokenize_list = tokenizer.texts_to_sequences(list(train_list))

#test_tokenize_list = tokenizer.texts_to_sequences(list(test_list))
sen_length = [len(i) for i in train_tokenize_list]

plt.hist(sen_length, bins=np.arange(0, 200, 10))

plt.show()
max_len = 190

train_pad_list = pad_sequences(train_tokenize_list, maxlen= max_len)

#test_pad_list = pad_sequences(test_tokenize_list, maxlen= max_len)
def get_coefs(word, *arr):

    return word, np.asarray(arr, dtype='float32')





def load_embeddings(path):

    with open(path) as f:

        return dict(get_coefs(*line.strip().split(' ')) for line in f)





def build_matrix(word_index, path):

    embedding_index = load_embeddings(path)

    embedding_matrix = np.zeros((len(word_index) + 1, 300))

    for word, i in word_index.items():

        try:

            embedding_matrix[i] = embedding_index[word]

        except KeyError:

            pass

    return embedding_matrix

    
EMBEDDING_FILES = [

    '../input/fasttext-crawl-300d-2m/crawl-300d-2M.vec'

]

embedding_matrix = np.concatenate(

    [build_matrix(tokenizer.word_index, f) for f in EMBEDDING_FILES], axis=-1)
#embedding_index = load_embeddings('../input/fasttext-crawl-300d-2m/crawl-300d-2M.vec')
#type(embedding_index)
# import operator 



# def check_coverage(vocab,embeddings_index):

#     a = {}

#     oov = {}

#     k = 0

#     i = 0

#     for word in vocab:

#         try:

#             a[word] = embeddings_index[word]

#             k += vocab[word]

#         except:



#             oov[word] = vocab[word]

#             i += vocab[word]

#             pass



#     print('Found embeddings for {:.2%} of vocab'.format(len(a) / len(vocab)))

#     print('Found embeddings for  {:.2%} of all text'.format(k / (k + i)))

#     sorted_x = sorted(oov.items(), key=operator.itemgetter(1))[::-1]



#     return sorted_x
#import tqdm

#oov = check_coverage(tokenizer.word_docs,embedding_index)
# inp = Input(shape=(max_len,))

# embed_size = 300

# e = Embedding(*embedding_matrix.shape, weights = [embedding_matrix], trainable = False)(inp)

# x = CuDNNLSTM(128, return_sequences= True, name = 'lstm_layer')(e)

# x = CuDNNLSTM(128, return_sequences= True, name = 'lstm_layer2')(x)

# x = GlobalMaxPool1D()(x)

# x = Dropout(0.1)(x)

# x = Dense(64, activation='relu')(x)

# x = Dropout(0.1)(x)

# x = Dense(1, activation='sigmoid')(x)
inp = Input(shape=(max_len,))

x = Embedding(*embedding_matrix.shape, weights=[embedding_matrix], trainable=False)(inp)

x = SpatialDropout1D(0.3)(x)

x1 = Bidirectional(CuDNNLSTM(256, return_sequences=True))(x)

x2 = Bidirectional(CuDNNGRU(128, return_sequences=True))(x1)

max_pool1 = GlobalMaxPool1D()(x1)

max_pool2 = GlobalMaxPool1D()(x2)

conc = Concatenate()([max_pool1, max_pool2])

predictions = Dense(1, activation='sigmoid')(conc)

model = Model(inputs=inp, outputs=predictions)
#model = Model(inputs=inp, outputs=x)

model.compile(optimizer='adam', loss = 'binary_crossentropy', metrics=['accuracy'])
filepath='weights.{epoch:02d}-{val_loss:.2f}.hdf5'

mcCallBack = keras.callbacks.ModelCheckpoint(filepath,save_best_only=True,mode='auto')
batch_size = 128

epochs = 3

model.fit(train_pad_list, y, epochs= epochs, batch_size=batch_size, validation_split=0.1, callbacks= [mcCallBack])
test_list = test['comment_text']

test_tokenize_list = tokenizer.texts_to_sequences(list(test_list))

test_pad_list = pad_sequences(test_tokenize_list, maxlen= max_len)
y_test = model.predict(test_pad_list, verbose=1)
sample_sub = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/sample_submission.csv')

sample_sub.head()
sample_sub['prediction'] = y_test

#sample_sub.to_csv('../input/submission.csv', index = False)
sample_sub.to_csv('submission.csv', index = False)
# import os

# import sys

# import requests

# from tqdm import tqdm



# # if len(sys.argv) != 2:

# #     print('You must enter the model name as a parameter, e.g.: download_model.py 117M')

# #     sys.exit(1)



# model = '117M'



# subdir = os.path.join('models', model)

# if not os.path.exists(subdir):

#     os.makedirs(subdir)

# subdir = subdir.replace('\\','/') # needed for Windows



# for filename in ['checkpoint','encoder.json','hparams.json','model.ckpt.data-00000-of-00001', 'model.ckpt.index', 'model.ckpt.meta', 'vocab.bpe']:



#     r = requests.get("https://storage.googleapis.com/gpt-2/" + subdir + "/" + filename, stream=True)



#     with open(os.path.join(subdir, filename), 'wb') as f:

#         file_size = int(r.headers["content-length"])

#         chunk_size = 1000

#         with tqdm(ncols=100, desc="Fetching " + filename, total=file_size, unit_scale=True) as pbar:

#             # 1k for chunk_size, since Ethernet packet size is around 1500 bytes

#             for chunk in r.iter_content(chunk_size=chunk_size):

#                 f.write(chunk)

#                 pbar.update(chunk_size)
# import os

# os.makedirs('models')