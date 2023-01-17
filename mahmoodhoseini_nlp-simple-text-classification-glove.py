import numpy as np, pandas as pd

import matplotlib.pylab as plt

import tensorflow as tf

from keras.layers import Input, Dense, LSTM, Embedding, Dropout, Activation

from keras.models import Model

from keras import initializers, optimizers

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.callbacks import ModelCheckpoint

import seaborn, re

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sklearn.model_selection import train_test_split



import warnings

warnings.filterwarnings('ignore')

%matplotlib inline

print('tf version ' + tf.version.VERSION)

print('GPU is avaiable' if tf.test.is_gpu_available() else 'GPU is NOT avaialable')
train_df = pd.read_csv('../input/nlp-getting-started/train.csv')

train_df.info()
train_df.head()
plt.style.use('ggplot')

fig, axes = plt.subplots(1, 3, figsize=(11,3))



train_df.text.str.len().groupby(train_df.target).mean().plot(kind='bar', color='c', ax = axes[0])

axes[0].set_title('Avg txt length')



seaborn.distplot(train_df[train_df.target==0].text.str.len(), ax=axes[1], color='b', label='fake')

axes[1].legend()



seaborn.distplot(train_df[train_df.target==1].text.str.len(), ax=axes[2], color='r', label='real')

axes[2].legend()
def get_ngrams(txts, ngram_range=(2,2)):

    vec = CountVectorizer(ngram_range=ngram_range).fit(txts)

    BoW = vec.transform(txts)

    sum_words = BoW.sum(axis=0) 

    wfreq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]

    wfreq = sorted(wfreq, key = lambda x: x[1], reverse=True)

    return wfreq



train_ngrams = get_ngrams(train_df.text)

print(train_ngrams[:20])
def remove_url(txt):

    url = re.compile(r'https?://\S+|www\.\S+')

    return url.sub(r'',txt)



train_df.text = train_df.text.apply(lambda x: remove_url(x))
train_ngrams = get_ngrams(train_df.text)



plt.style.use('ggplot')

fig, ax = plt.subplots(1, 1, figsize=(9,15))



word, freq = map(list, zip(*train_ngrams[:80]))

seaborn.barplot(x=freq, y=word);
X_train, X_valid, y_train, y_valid = train_test_split(train_df.text.tolist(),

                                                      train_df.target.tolist(),

                                                      test_size=0.2)



## Bag of Words

count_vec = CountVectorizer()

cnt_train = count_vec.fit_transform(X_train)

cnt_valid = count_vec.fit_transform(X_valid)
## TF-IDF

tfidf_vec = TfidfVectorizer()

tfidf_train = tfidf_vec.fit_transform(X_train)

tfidf_valid = tfidf_vec.fit_transform(X_valid)
## https://www.kaggle.com/vbmokin/nlp-eda-bag-of-words-tf-idf-glove-bert

from sklearn.decomposition import TruncatedSVD

import matplotlib as mpl



def plot_LSA(data, labels, ax):

    lsa = TruncatedSVD(n_components=2)

    lsa.fit(data)

    lsa_scores = lsa.transform(data)

    color_mapper = {label:idx for idx,label in enumerate(set(labels))}

    color_column = [color_mapper[label] for label in labels]

    ax.scatter(lsa_scores[:,0], lsa_scores[:,1], s=20, alpha=0.5, c=labels)



fig, axes = plt.subplots(1, 2, figsize=(9, 4))          

plot_LSA(cnt_train, y_train, axes[0])

axes[0].set_title('BoW')

plot_LSA(tfidf_train, y_train, axes[1])

axes[1].set_title('TF-IDF');
from nltk import tokenize



docs = []

for txt in train_df.text :

    docs.append([w for w in tokenize.word_tokenize(txt.lower())])
def convert_to_one_hot (y, C) :

    Y = np.eye(C)[y.reshape(-1)]

    return Y



y_train_oh = convert_to_one_hot(np.array(y_train), C=2)

y_valid_oh = convert_to_one_hot(np.array(y_valid), C=2)
def read_glove_vecs (gfile) :

    with open(gfile, 'r') as f :

        words = set()

        word2vec_map = {}

        for line in f :

            line = line.strip().split()

            curr_word = line[0]

            words.add(curr_word)

            word2vec_map[curr_word] = np.array(line[1:], dtype=np.float32)



        ii = 1

        word2idx = {}

        idx2word = {}

        for w in words :

            word2idx[w] = ii

            idx2word[ii] = w

            ii = ii+1



    return word2vec_map, word2idx, idx2word



word2vec_map, word2idx, idx2word = read_glove_vecs('../input/glove-twitter/glove.twitter.27B.100d.txt')
for w in word2vec_map.keys() :

    if len(word2vec_map[w]) ==  len(word2vec_map['love']) - 1:

        word2vec_map[w] = np.pad(word2vec_map[w], [(0,1)], mode='constant') 

print(word2vec_map['love'])
def sente2indices (X, word2idx, maxLen=50) :

    m = np.shape(X)[0]

    ids = np.zeros((m, maxLen))

    for ii in range(m) :

        words = X[ii].lower().split()

        for idx, w in enumerate(words) :

            if w in word2idx.keys() :

                ids[ii, idx] = word2idx[w]



    return ids
def create_embedding_layer (word2vec_map, word2idx) :

    vocab_len = len(word2idx) + 1

    emb_dim = np.shape(word2vec_map['love'])[0]

    emb_mat = np.zeros((vocab_len, emb_dim))

    for w, ii in word2idx.items() :

        emb_mat[ii, :] = word2vec_map[w]



    embedding_layer = Embedding(vocab_len, emb_dim, trainable=False)

    embedding_layer.build((None,))

    embedding_layer.set_weights([emb_mat])

    return embedding_layer
def txt_cls (input_shape, word2vec_map, word2idx) :

    X_input = Input(input_shape)

    embedding_layer = create_embedding_layer(word2vec_map, word2idx)

    X = embedding_layer(X_input)

    X = LSTM(units=32, activation='tanh', use_bias=True, bias_initializer='zeros',

        kernel_initializer='glorot_uniform', return_sequences=True)(X)

    X = Dropout(0.2)(X)

    X = LSTM(units=32, activation='tanh', use_bias=True, bias_initializer='zeros',

        kernel_initializer='glorot_uniform', return_sequences=False)(X)

    X = Dropout(0.2)(X)

    X = Dense(2, activation=None, use_bias=True, bias_initializer='zeros',

        kernel_initializer='glorot_uniform')(X)

    X = Activation('sigmoid')(X)



    model = Model(inputs=X_input, outputs=X)



    return model
## 1

maxLen = 50

clsModel = txt_cls((maxLen,), word2vec_map, word2idx)

clsModel.summary()
## 2

optim = optimizers.Adam(lr=0.0005)

clsModel.compile(loss='categorical_crossentropy', optimizer=optim, metrics=['accuracy'])
## 3

cp_fpath = "clsModel.hdf5"

mdlcp_cb = ModelCheckpoint(cp_fpath, monitor='val_accuracy', mode='max', save_best_only=True,

                          save_weights_only=True, verbose=1)



X_train_ids = sente2indices(X_train, word2idx, maxLen)

X_valid_ids = sente2indices(X_valid, word2idx, maxLen)

clsRes = clsModel.fit(x=X_train_ids, 

                      y=y_train_oh,

                      validation_data=(X_valid_ids, y_valid_oh),

                      epochs=100, 

                      batch_size=512, 

                      shuffle=True,

                      callbacks=[mdlcp_cb],

                      verbose=1)
plt.style.use('dark_background')

fig, axes = plt.subplots(1, 2, figsize=(9,4))



axes[0].plot(clsRes.history['loss'], 'w', label='train')

axes[0].plot(clsRes.history['val_loss'], 'orange', label='valid')

axes[0].set_ylabel('loss')



axes[1].plot(clsRes.history['accuracy'], 'w', label='train')

axes[1].plot(clsRes.history['val_accuracy'], 'orange', label='valid')

axes[1].set_ylabel('accuracy')
clsModel_json = clsModel.to_json()

with open('clsModel.json', 'w') as myfile:

     myfile.write(clsModel_json)
clsModel = txt_cls((maxLen,), word2vec_map, word2idx)

clsModel.compile(loss='categorical_crossentropy', optimizer=optim, metrics=['accuracy'])

clsModel.load_weights('/kaggle/working/clsModel.hdf5')
# print mislabelled examples

pred = clsModel.predict(X_valid_ids)

for ii in range(50):

    x = X_valid_ids

    num = np.argmax(pred[ii])

    if(num != y_valid[ii]):

        print('pred: ' + str(num) + 

              ', true: ' + str(y_valid[ii]) + 

               ' --> ', X_valid[ii] + '\n')
# print correctly labelled examples

pred = clsModel.predict(X_valid_ids)

for ii in range(20):

    x = X_valid_ids

    num = np.argmax(pred[ii])

    if(num == y_valid[ii]):

        print('pred: ' + str(num) + 

              ', true: ' + str(y_valid[ii]) + 

               ' --> ', X_valid[ii] + '\n')
## 4

test_df = pd.read_csv('../input/nlp-getting-started/test.csv')

test_df.text = test_df.text.apply(lambda x: remove_url(x))



X_test_ids = sente2indices(test_df.text.tolist(), word2idx, maxLen)

y_test_oh = clsModel.predict(X_test_ids)

y_test = [np.argmax(x) for x in y_test_oh]

test_df['target'] = y_test

test_df.head(10)
sub_df = pd.read_csv('../input/nlp-getting-started/sample_submission.csv')

sub_df.target = y_test

sub_df.head()

sub_df.to_csv("sub-glove.csv", index=False, header=True)