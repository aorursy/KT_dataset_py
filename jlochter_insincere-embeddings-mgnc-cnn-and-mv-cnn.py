import os
import time
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from tqdm import tqdm
import math
from sklearn.model_selection import train_test_split
from sklearn import metrics

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Add, Concatenate, Dense, Input, LSTM, Embedding, Dropout, Activation, CuDNNGRU, Conv1D, Convolution2D, Reshape
from keras.layers import Bidirectional, GlobalMaxPool1D, MaxPool1D, GlobalMaxPool2D, Flatten, ZeroPadding1D
from keras.models import Model
from keras import initializers, regularizers, constraints, optimizers, layers
train_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/test.csv")
print("Train shape : ",train_df.shape)
print("Test shape : ",test_df.shape)
## split to train and val
train_df, val_df = train_test_split(train_df, test_size=0.05, random_state=2018)

## some config values 
embed_size = 300 # how big is each word vector
max_features = 50000 # how many unique words to use (i.e num rows in embedding vector)
maxlen = 60 # max number of words in a question to use

## fill up the missing values
train_X = train_df["question_text"].fillna("_na_").values
val_X = val_df["question_text"].fillna("_na_").values
test_X = test_df["question_text"].fillna("_na_").values

## Tokenize the sentences
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(train_X))
train_X = tokenizer.texts_to_sequences(train_X)
val_X = tokenizer.texts_to_sequences(val_X)
test_X = tokenizer.texts_to_sequences(test_X)

## Pad the sentences 
train_X = pad_sequences(train_X, maxlen=maxlen)
val_X = pad_sequences(val_X, maxlen=maxlen)
test_X = pad_sequences(test_X, maxlen=maxlen)

## Get the target values
train_y = train_df['target'].values
val_y = val_df['target'].values
def getGloVeEmbeddings():
    EMBEDDING_FILE = '../input/embeddings/glove.840B.300d/glove.840B.300d.txt'
    def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
    embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE))

    all_embs = np.stack(embeddings_index.values())
    emb_mean,emb_std = all_embs.mean(), all_embs.std()
    embed_size = all_embs.shape[1]

    word_index = tokenizer.word_index
    nb_words = min(max_features, len(word_index))
    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
    for word, i in word_index.items():
        if i >= max_features: continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None: embedding_matrix[i] = embedding_vector
    return embedding_matrix

def getFastTextEmbeddings():
    EMBEDDING_FILE = '../input/embeddings/wiki-news-300d-1M/wiki-news-300d-1M.vec'
    def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
    embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE) if len(o)>100)

    all_embs = np.stack(embeddings_index.values())
    emb_mean,emb_std = all_embs.mean(), all_embs.std()
    embed_size = all_embs.shape[1]

    word_index = tokenizer.word_index
    nb_words = min(max_features, len(word_index))
    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
    for word, i in word_index.items():
        if i >= max_features: continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None: embedding_matrix[i] = embedding_vector
    return embedding_matrix

def getParagramEmbeddings():
    EMBEDDING_FILE = '../input/embeddings/paragram_300_sl999/paragram_300_sl999.txt'
    def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
    embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE, encoding="utf8", errors='ignore') if len(o)>100)

    all_embs = np.stack(embeddings_index.values())
    emb_mean,emb_std = all_embs.mean(), all_embs.std()
    embed_size = all_embs.shape[1]

    word_index = tokenizer.word_index
    nb_words = min(max_features, len(word_index))
    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
    for word, i in word_index.items():
        if i >= max_features: continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None: embedding_matrix[i] = embedding_vector
    return embedding_matrix


glove_embedding = getGloVeEmbeddings()
fasttext_embedding = getFastTextEmbeddings()
paragram_embedding = getParagramEmbeddings()
from keras.engine import Layer, InputSpec
import tensorflow as tf

class KMaxPooling(Layer):
    """
    K-max pooling layer that extracts the k-highest activations from a sequence (2nd dimension).
    TensorFlow backend.
    """
    def __init__(self, k=1, axis=1, **kwargs):
        super(KMaxPooling, self).__init__(**kwargs)
        self.input_spec = InputSpec(ndim=3)
        self.k = k

        assert axis in [1,2],  'expected dimensions (samples, filters, convolved_values),\
                   cannot fold along samples dimension or axis not in list [1,2]'
        self.axis = axis

        # need to switch the axis with the last elemnet
        # to perform transpose for tok k elements since top_k works in last axis
        self.transpose_perm = [0,1,2] #default
        self.transpose_perm[self.axis] = 2
        self.transpose_perm[2] = self.axis

    def compute_output_shape(self, input_shape):
        input_shape_list = list(input_shape)
        input_shape_list[self.axis] = self.k
        return tuple(input_shape_list)

    def call(self, x):
        # swap sequence dimension to get top k elements along axis=1
        transposed_for_topk = tf.transpose(x, perm=self.transpose_perm)

        # extract top_k, returns two tensors [values, indices]
        top_k = tf.nn.top_k(transposed_for_topk, k=self.k, sorted=True, name=None)[0]

        # return back to normal dimension but now sequence dimension has only k elements
        # performing another transpose will get the tensor back to its original shape
        # but will have k as its axis_1 size
        transposed_back = tf.transpose(top_k, perm=self.transpose_perm)

        return transposed_back


class Folding(Layer):

    def __init__(self, **kwargs):
        super(Folding, self).__init__(**kwargs)
        self.input_spec = InputSpec(ndim=3)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], int(input_shape[2]/2))

    def call(self, x):
        input_shape = x.get_shape().as_list()

        # split the tensor along dimension 2 into dimension_axis_size/2
        # which will give us 2 tensors
        splits = tf.split(x, num_or_size_splits=int(input_shape[2]/2), axis=2)

        # reduce sums of the pair of rows we have split onto
        reduce_sums = [tf.reduce_sum(split, axis=2) for split in splits]

        # stack them up along the same axis we have reduced
        row_reduced = tf.stack(reduce_sums, axis=2)
        return row_reduced

def BiRNN():

    inp = Input(shape=(maxlen,))

    #raw      = Embedding(max_features, embed_size)(inp)
    glove    = Embedding(max_features, embed_size, weights=[glove_embedding])(inp)
    fast     = Embedding(max_features, embed_size, weights=[fasttext_embedding])(inp)
    paragram = Embedding(max_features, embed_size, weights=[paragram_embedding])(inp)

    x = Add()([glove, fast, paragram])

    x = Bidirectional(CuDNNGRU(64, return_sequences=True))(x)
    x = GlobalMaxPool1D()(x)
    x = Dense(16, activation="relu")(x)
    x = Dropout(0.1)(x)
    x = Dense(1, activation="sigmoid")(x)

    model = Model(inputs=inp, outputs=x)

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    print(model.summary())
    
    return model

def MGNC_CNN():

    inp = Input(shape=(maxlen,))
    
    glove    = Embedding(max_features, embed_size, weights=[glove_embedding])(inp)
    fast     = Embedding(max_features, embed_size, weights=[fasttext_embedding])(inp)
    paragram = Embedding(max_features, embed_size, weights=[paragram_embedding])(inp)    
    
    filter_sizes = [3,5]
    
    conv_pools = []
    for text_embedding in [glove,fast,paragram]:
        for filter_size in filter_sizes:
            l_zero = ZeroPadding1D((filter_size-1,filter_size-1))(text_embedding)
            l_conv = Conv1D(filters=16, kernel_size=filter_size, padding='same', activation='tanh')(l_zero)
            l_pool = GlobalMaxPool1D()(l_conv)
            conv_pools.append(l_pool)
            
    l_merge = Concatenate(axis=1)(conv_pools)
    l_dense = Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01))(l_merge)
    l_out = Dense(1, activation='sigmoid')(l_dense)
        
    model = Model(inputs=inp, outputs=l_out)
    
    model.compile(loss='binary_crossentropy',
                    optimizer='adam',
                    metrics=['acc'])
    
    print(model.summary())
    
    return model

def MV_CNN():
    
    inp = Input(shape=(maxlen,))
    
    glove    = Embedding(max_features, embed_size, weights=[glove_embedding])(inp)
    fast     = Embedding(max_features, embed_size, weights=[fasttext_embedding])(inp)
    paragram = Embedding(max_features, embed_size, weights=[paragram_embedding])(inp)    
    
    k_top = 4
    filter_sizes = [3,5]

    layer_1 = []
    for text_embedding in [glove, fast, paragram]:
        conv_pools = []
        for filter_size in filter_sizes:
            l_zero = ZeroPadding1D((filter_size-1,filter_size-1))(text_embedding)
            l_conv = Conv1D(filters=128, kernel_size=filter_size, padding='same', activation='tanh')(l_zero)
            l_pool = KMaxPooling(k=30, axis=1)(l_conv)
            conv_pools.append((filter_size,l_pool))
        layer_1.append(conv_pools)
            
            
    last_layer = []
    for layer in layer_1: # no of embeddings used
        for (filter_size, input_feature_maps) in layer:
            l_zero = ZeroPadding1D((filter_size-1,filter_size-1))(input_feature_maps)
            l_conv = Conv1D(filters=128, kernel_size=filter_size, padding='same', activation='tanh')(l_zero)
            l_pool = KMaxPooling(k=k_top, axis=1)(l_conv)
            last_layer.append(l_pool)

    l_merge = Concatenate(axis=1)(last_layer)
    l_flat = Flatten()(l_merge)
    l_dense = Dense(128, activation='relu')(l_flat)
    l_out = Dense(1, activation='sigmoid')(l_dense)
        
    model = Model(inputs=inp, outputs=l_out)
    
    model.compile(loss='binary_crossentropy',
                    optimizer='adam',
                    metrics=['acc'])
    
    print(model.summary())
    
    return model
#mgnccnn = MGNC_CNN()
#mgnccnn.fit(train_X, train_y, batch_size=512, epochs=4, validation_data=(val_X, val_y))
#pred_y = mgnccnn.predict([val_X], batch_size=1024, verbose=1)

#max_f1 = 0
#max_thresh = 0

#for thresh in np.arange(0.1, 0.501, 0.01):
#    thresh = np.round(thresh, 2)
#    score = metrics.f1_score(val_y, (pred_y > thresh).astype(int))
#    print("F1 score at threshold {0} is {1}".format(thresh, score))
#    if score > max_f1:
#        max_f1 = score
#        max_thresh = thresh

#print("** Best F1 score at threshold {0} is {1}".format(max_thresh, max_f1))

#pred_mgnccnn = pred_y
mvcnn = MV_CNN()
mvcnn.fit(train_X, train_y, batch_size=1024, epochs=2, validation_data=(val_X, val_y))
pred_y = mvcnn.predict([val_X], batch_size=1024, verbose=1)

max_f1 = 0
max_thresh = 0

for thresh in np.arange(0.1, 0.501, 0.01):
    thresh = np.round(thresh, 2)
    score = metrics.f1_score(val_y, (pred_y > thresh).astype(int))
    print("F1 score at threshold {0} is {1}".format(thresh, score))
    if score > max_f1:
        max_f1 = score
        max_thresh = thresh

print("** Best F1 score at threshold {0} is {1}".format(max_thresh, max_f1))

pred_mgnccnn = pred_y
pred_y = mvcnn.predict([test_X], batch_size=1024, verbose=1)
pred_test_y = (pred_y > max_thresh).astype(int)
out_df = pd.DataFrame({"qid": test_df["qid"].values})
out_df['prediction'] = pred_test_y
out_df.to_csv("submission.csv", index=False)