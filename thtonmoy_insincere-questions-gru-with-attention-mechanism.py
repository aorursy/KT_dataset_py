#imports

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from sklearn.model_selection import train_test_split

from sklearn import metrics



from keras.models import Model

from keras.layers import Input, Embedding, Bidirectional, Dense, CuDNNLSTM, CuDNNGRU, Dropout, SpatialDropout1D, Concatenate, BatchNormalization

from keras.preprocessing.sequence import pad_sequences

from keras.preprocessing.text import Tokenizer

from keras import backend as K

from keras.engine.topology import Layer

from keras import initializers, regularizers, constraints, optimizers



import os

print(os.listdir("../input"))
def load_data():

    train_df = pd.read_csv("../input/train.csv")

    test_df = pd.read_csv("../input/test.csv")

    print("Train shape : ",train_df.shape)

    print("Test shape : ",test_df.shape)

    return train_df, test_df
train_df, test_df = load_data()

train_df.sample()
!ls ../input/embeddings/
def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
EMBEDDING_FILE = '../input/embeddings/glove.840B.300d/glove.840B.300d.txt'



embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE))

print('Found %s word vectors.' % len(embeddings_index))
from collections import Counter



def check_coverage(vocab,embeddings_index):

    a, oov, k, i = {}, {}, 0, 0

    for word in vocab:

        try:

            a[word] = embeddings_index[word]

            k += vocab[word]

        except:

            oov[word] = vocab[word]

            i += vocab[word]

            pass



    print(f'Found embeddings for {(len(a) / len(vocab)):.2%} of vocab')

    print(f'Found embeddings for  {(k / (k + i)):.2%} of all text')

    sorted_x = sorted(oov.items(), key=(lambda x: x[1]), reverse=True)



    return sorted_x



def get_vocab(question_series):

    sentences = question_series.str.split().values #get a list of lists of words

    words = [item for sublist in sentences for item in sublist] # flatten list into just words

    return dict(Counter(words)) # count words
vocab = get_vocab(train_df["question_text"])

out_of_vocab = check_coverage(vocab, embeddings_index)

out_of_vocab[:10]
punct = set('?!.,"#$%\'()*+-/:;<=>@[\\]^_`{|}~°√' + '“”’')

embed_punct = punct & set(embeddings_index.keys())



def clean_punctuation(txt):

    for p in "/-":

        txt = txt.replace(p, ' ')

    for p in "'`‘":

        txt = txt.replace(p, '')

    for p in punct:

        txt = txt.replace(p, f' {p} ' if p in embed_punct else ' _punct_ ') 

        #known punctuation gets space padded, otherwise we use a newn token

    return txt
train_df["question_text"] = train_df["question_text"].map(lambda x: clean_punctuation(x)).str.replace('\d+', ' # ')

test_df["question_text"] = test_df["question_text"].map(lambda x: clean_punctuation(x)).str.replace('\d+', ' # ')

vocab = get_vocab(train_df["question_text"])

out_of_vocab = check_coverage(vocab, embeddings_index)
len(vocab) - len(out_of_vocab)
maxlen = 65 # max number of words in a question to use

max_features = 60000 # how many unique words to use (i.e num rows in embedding vector)



## split to train and val

train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=201901)



# fill up the missing values

train_X = train_df["question_text"].fillna("_##_").values

val_X = val_df["question_text"].fillna("_##_").values

test_X = test_df["question_text"].fillna("_##_").values



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
def prepare_embedding_matrix(embeddings_index,word_index,num_words):

    all_embs = np.stack(embeddings_index.values())

    emb_mean,emb_std = all_embs.mean(), all_embs.std()

    embed_size = all_embs.shape[1]



    embedding_matrix = np.random.normal(emb_mean, emb_std, (num_words, embed_size))

    #embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))

    for word, i in word_index.items():

        if i >= max_features:

            continue

        embedding_vector = embeddings_index.get(word)

        if embedding_vector is not None:

            embedding_matrix[i] = embedding_vector

    return embedding_matrix
EMBEDDING_DIM = 300 # how big is each word vector

word_index = tokenizer.word_index

num_words = min(max_features, len(word_index) + 1)

embedding_matrix = prepare_embedding_matrix(embeddings_index,word_index,num_words)
class Attention(Layer):

    def __init__(self, step_dim,

                 W_regularizer=None, b_regularizer=None,

                 W_constraint=None, b_constraint=None,

                 bias=True, **kwargs):

        self.supports_masking = True

        self.init = initializers.get('glorot_uniform')



        self.W_regularizer = regularizers.get(W_regularizer)

        self.b_regularizer = regularizers.get(b_regularizer)



        self.W_constraint = constraints.get(W_constraint)

        self.b_constraint = constraints.get(b_constraint)



        self.bias = bias

        self.step_dim = step_dim

        self.features_dim = 0

        super(Attention, self).__init__(**kwargs)



    def build(self, input_shape):

        assert len(input_shape) == 3



        self.W = self.add_weight((input_shape[-1],),

                                 initializer=self.init,

                                 name='{}_W'.format(self.name),

                                 regularizer=self.W_regularizer,

                                 constraint=self.W_constraint)

        self.features_dim = input_shape[-1]



        if self.bias:

            self.b = self.add_weight((input_shape[1],),

                                     initializer='zero',

                                     name='{}_b'.format(self.name),

                                     regularizer=self.b_regularizer,

                                     constraint=self.b_constraint)

        else:

            self.b = None



        self.built = True



    def compute_mask(self, input, input_mask=None):

        return None



    def call(self, x, mask=None):

        features_dim = self.features_dim

        step_dim = self.step_dim



        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)),

                        K.reshape(self.W, (features_dim, 1))), (-1, step_dim))



        if self.bias:

            eij += self.b



        eij = K.tanh(eij)



        a = K.exp(eij)



        if mask is not None:

            a *= K.cast(mask, K.floatx())



        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())



        a = K.expand_dims(a)

        weighted_input = x * a

        return K.sum(weighted_input, axis=1)



    def compute_output_shape(self, input_shape):

        return input_shape[0],  self.features_dim
inp = Input(shape=(maxlen,))

x = Embedding(max_features, EMBEDDING_DIM, weights=[embedding_matrix],trainable=False)(inp)

x = SpatialDropout1D(0.25)(x)

x1 = Bidirectional(CuDNNLSTM(128, return_sequences=True))(x)

x2 = Bidirectional(CuDNNGRU(128, return_sequences=True))(x)

attn_lstm = Attention(maxlen)(x1)

attn_gru = Attention(maxlen)(x2)

concat = Concatenate()([attn_lstm,attn_gru])

concat = BatchNormalization()(concat)

d = Dense(256, activation="relu")(concat)

d = Dropout(0.3)(d)

out = Dense(1, activation="sigmoid")(d)

model = Model(inputs=inp, outputs=out)

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())
model.fit(train_X, train_y, batch_size=512, epochs=4, validation_data=(val_X, val_y))
pred_glove_val_y = model.predict([val_X], batch_size=1024, verbose=1)
from sklearn.metrics import roc_curve, precision_recall_curve

def threshold_search(y_true, y_proba, plot=False):

    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)

    thresholds = np.append(thresholds, 1.001) 

    F = 2 / (1/precision + 1/recall)

    best_score = np.max(F)

    best_th = thresholds[np.argmax(F)]

    if plot:

        plt.plot(thresholds, F, '-b')

        plt.plot([best_th], [best_score], '*r')

        plt.show()

    search_result = {'threshold': best_th , 'f1': best_score}

    return search_result 
result = threshold_search(val_y, pred_glove_val_y)

print(result)
from keras.models import load_model

model.save('my_model.h5')  # creates a HDF5 file 'my_model.h5'

#del model  # deletes the existing model



# returns a compiled model

# identical to the previous one

#model = load_model('/out/my_model.h5')
pred_glove_test_y = model.predict([test_X], batch_size=1024, verbose=1)


pred_test_y = pred_glove_test_y

pred_test_y = (pred_test_y > result['threshold']).astype(int)

out_df = pd.DataFrame({"qid":test_df["qid"].values})

out_df['prediction'] = pred_test_y

out_df.to_csv("submission.csv", index=False)
out_df.describe()