import csv

import re

import string



import operator

import os

import functools

import operator

import fastText



import numpy as np

import pandas as pd



from tqdm import tqdm



from keras.models import Model, Sequential

from keras.layers import Dense, Embedding, Input, InputSpec, GlobalMaxPool1D, GlobalAvgPool1D, Masking

from keras.layers import LSTM, GRU, Bidirectional, Dropout, SpatialDropout1D, BatchNormalization

from keras.layers import concatenate

from keras.preprocessing import text, sequence

from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler, Callback, ReduceLROnPlateau

from keras.optimizers import Adam, Nadam

from keras import initializers, regularizers, constraints

from tqdm import tqdm

from collections import Counter





import keras.backend as K

import tensorflow as tf



from sklearn.base import BaseEstimator, ClassifierMixin

from sklearn.utils.validation import check_X_y, check_is_fitted

from sklearn.linear_model import LogisticRegression

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sklearn.metrics import roc_auc_score

from sklearn.model_selection import train_test_split

from scipy import sparse



from nltk.tokenize import TweetTokenizer

from nltk.stem.wordnet import WordNetLemmatizer

from nltk.corpus import stopwords



from textacy.preprocess import preprocess_text



from keras.engine.topology import Layer

from keras import initializers, regularizers, constraints
for i in sorted(os.scandir('../input/fasttext-pretrained-word-vectors-english'), key=lambda x: x.stat().st_size, reverse=True):

    print(i.path)
max_features = 60000

maxlen = 250

embed_size = 300



file_path = "weights_base.best.hdf5"

emb_file = '../input/fasttext-pretrained-word-vectors-english/wiki.en.bin'

unused = set([i.strip() for i in open('../input/unused-words/unused.txt') if i.strip()])



tweet_tokenizer = TweetTokenizer(reduce_len=True)

lem = WordNetLemmatizer()

eng_stopwords = set(stopwords.words("english"))



list_classes = [

    "toxic",

    "severe_toxic",

    "obscene",

    "threat",

    "insult",

    "identity_hate",

]
CONTEXT_DIM = 100



class Attention(Layer):



    def __init__(self, regularizer=regularizers.l2(1e-10), **kwargs):

        self.regularizer = regularizer

        self.supports_masking = True

        super(Attention, self).__init__(**kwargs)



    def build(self, input_shape):

        assert len(input_shape) == 3        

        self.W = self.add_weight(name='W',

                                 shape=(input_shape[-1], CONTEXT_DIM),

                                 initializer='normal',

                                 trainable=True, 

                                 regularizer=self.regularizer)

        self.b = self.add_weight(name='b',

                                 shape=(CONTEXT_DIM,),

                                 initializer='normal',

                                 trainable=True, 

                                 regularizer=self.regularizer)

        self.u = self.add_weight(name='u',

                                 shape=(CONTEXT_DIM,),

                                 initializer='normal',

                                 trainable=True, 

                                 regularizer=self.regularizer)        

        super(Attention, self).build(input_shape)



    @staticmethod

    def softmax(x, dim):

        """Computes softmax along a specified dim. Keras currently lacks this feature.

        """

        if K.backend() == 'tensorflow':

            import tensorflow as tf

            return tf.nn.softmax(x, dim)

        elif K.backend() == 'theano':

            # Theano cannot softmax along an arbitrary dim.

            # So, we will shuffle `dim` to -1 and un-shuffle after softmax.

            perm = np.arange(K.ndim(x))

            perm[dim], perm[-1] = perm[-1], perm[dim]

            x_perm = K.permute_dimensions(x, perm)

            output = K.softmax(x_perm)



            # Permute back

            perm[dim], perm[-1] = perm[-1], perm[dim]

            output = K.permute_dimensions(x, output)

            return output

        else:

            raise ValueError("Backend '{}' not supported".format(K.backend()))



    def call(self, x, mask=None):

        ut = K.tanh(K.bias_add(K.dot(x, self.W), self.b)) * self.u



        # Collapse `attention_dims` to 1. This indicates the weight for each time_step.

        ut = K.sum(ut, axis=-1, keepdims=True)



        # Convert those weights into a distribution but along time axis.

        # i.e., sum of alphas along `time_steps` axis should be 1.

        self.at = self.softmax(ut, dim=1)

        if mask is not None:

            self.at *= K.cast(K.expand_dims(mask, -1), K.floatx())



        # Weighted sum along `time_steps` axis.

        return K.sum(x * self.at, axis=-2)



    def compute_output_shape(self, input_shape):

        return (input_shape[0], input_shape[-1])

    

    def get_config(self):

        config = {}

        base_config = super(Attention, self).get_config()

        return dict(list(base_config.items()) + list(config.items()))



    def compute_mask(self, inputs, mask):

        return None

def create_embedding(emb_file, word_index):

    if emb_file.endswith('bin'):

        embeddings_index = fastText.load_model(emb_file)

    else:

        embeddings_index = pd.read_table(emb_file,

                                         sep=" ",

                                         index_col=0,

                                         header=None,

                                         quoting=csv.QUOTE_NONE,

                                         usecols=range(embed_size + 1),

                                         dtype={h: np.float32 for h in range(1, embed_size + 1)},

                                         engine='c',

        )



    nb_words = min(max_features, len(word_index))



    # Initialize Random Matrix

    if emb_file.endswith('bin'):

        mean, std = 0.007565171, 0.29283202

    else:

        mean, std = embeddings_index.values.mean(), embeddings_index.values.std()



    embedding_matrix = np.random.normal(mean, std, (nb_words, embed_size))



    with tqdm(total=nb_words, desc='Embeddings', unit=' words') as pbar:

        for word, i in word_index.items():

            if i >= nb_words:

                continue

            if emb_file.endswith('bin'):

                if embeddings_index.get_word_id(word) != -1:

                    embedding_matrix[i] = embeddings_index.get_word_vector(word).astype(np.float32)

                    pbar.update()

            else:

                if word in embeddings_index.index:

                    embedding_matrix[i] = embeddings_index.loc[word].values

                    pbar.update()



    return embedding_matrix



def get_embedding(emb_file):

    return Embedding(min(max_features, len(tokenizer.word_index)), embed_size,

                     weights=[create_embedding(emb_file, tokenizer.word_index)],

                     input_length=maxlen,

                     trainable=False

    )



def tokenize(s):

    return re.sub('([{}“”¨«»®´·º½¾¿¡§£₤‘’])'.format(string.punctuation), r' \1 ', s).split()



def replace_numbers(s):

    dictionary = {

        '&': ' and ',

        '@': ' at ',

        '0': ' zero ',

        '1': ' one ',

        '2': ' two ',

        '3': ' three ',

        '4': ' four ',

        '5': ' five ',

        '6': ' six ',

        '7': ' seven ',

        '8': ' eight ',

        '9': ' nine ',

    }

    for k, v in dictionary.items():

        s = s.replace(k, v)

    return s



def text_cleanup(s, remove_unused=True):

    """

    This function receives ss and returns clean word-list

    """

    # Remove leaky elements like ip, user, numbers, newlines

    s = re.sub("\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", "_ip_", s)

    s = re.sub("\[\[.*\]", "", s)

    s = re.sub('\n', ' ', s)

    s = replace_numbers(s)



    # Split the sentences into words

    s = tweet_tokenizer.tokenize(s)



    # Lemmatize

    s = [lem.lemmatize(word, "v") for word in s]



    # Remove Stopwords

    s = ' '.join([w for w in s if not w in eng_stopwords])

    

    s = preprocess_text(s, fix_unicode=True,

                           lowercase=True,

                           no_currency_symbols=True,

                           transliterate=True,

                           no_urls=True,

                           no_emails=True,

                           no_contractions=True,

                           no_phone_numbers=True,

                           no_punct=True).strip()

    

    if remove_unused:

        s = ' '.join([i for i in s.split() if i not in unused])

    return s

train = pd.read_csv('../input/jigsaw-toxic-comment-classification-challenge/train.csv').sample(frac=1)

test  = pd.read_csv("../input/jigsaw-toxic-comment-classification-challenge/test.csv")



train['comment_text'] = train.comment_text.fillna("_na_").apply(text_cleanup)

test['comment_text']  = test.comment_text.fillna("_na_").apply(text_cleanup)



list_sentences_train = train.comment_text.tolist()

list_sentences_test  = test.comment_text.tolist()



y = train[list_classes].values



tokenizer = text.Tokenizer(num_words=max_features, lower=True)

tokenizer.fit_on_texts(list_sentences_train + list_sentences_test)



X_t  = sequence.pad_sequences(tokenizer.texts_to_sequences(list_sentences_train), maxlen=maxlen)

X_te = sequence.pad_sequences(tokenizer.texts_to_sequences(list_sentences_test),  maxlen=maxlen)



X_train, X_val, y_train, y_val = train_test_split(X_t, y, test_size=0.1, random_state=1337)
embedding = get_embedding(emb_file)
class RocAucEvaluation(Callback):



    def __init__(self, verbose=True):

        super(RocAucEvaluation, self).__init__()

        self.verbose = verbose



    def on_epoch_end(self, epoch, logs=None):

        logs   = logs or {}

        x_val  = self.validation_data[0]

        y_val  = self.validation_data[1]

        y_pred = self.model.predict(x_val, verbose=0)

        try:

            current  = roc_auc_score(y_val, y_pred)

        except ValueError:

            # Bug in AUC metric when TP = 100%

            # https://github.com/scikit-learn/scikit-learn/issues/1257

            current = 1.0



        logs['roc_auc'] = current



        if self.verbose:

            print("val_roc_auc: {:.6f}".format(current))



def create_model(embedding=None):

    inp = Input(shape=(maxlen,))



    x = embedding(inp)

    x = Bidirectional(GRU(64, return_sequences=True))(x)

    x = Attention()(x)

    x = Dense(6, activation="sigmoid")(x)



    model = Model(inputs=inp, outputs=x)



    model.compile(loss='binary_crossentropy', optimizer=Adam(lr=1e-3, clipnorm=4), metrics=['accuracy'])



    return model
model = create_model(embedding)
model.summary()
batch_size = 256

epochs = 1



model.fit(X_train, y_train,

          batch_size=batch_size,

          epochs=epochs,

          validation_data=(X_val, y_val),

          callbacks=[

              RocAucEvaluation(verbose=True),

              ModelCheckpoint(file_path,    monitor='roc_auc', mode='max', save_best_only=True),

              EarlyStopping(patience=10,    monitor="roc_auc", mode="max"),

              ReduceLROnPlateau(patience=0, monitor='roc_auc', mode='max', cooldown=2, min_lr=1e-7, factor=0.3)

          ]

)
model.load_weights(file_path)



sample_submission = pd.read_csv('../input/jigsaw-toxic-comment-classification-challenge/sample_submission.csv')

sample_submission[list_classes] = model.predict(X_te, verbose=True)

sample_submission.to_csv('submission.csv', index=False)