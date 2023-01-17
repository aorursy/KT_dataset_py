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
!ls ../input/glove-global-vectors-for-word-representation

import random

import copy

import time

import pandas as pd

import numpy as np

import gc

import catboost

from catboost import CatBoostClassifier



import re

import torch

from torchtext import data

#import spacy

from tqdm import tqdm_notebook, tnrange

from tqdm.auto import tqdm



tqdm.pandas(desc='Progress')

from collections import Counter

from textblob import TextBlob

from nltk import word_tokenize

import xgboost as xgb



import torch.nn as nn

import torch.optim as optim

import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from torch.autograd import Variable

from torchtext.data import Example

from sklearn.metrics import f1_score

import torchtext

import os 



from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences



# cross validation and metrics

from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import f1_score

from torch.optim.optimizer import Optimizer

from unidecode import unidecode



from sklearn.preprocessing import StandardScaler

from textblob import TextBlob

from multiprocessing import  Pool

from functools import partial

import numpy as np

from sklearn.decomposition import PCA

import torch as t

import torch.nn as nn

import torch.nn.functional as F

def seed_everything(seed=10):

    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.deterministic = True

seed_everything()

train_df = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")

train_df = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")
len(train_df[train_df.target == 0])/len(train_df)
train_df[train_df.keyword.notnull()]
train_df[train_df.location.notnull()]
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator



import matplotlib.pyplot as plt

"abc".lower()
text = " ".join(train_df.iloc[0:100].text)

wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(text)

plt.figure()

plt.imshow(wordcloud, interpolation="bilinear")

plt.axis("off")

plt.show()

from nltk.stem import PorterStemmer

# stemming.stem("hello")
train_df['text'] = train_df['text'].apply(lambda x: x.lower())
##REF : FUNCTIONS TAKEN FROM https://www.kaggle.com/gmhost/gru-capsule

def load_glove(word_index):

    EMBEDDING_FILE = '../input/embeddings/glove.840B.300d/glove.840B.300d.txt'

    def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')[:300]

    embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE))

    

    all_embs = np.stack(embeddings_index.values())

    emb_mean,emb_std = -0.005838499,0.48782197

    embed_size = all_embs.shape[1]



    # word_index = tokenizer.word_index

    nb_words = min(max_features, len(word_index))

    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))

    for word, i in word_index.items():

        if i >= max_features: continue

        embedding_vector = embeddings_index.get(word)

        #ALLmight

        if embedding_vector is not None: 

            embedding_matrix[i] = embedding_vector

        else:

            embedding_vector = embeddings_index.get(word.capitalize())

            if embedding_vector is not None: 

                embedding_matrix[i] = embedding_vector

    return embedding_matrix 

    

            

def load_fasttext(word_index):    

    EMBEDDING_FILE = '../input/embeddings/wiki-news-300d-1M/wiki-news-300d-1M.vec'

    def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')

    embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE) if len(o)>100)



    all_embs = np.stack(embeddings_index.values())

    emb_mean,emb_std = all_embs.mean(), all_embs.std()

    embed_size = all_embs.shape[1]



    # word_index = tokenizer.word_index

    nb_words = min(max_features, len(word_index))

    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))

    for word, i in word_index.items():

        if i >= max_features: continue

        embedding_vector = embeddings_index.get(word)

        if embedding_vector is not None: embedding_matrix[i] = embedding_vector



    return embedding_matrix



def load_para(word_index):

    EMBEDDING_FILE = '../input/embeddings/paragram_300_sl999/paragram_300_sl999.txt'

    def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')

    embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE, encoding="utf8", errors='ignore') if len(o)>100)



    all_embs = np.stack(embeddings_index.values())

    emb_mean,emb_std = -0.0053247833,0.49346462

    embed_size = all_embs.shape[1]



    # word_index = tokenizer.word_index

    nb_words = min(max_features, len(word_index))

    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))

    for word, i in word_index.items():

        if i >= max_features: continue

        embedding_vector = embeddings_index.get(word)

        if embedding_vector is not None: embedding_matrix[i] = embedding_vector

    

    return embedding_matrix
from nltk.stem import PorterStemmer

from nltk.corpus import stopwords

import string

import sklearn

from sklearn.linear_model import LogisticRegression

from sklearn import linear_model

from sklearn.svm import LinearSVC

  

import gensim 

from gensim.models import Word2Vec 

#REF : https://www.kaggle.com/mlwhiz/attention-pytorch-and-keras



mispell_dict = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not", "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not", "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",  "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would", "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would", "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is", "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as", "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would", "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",  "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is", "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have", "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have","you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have", "you're": "you are", "you've": "you have", 'colour': 'color', 'centre': 'center', 'favourite': 'favorite', 'travelling': 'traveling', 'counselling': 'counseling', 'theatre': 'theater', 'cancelled': 'canceled', 'labour': 'labor', 'organisation': 'organization', 'wwii': 'world war 2', 'citicise': 'criticize', 'youtu ': 'youtube ', 'Qoura': 'Quora', 'sallary': 'salary', 'Whta': 'What', 'narcisist': 'narcissist', 'howdo': 'how do', 'whatare': 'what are', 'howcan': 'how can', 'howmuch': 'how much', 'howmany': 'how many', 'whydo': 'why do', 'doI': 'do I', 'theBest': 'the best', 'howdoes': 'how does', 'mastrubation': 'masturbation', 'mastrubate': 'masturbate', "mastrubating": 'masturbating', 'pennis': 'penis', 'Etherium': 'Ethereum', 'narcissit': 'narcissist', 'bigdata': 'big data', '2k17': '2017', '2k18': '2018', 'qouta': 'quota', 'exboyfriend': 'ex boyfriend', 'airhostess': 'air hostess', "whst": 'what', 'watsapp': 'whatsapp', 'demonitisation': 'demonetization', 'demonitization': 'demonetization', 'demonetisation': 'demonetization'}

def _get_mispell(mispell_dict):

    mispell_re = re.compile('(%s)' % '|'.join(mispell_dict.keys()))

    return mispell_dict, mispell_re



mispellings, mispellings_re = _get_mispell(mispell_dict)

def replace_typical_misspell(text):

    def replace(match):

        return mispellings[match.group(0)]

    return mispellings_re.sub(replace, text)







class NLPTrain:

    

    def __convert_to_lowercase__(self):

        self.data[self.column] = self.data[self.column].apply(lambda x: x.lower())

    

    def __remove_punc__(self):

        self.data[self.column] = self.data[self.column].apply(lambda x: x.translate(str.maketrans('','',string.punctuation)))

    

    def __convert_num__(self):

        self.data[self.column] = self.data[self.column].apply(lambda x: x.translate(str.maketrans('','',string.digits)))



    def __remove_stop_words__(self):

        self.data[self.column] = self.data[self.column].apply(lambda x: ' '.join([word for word in x.split() if word not in (self.stop)]))

    

    def __stemming__(self):

        self.data[self.column] = self.data[self.column].apply(lambda x: self.stemming.stem(x))

        

    def __mispell__(self):

        self.data[self.column] = self.data[self.column].apply(lambda x : replace_typical_misspell(x))

        

    def __init__(self,data, column = "text",target ="target"):

        self.stemming = PorterStemmer()

        self.stop = stopwords.words('english')

        self.data = data

        self.target = target

        self.column = column

    

    def preprocess(self):

        self.__convert_to_lowercase__()

        self.__remove_punc__()

        self.__convert_num__()

        self.__remove_stop_words__()

        self.__stemming__()

        

    def train_test_and_validate(self):

        self.Xtrain,self.Xtest,self.y_train,self.y_test = sklearn.model_selection.train_test_split(self.data[self.column],

                                                                                                   self.data[self.target],

                                                                                                   test_size =0.4,

                                                                                                  random_state=42)

        self.Xval,self.Xtest,self.y_val,self.y_test = sklearn.model_selection.train_test_split(self.Xtest,

                                                                                               self.y_test,

                                                                                                test_size =0.5,

                                                                                                random_state=42)

        

    def tf_idf(self):

        self.vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(ngram_range=(1, 4))

        self.Xtrain_idf = self.vectorizer.fit_transform(self.Xtrain)

        self.Xval_idf = self.vectorizer.transform(self.Xval)

        

    def word2vec(self):

        pass

    

    def fit(self):

        self.clf1 = LogisticRegression(random_state=0).fit(self.Xtrain_idf, self.y_train)

        print("LR Test :" + str(self.clf1.score(self.Xtrain_idf, self.y_train)))

        print("LR Validation : " + str(self.clf1.score(self.Xval_idf, self.y_val)))

        

        self.clf2 = LinearSVC(random_state=0, tol=1e-5).fit(self.Xtrain_idf, self.y_train)

        print("SGD Test :" + str(self.clf2.score(self.Xtrain_idf, self.y_train)))

        print("SGD Validation : " + str(self.clf2.score(self.Xval_idf, self.y_val)))

#         model=xgb.XGBClassifier(random_state=1,learning_rate=0.01)

#         model.fit(self.Xtrain_idf, self.y_train)

#         print("XGBoost:" + str(model.score(self.Xval_idf, self.y_val)))



    

#         model = CatBoostClassifier(

#         iterations=10,

#         learning_rate=0.0001,

#         loss_function='CrossEntropy')

#         model.fit(

#             self.Xtrain_idf, self.y_train,

#             eval_set=(self.Xval_idf, self.y_val),

#             verbose=True,

#             plot=True

#         )

        

#         print('Model is fitted: ' + str(model.is_fitted()))

#         print('Model params:')



    

    def attention_network(self):

        

        tokenizer = Tokenizer(num_words=max_features)

        tokenizer.fit_on_texts(list(self.Xtrain))

        train_X = tokenizer.texts_to_sequences(self.Xtrain)

        test_X = tokenizer.texts_to_sequences(self.Xval)

        train_X = pad_sequences(train_X, maxlen=maxlen)

        test_X = pad_sequences(test_X, maxlen=maxlen)

        train_y = self.y_train.values

        

        seed_everything()

        paragram_embeddings = np.random.randn(120000,300)

        glove_embeddings = np.random.randn(120000,300)

        self.embedding_matrix = np.mean([glove_embeddings, paragram_embeddings], axis=0)

        self.train_X_token = train_X

        self.train_y_token = train_y

        self.test_X_token = test_X

        



embed_size = 300 # how big is each word vector

max_features = 120000 # how many unique words to use (i.e num rows in embedding vector)

maxlen = 70 # max number of words in a question to use

batch_size = 512 # how many samples to process at once

n_epochs = 5 # how many times to iterate over all samples

n_splits = 5 # Number of K-fold Splits

SEED = 10

debug = 0

from keras.layers import Dense, Input, CuDNNLSTM, Embedding, Dropout, Activation, CuDNNGRU, Conv1D

from keras.layers import Bidirectional, GlobalMaxPool1D, GlobalMaxPooling1D, GlobalAveragePooling1D

from keras.layers import Input, Embedding, Dense, Conv2D, MaxPool2D, concatenate

from keras.layers import Reshape, Flatten, Concatenate, Dropout, SpatialDropout1D

from keras.optimizers import Adam

from keras.models import Model

from keras.engine.topology import Layer

from keras import initializers, regularizers, constraints, optimizers, layers





from keras.layers import *

from keras.models import *

from keras import initializers, regularizers, constraints, optimizers, layers

from keras.initializers import *

from keras.optimizers import *

import keras.backend as K

from keras.callbacks import *

import tensorflow as tf



def model_train_cv(x_train,y_train,nfold,model_obj):

    splits = list(StratifiedKFold(n_splits=nfold, shuffle=True, random_state=SEED).split(x_train, y_train))

    x_train = x_train

    y_train = np.array(y_train)

    # matrix for the out-of-fold predictions

    train_oof_preds = np.zeros((x_train.shape[0]))

    for i, (train_idx, valid_idx) in enumerate(splits):

        print(f'Fold {i + 1}')

        x_train_fold = x_train[train_idx.astype(int)]

        y_train_fold = y_train[train_idx.astype(int)]

        x_val_fold = x_train[valid_idx.astype(int)]

        y_val_fold = y_train[valid_idx.astype(int)]

        # Changed it here a little bit since the custom attention layer is not getting deepcopy

        clf = model_obj

        clf.load_weights('model.h5')

        clf.fit(x_train_fold, y_train_fold, batch_size=512, epochs=5, validation_data=(x_val_fold, y_val_fold))

        

        valid_preds_fold = clf.predict(x_val_fold)[:,0]



        # storing OOF predictions

        train_oof_preds[valid_idx] = valid_preds_fold

    return train_oof_preds
def dot_product(x, kernel):

    """

    Wrapper for dot product operation, in order to be compatible with both

    Theano and Tensorflow

    Args:

        x (): input

        kernel (): weights

    Returns:

    """

    if K.backend() == 'tensorflow':

        return K.squeeze(K.dot(x, K.expand_dims(kernel)), axis=-1)

    else:

        return K.dot(x, kernel)

    



class AttentionWithContext(Layer):

    """

    Attention operation, with a context/query vector, for temporal data.

    Supports Masking.

    Follows the work of Yang et al. [https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf]

    "Hierarchical Attention Networks for Document Classification"

    by using a context vector to assist the attention

    # Input shape

        3D tensor with shape: `(samples, steps, features)`.

    # Output shape

        2D tensor with shape: `(samples, features)`.

    How to use:

    Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.

    The dimensions are inferred based on the output shape of the RNN.

    Note: The layer has been tested with Keras 2.0.6

    Example:

        model.add(LSTM(64, return_sequences=True))

        model.add(AttentionWithContext())

        # next add a Dense layer (for classification/regression) or whatever...

    """

    def __init__(self,

                 W_regularizer=None, u_regularizer=None, b_regularizer=None,

                 W_constraint=None, u_constraint=None, b_constraint=None,

                 bias=True, **kwargs):



        self.supports_masking = True

        self.init = initializers.get('glorot_uniform')



        self.W_regularizer = regularizers.get(W_regularizer)

        self.u_regularizer = regularizers.get(u_regularizer)

        self.b_regularizer = regularizers.get(b_regularizer)



        self.W_constraint = constraints.get(W_constraint)

        self.u_constraint = constraints.get(u_constraint)

        self.b_constraint = constraints.get(b_constraint)



        self.bias = bias

        super(AttentionWithContext, self).__init__(**kwargs)



    def build(self, input_shape):

        assert len(input_shape) == 3



        self.W = self.add_weight((input_shape[-1], input_shape[-1],),

                                 initializer=self.init,

                                 name='{}_W'.format(self.name),

                                 regularizer=self.W_regularizer,

                                 constraint=self.W_constraint)

        if self.bias:

            self.b = self.add_weight((input_shape[-1],),

                                     initializer='zero',

                                     name='{}_b'.format(self.name),

                                     regularizer=self.b_regularizer,

                                     constraint=self.b_constraint)



        self.u = self.add_weight((input_shape[-1],),

                                 initializer=self.init,

                                 name='{}_u'.format(self.name),

                                 regularizer=self.u_regularizer,

                                 constraint=self.u_constraint)



        super(AttentionWithContext, self).build(input_shape)



    def compute_mask(self, input, input_mask=None):

        # do not pass the mask to the next layers

        return None



    def call(self, x, mask=None):

        uit = dot_product(x, self.W)



        if self.bias:

            uit += self.b



        uit = K.tanh(uit)

        ait = dot_product(uit, self.u)



        a = K.exp(ait)



        # apply mask after the exp. will be re-normalized next

        if mask is not None:

            # Cast the mask to floatX to avoid float64 upcasting in theano

            a *= K.cast(mask, K.floatx())



        # in some cases especially in the early stages of training the sum may be almost zero

        # and this results in NaN's. A workaround is to add a very small positive number ε to the sum.

        # a /= K.cast(K.sum(a, axis=1, keepdims=True), K.floatx())

        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())



        a = K.expand_dims(a)

        weighted_input = x * a

        return K.sum(weighted_input, axis=1)



    def compute_output_shape(self, input_shape):

        return input_shape[0], input_shape[-1]





def model_lstm_atten(embedding_matrix):

    inp = Input(shape=(maxlen,))

    x = Embedding(max_features, embed_size, weights=[embedding_matrix], trainable=False)(inp)

    x = Bidirectional(CuDNNLSTM(128, return_sequences=True))(x)

    x = Bidirectional(CuDNNLSTM(64, return_sequences=True))(x)

    x = AttentionWithContext()(x)

    x = Dense(64, activation="relu")(x)

    x = Dense(1, activation="sigmoid")(x)

    model = Model(inputs=inp, outputs=x)

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model





# if debug:

# else:

nlp_trainer = NLPTrain(train_df,"text")
nlp_trainer.preprocess()

nlp_trainer.train_test_and_validate()
nlp_trainer.tf_idf()
nlp_trainer.fit()
model1 = gensim.models.Word2Vec(nlp_trainer.Xtrain, min_count = 1,  

                              size = 100, window = 5) 

nlp_trainer.attention_network()

# model = model_lstm_atten(nlp_trainer.embedding_matrix)

# train_oof_preds = model_train_cv(nlp_trainer.train_X_token,nlp_trainer.train_y_token,5,model)
class MyDataset(Dataset):

    def __init__(self,dataset):

        self.dataset = dataset

    def __getitem__(self,index):

        data,target = self.dataset[index]

        return data,target,index

    def __len__(self):

        return len(self.dataset)

def pytorch_model_run_cv(x_train,y_train,features,x_test, model_obj, feats = False,clip = True):

    seed_everything()

    avg_losses_f = []

    avg_val_losses_f = []

    # matrix for the out-of-fold predictions

    train_preds = np.zeros((len(x_train)))

    # matrix for the predictions on the test set

    test_preds = np.zeros((len(x_test)))

    splits = list(StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED).split(x_train, y_train))

    for i, (train_idx, valid_idx) in enumerate(splits):

        seed_everything(i*1000+i)

        x_train = np.array(x_train)

        y_train = np.array(y_train)

        if feats:

            features = np.array(features)

        x_train_fold = torch.tensor(x_train[train_idx.astype(int)], dtype=torch.long).cuda()

        y_train_fold = torch.tensor(y_train[train_idx.astype(int), np.newaxis], dtype=torch.float32).cuda()

        if feats:

            kfold_X_features = features[train_idx.astype(int)]

            kfold_X_valid_features = features[valid_idx.astype(int)]

        x_val_fold = torch.tensor(x_train[valid_idx.astype(int)], dtype=torch.long).cuda()

        y_val_fold = torch.tensor(y_train[valid_idx.astype(int), np.newaxis], dtype=torch.float32).cuda()

        

        model = copy.deepcopy(model_obj)



        model.cuda()



        loss_fn = torch.nn.BCEWithLogitsLoss(reduction='sum')

        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), 

                                 lr=0.001)

        

        ################################################################################################

        scheduler = False

        ###############################################################################################



        train = MyDataset(torch.utils.data.TensorDataset(x_train_fold, y_train_fold))

        valid = MyDataset(torch.utils.data.TensorDataset(x_val_fold, y_val_fold))

        

        train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)

        valid_loader = torch.utils.data.DataLoader(valid, batch_size=batch_size, shuffle=False)



        print(f'Fold {i + 1}')

        for epoch in range(n_epochs):

            start_time = time.time()

            model.train()



            avg_loss = 0.  

            for i, (x_batch, y_batch, index) in enumerate(train_loader):

                if feats:       

                    f = kfold_X_features[index]

                    y_pred = model([x_batch,f])

                else:

                    y_pred = model(x_batch)



                if scheduler:

                    scheduler.batch_step()



                # Compute and print loss.

                loss = loss_fn(y_pred, y_batch)

                optimizer.zero_grad()

                loss.backward()

                if clip:

                    nn.utils.clip_grad_norm_(model.parameters(),1)

                optimizer.step()

                avg_loss += loss.item() / len(train_loader)

                

            model.eval()

            

            valid_preds_fold = np.zeros((x_val_fold.size(0)))

            test_preds_fold = np.zeros((len(x_test)))

            

            avg_val_loss = 0.

            for i, (x_batch, y_batch,index) in enumerate(valid_loader):

                if feats:

                    f = kfold_X_valid_features[index]            

                    y_pred = model([x_batch,f]).detach()

                else:

                    y_pred = model(x_batch).detach()

                

                avg_val_loss += loss_fn(y_pred, y_batch).item() / len(valid_loader)

                valid_preds_fold[index] = sigmoid(y_pred.cpu().numpy())[:, 0]

            

            elapsed_time = time.time() - start_time 

            print('Epoch {}/{} \t loss={:.4f} \t val_loss={:.4f} \t time={:.2f}s'.format(

                epoch + 1, n_epochs, avg_loss, avg_val_loss, elapsed_time))

        avg_losses_f.append(avg_loss)

        avg_val_losses_f.append(avg_val_loss) 

        # predict all samples in the test set batch per batch

        for i, (x_batch,) in enumerate(test_loader):

            if feats:

                f = test_features[i * batch_size:(i+1) * batch_size]

                y_pred = model([x_batch,f]).detach()

            else:

                y_pred = model(x_batch).detach()



            test_preds_fold[i * batch_size:(i+1) * batch_size] = sigmoid(y_pred.cpu().numpy())[:, 0]

            

        train_preds[valid_idx] = valid_preds_fold

        test_preds += test_preds_fold / len(splits)



    print('All \t loss={:.4f} \t val_loss={:.4f} \t '.format(np.average(avg_losses_f),np.average(avg_val_losses_f)))

    return train_preds, test_preds
class Attention(nn.Module):

    def __init__(self, feature_dim, step_dim, bias=True, **kwargs):

        super(Attention, self).__init__(**kwargs)

        

        self.supports_masking = True



        self.bias = bias

        self.feature_dim = feature_dim

        self.step_dim = step_dim

        self.features_dim = 0

        

        weight = torch.zeros(feature_dim, 1)

        nn.init.kaiming_uniform_(weight)

        self.weight = nn.Parameter(weight)

        

        if bias:

            self.b = nn.Parameter(torch.zeros(step_dim))

        

    def forward(self, x, mask=None):

        feature_dim = self.feature_dim 

        step_dim = self.step_dim



        eij = torch.mm(

            x.contiguous().view(-1, feature_dim), 

            self.weight

        ).view(-1, step_dim)

        

        if self.bias:

            eij = eij + self.b

            

        eij = torch.tanh(eij)

        a = torch.exp(eij)

        

        if mask is not None:

            a = a * mask



        a = a / (torch.sum(a, 1, keepdim=True) + 1e-10)



        weighted_input = x * torch.unsqueeze(a, -1)

        return torch.sum(weighted_input, 1)



class Attention_Net(nn.Module):

    def __init__(self):

        super(Attention_Net, self).__init__()

        drp = 0.1

        self.embedding = nn.Embedding(max_features, embed_size)

        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))

        self.embedding.weight.requires_grad = True



        self.embedding_dropout = nn.Dropout2d(0.1)

        self.lstm = nn.LSTM(embed_size, 128, bidirectional=True, batch_first=True)

        self.lstm2 = nn.GRU(128*2, 64, bidirectional=True, batch_first=True)



        self.attention_layer = Attention(128, maxlen)

        

        self.linear = nn.Linear(64*2 , 64)

        self.relu = nn.ReLU()

        self.out = nn.Linear(64, 1)



    def forward(self, x):

        h_embedding = self.embedding(x)

        h_embedding = torch.squeeze(torch.unsqueeze(h_embedding, 0))

        h_lstm, _ = self.lstm(h_embedding)

        h_lstm, _ = self.lstm2(h_lstm)

        h_lstm_atten = self.attention_layer(h_lstm)

        conc = self.relu(self.linear(h_lstm_atten))

        out = self.out(conc)

        return out
paragram_embeddings = np.random.randn(120000,300)

glove_embeddings = np.random.randn(120000,300)

embedding_matrix = np.mean([glove_embeddings, paragram_embeddings], axis=0)

def sigmoid(x):

    return 1 / (1 + np.exp(-x))



# always call this before training for deterministic results

seed_everything()



x_test_cuda = torch.tensor(nlp_trainer.test_X_token, dtype=torch.long).cuda()

test = torch.utils.data.TensorDataset(x_test_cuda)

test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False)
train_preds , test_preds = pytorch_model_run_cv(nlp_trainer.train_X_token,nlp_trainer.train_y_token,None,nlp_trainer.test_X_token,Attention_Net(), feats = False, clip=False)
def bestThresshold(y_train,train_preds):

    tmp = [0,0,0] # idx, cur, max

    delta = 0

    for tmp[0] in tqdm(np.arange(0.1, 0.501, 0.01)):

        tmp[1] = f1_score(y_train, np.array(train_preds)>tmp[0])

        if tmp[1] > tmp[2]:

            delta = tmp[0]

            tmp[2] = tmp[1]

    print('best threshold is {:.4f} with F1 score: {:.4f}'.format(delta, tmp[2]))

    return delta , tmp[2]



delta, _ = bestThresshold(nlp_trainer.train_y_token,train_preds)
test_preds