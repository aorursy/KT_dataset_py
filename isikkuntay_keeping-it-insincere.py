import os
import json
import string
import numpy as np
import pandas as pd
import keras
from pandas.io.json import json_normalize
import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()
from math import floor

%matplotlib inline

from plotly import tools
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go

from sklearn import model_selection, preprocessing, metrics, ensemble, naive_bayes, linear_model
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
import lightgbm as lgb

import time
from tqdm import tqdm
import math
from sklearn.model_selection import train_test_split

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, GRU, CuDNNGRU, Conv1D
from keras.layers import Bidirectional, GlobalMaxPool1D, Concatenate, Add, Flatten, CuDNNLSTM
from keras.models import Model
from keras import backend as K
from keras import initializers, regularizers, constraints, optimizers, layers
from keras.engine.topology import Layer
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

pd.options.mode.chained_assignment = None
pd.options.display.max_columns = 999

from wordcloud import WordCloud, STOPWORDS
from collections import defaultdict
import nltk 
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize, sent_tokenize 
stop_words = set(stopwords.words('english')) 
import regex as re
puncts = [',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&', '/', '[', ']', '>', '%', '=', '#', '*', '+', '\\', '•',  '~', '@', '£', 
 '·', '_', '{', '}', '©', '^', '®', '`',  '<', '→', '°', '€', '™', '›',  '♥', '←', '×', '§', '″', '′', 'Â', '█', '½', 'à', '…', 
 '“', '★', '”', '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶', '↑', '±', '¿', '▾', '═', '¦', '║', '―', '¥', '▓', '—', '‹', '─', 
 '▒', '：', '¼', '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲', 'è', '¸', '¾', 'Ã', '⋅', '‘', '∞', 
 '∙', '）', '↓', '、', '│', '（', '»', '，', '♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø', '¹', '≤', '‡', '√', ]
def clean_text(x):
    x = str(x)
    for punct in puncts:
        x = x.replace(punct, f' {punct} ')
    return x
train_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/test.csv")
print("Train shape : ",train_df.shape)
print("Test shape : ",test_df.shape)
# Cleaning the data
train_df["question_text"] = train_df["question_text"].apply(lambda x: clean_text(x))
test_df["question_text"] = test_df["question_text"].apply(lambda x: clean_text(x))
countries = ["United States", "France", "India", "China", "Russia", "Saudi Arabia", "Iran", \
             "Germany","England", "Europe", "Mexico", "Italy", "Spain", "Turkey", "Syria", "Israel"]

politicians = ["Trump", "Hillary", "Pence", "Obama", "Bush", "Clinton", "Putin", "Merkel"]

groups = ["Hispanics", "Blacks", "Whites", "Asians", "Browns", "Arabs", "Natives", "Caucasians", "Africans",\
         "Indians", "Chineses", "Mexicans", "British", "Americans", "Spanish", "Greek", "Italians", \
          "Turkish", "French", "English", "Germans", "Russians","Arabic", "Persians", "European", "Madeup"]

political_groups = ["democrats", "republicans", "liberals", "socialists", "communists"]

religious_groups = ["christians", "muslims", "jews", "buddhists", "hindus", "israelites", "zionists"]

goofy_test_questions = []
for country in countries:
    goofy_test_questions.append("Why do not people see that " + country + " is a terrorist country?")

for country1 in countries:
    for country2 in countries:
        goofy_test_questions.append("When will " + country1 + " end its occupation of " + country2 + "?")
        
for politician in politicians:
    goofy_test_questions.append("How did an idiot like " + politician + " get elected?")

for politician in politicians:
    goofy_test_questions.append("How did a moron like " + politician + " get elected?")

for politician in politicians:
    goofy_test_questions.append("How did an incompetent officer like " + politician + " get elected?")

for politician in politicians:
    goofy_test_questions.append("How did " + politician + " get elected?")

for politician in politicians:
    goofy_test_questions.append("How does " + politician + " feel about the economic wreck he left behind?")
    
for group in groups:
    goofy_test_questions.append("Are all " + group + " stupid?")

for group in groups:
    goofy_test_questions.append("Are " + group + " stupid?")

for group in groups:
    goofy_test_questions.append("Are all " + group + " rude?")

for group in groups:
    goofy_test_questions.append("Are " + group + " rude?")

for country in countries:
    goofy_test_questions.append("Why does " + country + " food taste so good?")

for country in countries:
    goofy_test_questions.append("Why does " + country + " food taste so bad?")

for group in groups:
    goofy_test_questions.append("Why do " + group + " people think themselves so high?")

for group in groups:
    goofy_test_questions.append("Should all " + group + " be spaded?")

for group in groups:
    goofy_test_questions.append("Should all " + group + " be sent to Mars and help build an infrastructure there to be useful for mankind and the world?")

for group1 in groups:
    for group2 in groups:
        goofy_test_questions.append("What do " + group1 + " think of " + group2 + "?")

for political_group in political_groups:
    goofy_test_questions.append("Do " + political_group + " even care about the constitution?")

for political_group in political_groups:
    goofy_test_questions.append("Do " + political_group + " care about the country?")

for religious_group in religious_groups:
    goofy_test_questions.append("Do " + religious_group + " really believe in a book written by a child molester?")

goofy_test_y = np.ones(len(goofy_test_questions))


for sentence in goofy_test_questions:
    sentence = clean_text(sentence)
        
goofy_test_X = np.asarray(goofy_test_questions)
## some config values 
embed_size = 300 # how big is each word vector
max_features = 90000 # how many unique words to use (i.e num rows in embedding vector)
maxlen = 60 # max number of words in a question to use
def get_long_sentences(sent_list, cut_off_len = maxlen):
    ids_for_long = []
    section_list = []
    for i, sent in enumerate(list(sent_list)):
        for section in re.split(r'([\w+\s?]+[.?!])', sent):
            if len(section) > 1:
                section_list.append(section)
                ids_for_long.append(i)
    return section_list, ids_for_long
def get_generalized_sentences(sent_list, y_list):
    global groups, political_groups, countries, religious_groups
    general_X_list = []
    general_y_list = []
    for i, sent in enumerate(list(sent_list)):
        for word in sent.split():
            if (word in groups) or (word in political_groups) or (word in countries) or (word in religious_groups) :
                new_sent = sent.replace(word, "oovword")
                general_X_list.append(new_sent)
                general_y_list.append(y_list[i])
    return general_X_list, general_y_list
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
## split to train and val
train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=2018)
## Get the target values
train_y = train_df['target'].values
val_y = val_df['target'].values
## fill up the missing values
train_X = train_df["question_text"].fillna("_na_").values
val_X = val_df["question_text"].fillna("_na_").values
test_X = test_df["question_text"].fillna("_na_").values
generalized_train_X, generalized_train_y = get_generalized_sentences(train_X, train_y)
long_val_X, long_id_list = get_long_sentences(val_X)
long_test_X, long_test_id = get_long_sentences(test_X)
## Tokenize the sentences
tokenizer = Tokenizer(num_words=max_features)

tokenizer_list = list(train_X)

tokenizer.fit_on_texts(tokenizer_list)
train_X = tokenizer.texts_to_sequences(train_X)
val_X = tokenizer.texts_to_sequences(val_X)
test_X = tokenizer.texts_to_sequences(test_X)

long_val_X = tokenizer.texts_to_sequences(np.asarray(long_val_X))
long_test_X = tokenizer.texts_to_sequences(np.asarray(long_test_X))

generalized_train_X = tokenizer.texts_to_sequences(np.asarray(generalized_train_X))
generalized_train_y = np.asarray(generalized_train_y)
goofy_test_X = tokenizer.texts_to_sequences(goofy_test_X)
## Pad the sentences for short
train_X = pad_sequences(train_X, maxlen=maxlen)
val_X = pad_sequences(val_X, maxlen=maxlen)
test_X = pad_sequences(test_X, maxlen=maxlen)

long_val_X = pad_sequences(long_val_X, maxlen=maxlen)
long_test_X = pad_sequences(long_test_X, maxlen=maxlen)
generalized_train_X = pad_sequences(generalized_train_X, maxlen=maxlen)
goofy_test_X = pad_sequences(goofy_test_X, maxlen=maxlen)
EMBEDDING_FILE = '../input/embeddings/glove.840B.300d/glove.840B.300d.txt'
def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE))

all_embs = np.stack(embeddings_index.values())
emb_mean,emb_std = all_embs.mean(), all_embs.std()
embed_size = all_embs.shape[1]

random_vector = np.random.rand(300)

word_index = tokenizer.word_index
nb_words = min(max_features, len(word_index))
embedding_matrix1 = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
for word, i in word_index.items():
    if i >= max_features: continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None: 
        embedding_matrix1[i] = embedding_vector
    else:
        embedding_matrix1[i] = random_vector
EMBEDDING_FILE = '../input/embeddings/wiki-news-300d-1M/wiki-news-300d-1M.vec'
def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE) if len(o)>100)

all_embs = np.stack(embeddings_index.values())
emb_mean,emb_std = all_embs.mean(), all_embs.std()
embed_size = all_embs.shape[1]

word_index = tokenizer.word_index
nb_words = min(max_features, len(word_index))
embedding_matrix2 = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
for word, i in word_index.items():
    if i >= max_features: continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None: embedding_matrix2[i] = embedding_vector
inp = Input(shape=(maxlen,))
model1_out = Embedding(max_features, embed_size, weights=[embedding_matrix1],trainable=False)(inp)
model1_out = Bidirectional(CuDNNGRU(128, return_sequences=True))(model1_out)
model1_out = AttentionWithContext()(model1_out)
model1_out = Dense(64, activation="relu")(model1_out)
model1_out = Dropout(0.1)(model1_out)
model1_out = Dense(32, activation="relu")(model1_out)
model1_out = Dropout(0.1)(model1_out)
model1_out = Dense(1, activation="sigmoid")(model1_out)
model = Model(inputs=inp, outputs=model1_out)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
inp = Input(shape=(maxlen,))
model2_out = Embedding(max_features, embed_size, weights=[embedding_matrix2],trainable=False)(inp)
model2_out = Bidirectional(CuDNNGRU(128, return_sequences=True))(model2_out)
model2_out = AttentionWithContext()(model2_out)
model2_out = Dense(64, activation="relu")(model2_out)
model2_out = Dropout(0.1)(model2_out)
model2_out = Dense(32, activation="relu")(model2_out)
model2_out = Dropout(0.1)(model2_out)
model2_out = Dense(1, activation="sigmoid")(model2_out)
model2 = Model(inputs=inp, outputs=model2_out)
model2.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model2.summary())
def train_model(model, all_train_X, all_train_y, all_val_X, all_val_y, epochs=2):
    filepath="weights_best.h5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=2, save_best_only=True, mode='min')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=1, min_lr=0.0001, verbose=2)
    earlystopping = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=2, verbose=2, mode='auto')
    callbacks = [checkpoint, reduce_lr]
    for e in range(epochs):
        model.fit(all_train_X, all_train_y, batch_size=1024, epochs=1, validation_data=(all_val_X, all_val_y), callbacks=callbacks)
    model.load_weights(filepath)
    return model
model = train_model(model, train_X, train_y, val_X, val_y, epochs=12)
pred_val_y = model.predict(val_X, batch_size=1024, verbose=1)
model2 = train_model(model2, train_X, train_y, val_X, val_y, epochs=5)
alt_pred_val_y = model2.predict(val_X, batch_size=1024, verbose=1)
'''
A function specific to this competition since the organizers don't want probabilities 
and only want 0/1 classification maximizing the F1 score. This function computes the best F1 score by looking at val set predictions
'''

def f1_smart(y_true, y_pred):
    thresholds = []
    for thresh in np.arange(0.1, 0.901, 0.01):
        thresh = np.round(thresh, 2)
        res = metrics.f1_score(y_true, (y_pred > thresh).astype(int))
        thresholds.append([thresh, res])
        print("F1 score at threshold {0} is {1}".format(thresh, res))

    thresholds.sort(key=lambda x: x[1], reverse=True)
    best_thresh = thresholds[0][0]
    best_f1 = thresholds[0][1]
    print("Best threshold: ", best_thresh)
    return  best_f1, best_thresh
f1, threshold = f1_smart(val_y, pred_val_y)
print('Optimal F1: {} at threshold: {}'.format(f1, threshold))
f1, alt_threshold = f1_smart(val_y, alt_pred_val_y)
print('Optimal F1: {} at threshold: {}'.format(f1, alt_threshold))
long_pred_val_y = model.predict(long_val_X, batch_size=1024, verbose=1)
def update_preds(pred_val_y, long_pred_val_y, long_id_list):
    global threshold
    threshold_range = 0.10
    copy_pred_val_y = pred_val_y
    for i, pred in enumerate(list(pred_val_y)):
        if (pred < (threshold + threshold_range)) and (pred > (threshold - threshold_range)):
        # We do this extra step only if the model is not confident about the pred
            count = 1
            sum_pred = pred
            for long_id, long_pred in zip(long_id_list, long_pred_val_y.tolist()):
                if long_id == i:
                    if (long_pred > (threshold + threshold_range)) or (long_pred < (threshold - threshold_range)):
                        if long_pred[0] > pred:
                    # We keep the pred that is closer to insincere                     
                            copy_pred_val_y[i] = long_pred[0]
    return copy_pred_val_y
pred_val_y = update_preds(pred_val_y, long_pred_val_y, long_id_list)
f1, threshold = f1_smart(val_y, pred_val_y)
print('Optimal F1: {} at threshold: {}'.format(f1, threshold))
all_train_X = np.concatenate((train_X,val_X),axis=0)
all_train_y = np.concatenate((train_y,val_y),axis=0)

all_train_X = np.concatenate((train_X,generalized_train_X),axis=0)
all_train_y = np.concatenate((train_y,generalized_train_y),axis=0)

model = train_model(model, train_X, train_y, val_X, val_y, epochs=5)
def seq_ensemble(pred_y, alt_pred_y):
    global threshold, alt_threshold
    threshold_range_list = [0.15,0.12,0.10,0.08,0.05,0.03,0.02,0.01]
    copy_pred_val_y = pred_val_y
    for threshold_range in threshold_range_list:
        for i, pred_pair in enumerate(zip(list(pred_val_y),list(alt_pred_y))):
            if (pred_pair[0] < (threshold + threshold_range)) and \
                            (pred_pair[0] > (threshold - threshold_range)):
                if (pred_pair[1] > (alt_threshold + threshold_range)) or \
                            (pred_pair[1] < (alt_threshold - threshold_range)):
                    copy_pred_val_y[i] = pred_pair[1]    
    return copy_pred_val_y
pred_val_y = seq_ensemble(pred_val_y, alt_pred_val_y)
f1, threshold = f1_smart(val_y, pred_val_y)
print('Optimal F1: {} at threshold: {}'.format(f1, threshold))
pred_test_y = model.predict(test_X, batch_size=1024, verbose=1)
long_pred_test_y = model.predict(long_test_X, batch_size=1024, verbose=1)
pred_test_y = update_preds(pred_test_y, long_pred_test_y, long_test_id)
pred_test_y = (pred_test_y>threshold).astype(int)
out_df = pd.DataFrame({"qid":test_df["qid"].values})
out_df['prediction'] = pred_test_y
out_df.to_csv("submission.csv", index=False)
pred_goofy_y = model.predict(goofy_test_X, batch_size=10, verbose=1)
f1, threshold = f1_smart(goofy_test_y, pred_goofy_y)
print('Optimal F1: {} at threshold: {}'.format(f1, threshold))

for question,pred in zip(goofy_test_questions, pred_goofy_y): 
    print(pred, " ", question, "\n")