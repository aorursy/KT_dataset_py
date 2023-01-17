import os

import time

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from tqdm import tqdm

from keras.engine.topology import Layer

import math

import operator 

from sklearn.model_selection import train_test_split

from sklearn import metrics

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, CuDNNGRU, Conv1D, TimeDistributed, CuDNNLSTM,Conv2D

from keras.layers import Bidirectional, GlobalMaxPool1D, GlobalAveragePooling1D, concatenate, Flatten, Reshape, AveragePooling2D, Average

from keras.models import Model

from keras.layers import Wrapper

import keras.backend as K

from keras.optimizers import Adam

from keras import initializers, regularizers, constraints, optimizers, layers

tqdm.pandas()
train_df = pd.read_csv("../input/train.csv")

test_df = pd.read_csv("../input/test.csv")

print("Train shape : ",train_df.shape)

print("Test shape : ",test_df.shape)
train_df[train_df.target==1].head()
contraction_mapping = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not", "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not", "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",  "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would", "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would", "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is", "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as", "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would", "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",  "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is", "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have", "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have","you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have", "you're": "you are", "you've": "you have" }

def clean_contractions(text, mapping):

    specials = ["’", "‘", "´", "`"]

    for s in specials:

        text = text.replace(s, "'")

    text = ' '.join([mapping[t] if t in mapping else t for t in text.split(" ")])

    return text

contraction_patterns = [ (r'won\'t', 'will not'), (r'can\'t', 'cannot'), (r'i\'m', 'i am'), (r'ain\'t', 'is not'), (r'(\w+)\'ll', '\g<1> will'), (r'(\w+)n\'t', '\g<1> not'),

                         (r'(\w+)\'ve', '\g<1> have'), (r'(\w+)\'s', '\g<1> is'), (r'(\w+)\'re', '\g<1> are'), (r'(\w+)\'d', '\g<1> would'), (r'&', 'and'), (r'dammit', 'damn it'), (r'dont', 'do not'), (r'wont', 'will not') ]

def replaceContraction(text):

    patterns = [(re.compile(regex), repl) for (regex, repl) in contraction_patterns]

    for (pattern, repl) in patterns:

        (text, count) = re.subn(pattern, repl, text)

    return text

def clean_text(x):



    x = str(x)

    for punct in "/-'":

        x = x.replace(punct, ' ')

    for punct in '&':

        x = x.replace(punct, f' {punct} ')

    for punct in '?!.,"#$%\'()*+-/:;<=>@[\\]^_`{|}~' + '“”’':

        x = x.replace(punct, '')

    return x

import re



def clean_numbers(x):



    x = re.sub('[0-9]{5,}', ' number ', x)

    x = re.sub('[0-9]{4}', ' number ', x)

    x = re.sub('[0-9]{3}', ' number ', x)

    x = re.sub('[0-9]{2}', ' number ', x)

    return x



punct_mapping = {"‘": "'", "₹": "e", "´": "'", "°": "", "€": "e", "™": "tm", "√": " sqrt ", "×": "x", "²": "2", "—": "-", "–": "-", "’": "'", "_": "-", "`": "'", '“': '"', '”': '"', '“': '"', "£": "e", '∞': 'infinity', 'θ': 'theta', '÷': '/', 'α': 'alpha', '•': '.', 'à': 'a', '−': '-', 'β': 'beta', '∅': '', '³': '3', 'π': 'pi', }

punct = "/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~" + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&'

def clean_special_chars(text, punct, mapping):

    for p in mapping:

        text = text.replace(p, mapping[p])

    

    for p in punct:

        text = text.replace(p, f' {p} ')

    

    specials = {'\u200b': ' ', '…': ' ... ', '\ufeff': '', 'करना': '', 'है': ''}  # Other special characters that I have to deal with in last

    for s in specials:

        text = text.replace(s, specials[s])

    

    return text



mispell_dict = {'advanatges': 'advantages', 'irrationaol': 'irrational' , 'defferences': 'differences','lamboghini':'lamborghini','hypothical':'hypothetical', 'colour': 'color', 'centre': 'center', 'favourite': 'favorite', 'travelling': 'traveling', 'counselling': 'counseling', 'theatre': 'theater', 'cancelled': 'canceled', 'labour': 'labor', 'organisation': 'organization', 'wwii': 'world war 2', 'citicise': 'criticize', 'youtu ': 'youtube ', 'Qoura': 'Quora', 'sallary': 'salary', 'Whta': 'What', 'narcisist': 'narcissist', 'howdo': 'how do', 'whatare': 'what are', 'howcan': 'how can', 'howmuch': 'how much', 'howmany': 'how many', 'whydo': 'why do', 'doI': 'do I', 'theBest': 'the best', 'howdoes': 'how does', 'mastrubation': 'masturbation', 'mastrubate': 'masturbate', "mastrubating": 'masturbating', 'pennis': 'penis', 'Etherium': 'Ethereum', 'narcissit': 'narcissist', 'bigdata': 'big data', '2k17': '2017', '2k18': '2018', 'qouta': 'quota', 'exboyfriend': 'ex boyfriend', 'airhostess': 'air hostess', "whst": 'what', 'watsapp': 'whatsapp', 'demonitisation': 'demonetization', 'demonitization': 'demonetization', 'demonetisation': 'demonetization', 'pokémon': 'pokemon'}

def correct_spelling(x, dic):

    for word in dic.keys():

        x = x.replace(word, dic[word])

    return x
train_df["question_text"] = train_df["question_text"].progress_apply(lambda x: x.lower())

train_df['question_text'] = train_df['question_text'].progress_apply(lambda x: correct_spelling(x, mispell_dict))

train_df["question_text"] = train_df["question_text"].progress_apply(lambda x: clean_numbers(x))

train_df["question_text"] = train_df["question_text"].progress_apply(lambda x: clean_special_chars(x, punct, punct_mapping))





test_df["question_text"] = test_df["question_text"].progress_apply(lambda x: x.lower())

test_df['question_text'] = test_df['question_text'].progress_apply(lambda x: correct_spelling(x, mispell_dict))

test_df["question_text"] = test_df["question_text"].progress_apply(lambda x: clean_numbers(x))

test_df["question_text"] = test_df["question_text"].progress_apply(lambda x: clean_special_chars(x, punct, punct_mapping))

## split to train and val

train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=2018)

## some config values 

embed_size = 300 # how big is each word vector

max_features = 90000 # how many unique words to use (i.e num rows in embedding vector)

maxlen = 50 # max number of words in a question to use

## fill up the missing values

train_X = train_df["question_text"].fillna("_na_").values

val_X = val_df["question_text"].fillna("_na_").values

test_X = test_df["question_text"].fillna("_na_").values
train_X[1]
## Tokenize the sentences

tokenizer = Tokenizer(num_words=max_features, char_level=False, oov_token='<OOV>')

tokenizer.fit_on_texts(list(train_X))



train_XT = tokenizer.texts_to_sequences(train_X)

val_XT = tokenizer.texts_to_sequences(val_X)

test_XT = tokenizer.texts_to_sequences(test_X)

## Pad the sentences 

train_XT = pad_sequences(train_XT, maxlen=maxlen)

val_XT = pad_sequences(val_XT, maxlen=maxlen)

test_XT = pad_sequences(test_XT, maxlen=maxlen)

## Get the target values

train_y = train_df['target'].values

val_y = val_df['target'].values
print(train_X[1])

print(train_XT[1])
coverage = np.zeros((max_features))
EMBEDDING_FILE = '../input/embeddings/glove.840B.300d/glove.840B.300d.txt'

word_index = tokenizer.word_index

def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')

embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE) if o.split(" ")[0] in word_index)

all_embs = np.stack(embeddings_index.values())

emb_mean,emb_std = all_embs.mean(), all_embs.std()

embed_size = all_embs.shape[1]

no_vocab={}

nb_words = min(max_features, len(word_index))

embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))

for word, i in word_index.items():

    if i >= max_features: continue

    embedding_vector = embeddings_index.get(word)

    if embedding_vector is not None:

        coverage[i] +=1

        embedding_matrix[i] = embedding_vector

EMBEDDING_FILE = '../input/embeddings/paragram_300_sl999/paragram_300_sl999.txt'

word_index = tokenizer.word_index

def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')

embeddings_index2 = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE, encoding="utf8", errors='ignore') if len(o)>100 and o.split(" ")[0] in word_index)

all_embs = np.stack(embeddings_index2.values())

emb_mean,emb_std = all_embs.mean(), all_embs.std()

embed_size = all_embs.shape[1]

no_vocab={}

nb_words = min(max_features, len(word_index))

embedding_matrix2 = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))

for word, i in word_index.items():

    if i >= max_features: continue

    embedding_vector = embeddings_index2.get(word)

    if embedding_vector is not None:

        coverage[i] +=1

        embedding_matrix2[i] = embedding_vector
EMBEDDING_FILE = '../input/embeddings/wiki-news-300d-1M/wiki-news-300d-1M.vec'

word_index = tokenizer.word_index

def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')

embeddings_index3 = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE, encoding="utf8", errors='ignore') if len(o)>100 and o.split(" ")[0] in word_index)

all_embs = np.stack(embeddings_index3.values())

emb_mean,emb_std = all_embs.mean(), all_embs.std()

embed_size = all_embs.shape[1]

no_vocab={}

nb_words = min(max_features, len(word_index))

embedding_matrix3 = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))

for word, i in word_index.items():

    if i >= max_features: continue

    embedding_vector = embeddings_index3.get(word)

    if embedding_vector is not None:

        coverage[i] +=1

        embedding_matrix3[i] = embedding_vector
from gensim.models import KeyedVectors

news_path = '../input/embeddings/GoogleNews-vectors-negative300/GoogleNews-vectors-negative300.bin'

embeddings_index4 = KeyedVectors.load_word2vec_format(news_path, binary=True)

no_vocab={}

nb_words = min(max_features, len(word_index))

embedding_matrix4 = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))

for word, i in word_index.items():

    if i >= max_features:

        continue

    try:

        embedding_vector = embeddings_index4.get_vector(word)       

    except (KeyError):

        continue

    

    if embedding_vector is not None:

        coverage[i] +=1

        embedding_matrix4[i] = embedding_vector
unique, counts = np.unique(coverage, return_counts=True)

dict(zip(unique, counts))
embedding_matrix = np.mean([embedding_matrix2,embedding_matrix3,embedding_matrix, embedding_matrix4],axis=0)
embeddings_index = {**embeddings_index2,**embeddings_index3, **embeddings_index}
import operator, gc

def check_coverage(vocab, embeddings_index):

    known_words = {}

    unknown_words = {}

    nb_known_words = 0

    nb_unknown_words = 0

    for word in vocab.keys():

        try:

            known_words[word] = embeddings_index[word]

            nb_known_words += vocab[word]

        except:

            unknown_words[word] = vocab[word]

            nb_unknown_words += vocab[word]

            pass



    print('Found embeddings for {:.2%} of vocab'.format(len(known_words) / len(vocab)))

    print('Found embeddings for  {:.2%} of all text'.format(nb_known_words / (nb_known_words + nb_unknown_words)))

    unknown_words = sorted(unknown_words.items(), key=operator.itemgetter(1))[::-1]



    return unknown_words
no = check_coverage(tokenizer.word_index ,embeddings_index)

no = check_coverage(tokenizer.word_index ,embeddings_index4)

del embeddings_index, embeddings_index2

del embeddings_index3, embeddings_index4

gc.collect()   
no[0:10]
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
class DropConnect(Wrapper):

    def __init__(self, layer, prob=1., **kwargs):

        self.prob = prob

        self.layer = layer

        super(DropConnect, self).__init__(layer, **kwargs)

        if 0. < self.prob < 1.:

            self.uses_learning_phase = True



    def build(self, input_shape):

        if not self.layer.built:

            self.layer.build(input_shape)

            self.layer.built = True

        super(DropConnect, self).build()



    def compute_output_shape(self, input_shape):

        return self.layer.compute_output_shape(input_shape)



    def call(self, x):

        if 0. < self.prob < 1.:

            self.layer.kernel = K.in_train_phase(K.dropout(self.layer.kernel, self.prob), self.layer.kernel)

            self.layer.bias = K.in_train_phase(K.dropout(self.layer.bias, self.prob), self.layer.bias)

        return self.layer.call(x)
def model():

    ad = Adam(lr=0.0015, beta_1=0.9, beta_2=0.999, epsilon=None)

    inp = Input(shape=(maxlen,))

    x = Embedding(max_features, embed_size,trainable=True, weights=[embedding_matrix])(inp)

    x = TimeDistributed(DropConnect(Dense(128, activation="relu"), 0.3))(x)

    x = Bidirectional(CuDNNLSTM(64, return_sequences=True))(x)

    x = Dropout(0.4)(x)

    x = CuDNNLSTM(120, return_sequences=True)(x)

    

    #y = GlobalAveragePooling1D()(x)

    #x = GlobalMaxPool1D()(x)

    x=Attention(maxlen)(x)

    #x = concatenate([x,y])

    x=DropConnect(Dense(32, activation="tanh"), 0.3)(x)

    x = Dense(64, activation="tanh")(x)

    #x = Dropout(0.2)(x)

    x = Dense(1, activation="sigmoid")(x)    

    model = Model(inputs=inp, outputs=x)

    model.compile(loss='binary_crossentropy', optimizer=ad, metrics=['binary_accuracy'])

    return model

model1 = model()

model2 =model()

print(model1.summary())
## Train the model 

model1.fit(train_XT, train_y, batch_size=512, epochs=2, validation_data=(val_XT, val_y))

pred_cnn_val_y1 = model1.predict([val_XT], batch_size=1024, verbose=1)
## Train the model 

model2.fit(train_XT, train_y, batch_size=512, epochs=2, validation_data=(val_XT, val_y))


max_t = 0

max_f1 = 0

for thresh in np.arange(0.1, 0.701, 0.01):

    thresh = np.round(thresh, 2)

    f1 = metrics.f1_score(val_y, (pred_cnn_val_y1>thresh).astype(int))

    #print("F1 score at threshold {0} is {1}".format(thresh, f1))

    if(f1>max_f1):

        max_f1 = f1

        max_t = thresh

print(max_t, max_f1) 
pred_cnn_val_y2 = model2.predict([val_XT], batch_size=1024, verbose=1)

max_t = 0

max_f1 = 0

for thresh in np.arange(0.1, 0.701, 0.01):

    thresh = np.round(thresh, 2)

    f1 = metrics.f1_score(val_y, (pred_cnn_val_y2>thresh).astype(int))

    #print("F1 score at threshold {0} is {1}".format(thresh, f1))

    if(f1>max_f1):

        max_f1 = f1

        max_t = thresh

print(max_t, max_f1) 
pred_cnn_val_y = pred_cnn_val_y1*0.5 + pred_cnn_val_y2*0.5

max_t = 0

max_f1 = 0

for thresh in np.arange(0.1, 0.701, 0.01):

    thresh = np.round(thresh, 2)

    f1 = metrics.f1_score(val_y, (pred_cnn_val_y>thresh).astype(int))

    #print("F1 score at threshold {0} is {1}".format(thresh, f1))

    if(f1>max_f1):

        max_f1 = f1

        max_t = thresh

print(max_t, max_f1) 

    
pred_cnn_test_y1 = model1.predict([test_XT], batch_size=1024, verbose=1)

pred_cnn_test_y2 = model2.predict([test_XT], batch_size=1024, verbose=1)
pred_cnn_test_y = pred_cnn_test_y1*0.5 + pred_cnn_test_y2*0.5

pred_test_y = (pred_cnn_test_y>max_t).astype(int)

out_df = pd.DataFrame({"qid":test_df["qid"].values})

out_df['prediction'] = pred_test_y

out_df.to_csv("submission.csv", index=False)