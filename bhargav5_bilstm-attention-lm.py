# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input/quora-insincere-questions-classification/"))

# Any results you write to the current directory are saved as output.
from sklearn.model_selection import train_test_split
from sklearn import metrics
import re
import gc

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, CuDNNLSTM, Embedding, Dropout, Activation, CuDNNGRU, Conv1D, GRU
from keras.layers import Bidirectional, GlobalMaxPool1D, GlobalAvgPool1D, BatchNormalization
from keras.optimizers import Adam
from keras.models import Model
from keras import backend as K
from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints, optimizers, layers
from keras.callbacks import ModelCheckpoint
import keras.utils as ku
embed_size = 300 # how big is each word vector
max_features = 100000 # how many unique words to use (i.e num rows in embedding vector)
maxlen = 72 # max number of words in a question to use
SEED = 2019
from tqdm import tqdm
tqdm.pandas()
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

def clean_numbers(x):
    x = re.sub('[0-9]{5,}', '#####', x)
    x = re.sub('[0-9]{4}', '####', x)
    x = re.sub('[0-9]{3}', '###', x)
    x = re.sub('[0-9]{2}', '##', x)
    return x

mispell_dict = {"aren't" : "are not",
"can't" : "cannot",
"couldn't" : "could not",
"didn't" : "did not",
"doesn't" : "does not",
"don't" : "do not",
"hadn't" : "had not",
"hasn't" : "has not",
"haven't" : "have not",
"he'd" : "he would",
"he'll" : "he will",
"he's" : "he is",
"i'd" : "I would",
"i'd" : "I had",
"i'll" : "I will",
"i'm" : "I am",
"isn't" : "is not",
"it's" : "it is",
"it'll":"it will",
"i've" : "I have",
"let's" : "let us",
"mightn't" : "might not",
"mustn't" : "must not",
"shan't" : "shall not",
"she'd" : "she would",
"she'll" : "she will",
"she's" : "she is",
"shouldn't" : "should not",
"that's" : "that is",
"there's" : "there is",
"they'd" : "they would",
"they'll" : "they will",
"they're" : "they are",
"they've" : "they have",
"we'd" : "we would",
"we're" : "we are",
"weren't" : "were not",
"we've" : "we have",
"what'll" : "what will",
"what're" : "what are",
"what's" : "what is",
"what've" : "what have",
"where's" : "where is",
"who'd" : "who would",
"who'll" : "who will",
"who're" : "who are",
"who's" : "who is",
"who've" : "who have",
"won't" : "will not",
"wouldn't" : "would not",
"you'd" : "you would",
"you'll" : "you will",
"you're" : "you are",
"you've" : "you have",
"'re": " are",
"wasn't": "was not",
"we'll":" will",
"didn't": "did not",
"tryin'":"trying"}

def _get_mispell(mispell_dict):
    mispell_re = re.compile('(%s)' % '|'.join(mispell_dict.keys()))
    return mispell_dict, mispell_re

mispellings, mispellings_re = _get_mispell(mispell_dict)
def replace_typical_misspell(text):
    def replace(match):
        return mispellings[match.group(0)]
    return mispellings_re.sub(replace, text)
def data_preparation():
    train_df = pd.read_csv("../input/quora-insincere-questions-classification/train.csv")
    test_df = pd.read_csv("../input/quora-insincere-questions-classification/test.csv")
    
    print("Train shape : ",train_df.shape)
    print("Test shape : ",test_df.shape)
    
    # lower
    train_df["question_text"] = train_df["question_text"].progress_apply(lambda x: x.lower())
    test_df["question_text"] = test_df["question_text"].progress_apply(lambda x: x.lower())
    
    # Clean the text
    train_df["question_text"] = train_df["question_text"].progress_apply(lambda x: clean_text(x))
    test_df["question_text"] = test_df["question_text"].progress_apply(lambda x: clean_text(x))
    
    # Clean numbers
    train_df["question_text"] = train_df["question_text"].progress_apply(lambda x: clean_numbers(x))
    test_df["question_text"] = test_df["question_text"].progress_apply(lambda x: clean_numbers(x))
    
    # Clean speelings
    train_df["question_text"] = train_df["question_text"].progress_apply(lambda x: replace_typical_misspell(x))
    test_df["question_text"] = test_df["question_text"].progress_apply(lambda x: replace_typical_misspell(x))
    
    ## fill up the missing values
    data_df = pd.concat([train_df["question_text"], test_df["question_text"]])
    data = data_df.fillna("_##_").values
    train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=2019)
    train_X = train_df["question_text"].fillna("_##_").values
    val_X = val_df["question_text"].fillna("_##_").values
    test_X = test_df["question_text"].fillna("_##_").values
    print(data.shape)

    ## Tokenize the sentences
    print("Tokenization...")
    tokenizer = Tokenizer(num_words=max_features)
    tokenizer.fit_on_texts(list(data))
    total_words = len(tokenizer.word_index) + 1
    
    data = tokenizer.texts_to_sequences(data)
    train_X = tokenizer.texts_to_sequences(train_X)
    val_X = tokenizer.texts_to_sequences(val_X)
    test_X = tokenizer.texts_to_sequences(test_X)
    
    max_seq_len = max([len(x) for x in data])

    ## Pad the sentences 
    data = pad_sequences(data, maxlen=max_seq_len)
    train_X = pad_sequences(train_X, maxlen=max_seq_len)
    val_X = pad_sequences(val_X, maxlen=max_seq_len)
    test_X = pad_sequences(test_X, maxlen=max_seq_len)
    
    train_y = train_df['target'].values
    val_y = val_df['target'].values
    #Generate predictors and labels
    predictors, labels = data[:,:-1], data[:,-1]
    
    return predictors, labels, max_seq_len, total_words, tokenizer.word_index, train_X, train_y, val_X, val_y, test_X
def load_glove(word_index):
    EMBEDDING_FILE = '../input/quora-insincere-questions-classification/embeddings/glove.840B.300d/glove.840B.300d.txt'
    def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
    embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE))

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
    EMBEDDING_FILE = '../input/quora-insincere-questions-classification/embeddings/paragram_300_sl999/paragram_300_sl999.txt'
    def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
    embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE, encoding="utf8", errors='ignore') if len(o)>100)

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

def load_fasttext(word_index):
    EMBEDDING_FILE = '../input/quora-insincere-questions-classification/embeddings/wiki-news-300d-1M/wiki-news-300d-1M.vec'
    def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
    embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE) if len(o)>100)

    all_embs = np.stack(embeddings_index.values())
    emb_mean,emb_std = all_embs.mean(), all_embs.std()
    embed_size = all_embs.shape[1]


    nb_words = min(max_features, len(word_index))
    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
    for word, i in word_index.items():
        if i >= max_features: continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None: embedding_matrix[i] = embedding_vector
    
    return embedding_matrix
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
predictors, label, max_sequence_len, total_words, word_index, train_X, train_y, val_X, val_y, test_X = data_preparation()
print(len(predictors[0]), max_sequence_len, total_words, len(word_index), train_X.shape, val_X.shape)
#embedding_matrix = load_glove(word_index)
embedding_matrix = np.zeros((max_features, embed_size))
'''
inp = Input(shape=(max_sequence_len-1,), dtype='float32')
x = Embedding(max_features, embed_size, weights=[embedding_matrix], trainable=True)(inp)
x = Bidirectional(CuDNNGRU(128, return_sequences=True))(x)
x21 = Attention(step_dim=max_sequence_len-1)(x)
x1 = Bidirectional(CuDNNGRU(64, return_sequences=True))(x)
x2 = Attention(step_dim=max_sequence_len-1)(x1)
conc = layers.Concatenate()([x21, x2])
conc = Dropout(0.1)(conc)
out = Dense(total_words, activation='softmax')(conc)
'''
#model = Model(inputs=inp, outputs=out)
#model.compile(loss="sparse_categorical_crossentropy", optimizer='adam', metrics=['accuracy'])
#model.load_weights("../input/10epoch-lm/pre_trained_weights_5_to_10.h5")
#callback=ModelCheckpoint("pre_trained_weights_10_to_15.h5", monitor='val_loss', verbose=1, save_best_only=False, save_weights_only=True, mode='auto', period=1)
#model.fit(x=predictors, y=label, epochs=5, batch_size=512, shuffle=True, callbacks=[callback])
inp = Input(shape=(max_sequence_len-1,), dtype='float32')
x = Embedding(max_features, embed_size, weights=[embedding_matrix], trainable=True)(inp)
x = Bidirectional(CuDNNGRU(128, return_sequences=True))(x)
x21 = Attention(step_dim=max_sequence_len-1)(x)
x1 = Bidirectional(CuDNNGRU(64, return_sequences=True))(x)
x2 = Attention(step_dim=max_sequence_len-1)(x1)
conc = layers.Concatenate()([x21, x2])
conc = Dropout(0.1)(conc)
#out = Dense(total_words, activation='softmax')(conc)
o = Dense(32, activation='relu', name='newd')(conc)
o = Dropout(0.1, name="new_drop")(o)
#o = BatchNormalization()(o)
output = Dense(1, activation='sigmoid', name='out')(o)
model1 = Model(inputs=inp, outputs=output)
optimizer = Adam(lr=0.0001)
model1.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=['accuracy'])
model1.load_weights("../input/10epoch-lm/pre_trained_weights_5_to_10.h5", by_name=True)
train_X = train_X[:,1:]
val_X = val_X[:, 1:]
test_X = test_X[:, 1:]
train_X[0]
model1.fit(train_X, train_y, batch_size=512, epochs=2, validation_data=(val_X, val_y))
pred_noemb_val_y = model1.predict([val_X], batch_size=1024, verbose=1)
for thresh in np.arange(0.1, 0.501, 0.01):
    thresh = np.round(thresh, 2)
    print("F1 score at threshold {0} is {1}".format(thresh, metrics.f1_score(val_y, (pred_noemb_val_y>thresh).astype(int))))
pred_noemb_test_y = model1.predict([test_X], batch_size=1024, verbose=1)
test_df = pd.read_csv("../input/quora-insincere-questions-classification/test.csv")
pred_test_y = (pred_noemb_test_y>0.34).astype(int)
out_df = pd.DataFrame({"qid":test_df["qid"].values})
out_df['prediction'] = pred_test_y
out_df.to_csv("submission.csv", index=False)
