# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from multiprocessing import Pool

import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.

import random

from tqdm import tqdm



from sklearn.model_selection import train_test_split

from sklearn import metrics

from gensim.models import KeyedVectors

import torch



from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, Conv1D, GRU, CuDNNGRU, CuDNNLSTM, BatchNormalization

from keras.layers import Bidirectional, GlobalMaxPool1D, MaxPool1D, Add, Flatten, Layer

from keras.layers import GlobalAveragePooling1D, GlobalMaxPooling1D, concatenate, SpatialDropout1D, add, Reshape

from keras.models import Model, load_model

from keras import initializers, regularizers, constraints, optimizers, layers, callbacks, Sequential

from keras import backend as k

from keras.engine import InputSpec, Layer

from keras.optimizers import Adam

from keras.callbacks import *



import gensim

from gensim.models import Word2Vec

from textblob import TextBlob

import re



from nltk.stem import PorterStemmer

from nltk.stem.lancaster import LancasterStemmer

from nltk.stem import SnowballStemmer
ps = PorterStemmer()

ls = LancasterStemmer()

ss = SnowballStemmer('english')
train = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/train.csv')

test = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/test.csv')
identity_columns = ['male', 'female', 'homosexual_gay_or_lesbian', 'christian', 'jewish', 'muslim', 'black', 'white',

                    'psychiatric_or_mental_illness']

for col in identity_columns+['target']:

    train[col] = np.where(train[col]>=0.5, True, False)
def seed_everything(seed=1234):

    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.deterministic = True

    from tensorflow import set_random_seed

    set_random_seed(2)



seed_everything()
# Define some Global Variables

max_features = 150000 # Maximum Number of words we want to include in our dictionary

maxlen = 256 # No of words in question we want to create a sequence with

embed_size = 302# Size of word to vec embedding we are using
# Some preprocesssing that will be common to all the text classification methods you will see. 

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

    
contraction_mapping = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not", "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not", "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",  "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would", "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would", "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is", "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as", "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would", "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",  "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is", "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have", "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have","you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have", "you're": "you are", "you've": "you have" }
def clean_contractions(text, mapping):

    specials = ["’", "‘", "´", "`"]

    for s in specials:

        text = text.replace(s, "'")

    text = ' '.join([mapping[t] if t in mapping else t for t in text.split(" ")])

    return text
def load_and_proc():

    # split into train and validation

    train['comment_text'] = train['comment_text'].apply(lambda x : clean_text(x))

    test['comment_text'] = test['comment_text'].apply(lambda x : clean_text(x))

    train['comment_text'] = train['comment_text'].apply(lambda x : clean_numbers(x))

    test['comment_text'] = test['comment_text'].apply(lambda x : clean_numbers(x))

    train['comment_text'] = train['comment_text'].apply(lambda x : clean_contractions(x,contraction_mapping))

    test['comment_text'] = test['comment_text'].apply(lambda x : clean_contractions(x,contraction_mapping))

    

    df_train, df_valid = train_test_split(train, test_size=0.33)

    df_test = test

    

    df_train.loc[:,'set_'] = 'train'

    df_valid.loc[:,'set_'] = 'valid'

    df_test.loc[:,'set_'] = 'test'



    set_indices = df_train.loc[:,'set_']

    set_indices.append(df_valid.loc[:,'set_'])

    set_indices.append(df_test.loc[:,'set_'])



    y_train = np.asarray(df_train['target'])

    y_valid = np.asarray(df_valid['target'])



    set_indices_label = df_train.loc[:,'set_']

    set_indices_label = set_indices_label.append(df_valid.loc[:,'set_'])

    

    X_train = df_train['comment_text'].fillna('_##_').values

    X_val = df_valid['comment_text'].fillna('_##_').values

    X_test = df_test['comment_text'].fillna('_##_').values

    

    all_text = list(X_train)

    all_text.append(list(X_val))

    all_text.append(list(X_test))

    

    tokenizer = Tokenizer(num_words=max_features)

    tokenizer.fit_on_texts(all_text)

    

    X_train = tokenizer.texts_to_sequences(X_train)

    X_val = tokenizer.texts_to_sequences(X_val)

    X_test = tokenizer.texts_to_sequences(X_test)

    

    X_train = pad_sequences(X_train, maxlen=maxlen)

    X_val = pad_sequences(X_val, maxlen=maxlen)

    X_test = pad_sequences(X_test, maxlen=maxlen)

    

    #shuffling the data

    np.random.seed(2019)

    train_idx = np.random.permutation(len(X_train))

    valid_idx = np.random.permutation(len(X_val))

    print(type(X_train))

    print(type(y_train))

    X_train = X_train[train_idx]

    X_val = X_val[valid_idx]

    Y_train = y_train[train_idx]

    Y_val = y_valid[valid_idx]

    

    return X_train, X_val, X_test, Y_train, Y_val, tokenizer.word_index
X_train, X_val,X_test, Y_train, Y_val, word_index = load_and_proc()
#word_index.items()
import gc

gc.collect()

del train

del test
def load_glove():

    EMBEDDING_FILE = '../input/emb-model/glove.840B.300d.txt'

    def get_coef(word, *arr): return word, np.asarray(arr, dtype=np.float32)[:300]

    embeddings_index = dict(get_coef(*o.split(" ")) for o in open(EMBEDDING_FILE))

    

    return embeddings_index
def load_paragram():

    EMBEDDING_FILE = '../input/paragram-300-sl999/paragram_300_sl999.txt'

    def get_coef(word, *arr): return word, np.asarray(arr, dtype=np.float32)[:300]

    embeddings_index = dict(get_coef(*o.split(" ")) for o in open(EMBEDDING_FILE, errors='ignore', encoding='utf-8'))

    

    return embeddings_index
def load_fasttext():

    EMBEDDING_FILE = '../input/fasttext-wiki-news-300d-1m/wiki-news-300d-1M.vec'

    def get_coef(word, *arr): return word, np.asarray(arr, dtype=np.float32)

    embeddings_index = dict(get_coef(*o.split(" ")) for o in open(EMBEDDING_FILE) if len(o) > 100)

    

    return embeddings_index
def create_emb(word_index, embeddings_index):

    #embedding_index_paragram = load_paragram()

    #embeddings_index = load_glove()

    #all_embs_glove = np.stack(embeddings_index.values())

    #all_embs_paragram = np.stack(embedding_index_paragram.values())

    emb_mean,emb_std = -0.005838499,0.48782197

    #final_emb = np.concatenate([all_embs_glove,all_embs_paragram ])

    #embed_size = all_embs_glove.shape[1]

    nb_words = min(max_features, len(word_index))

    #embeddings_index.update(embedding_index_paragram)

    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))

    count_found = nb_words

    for word, i in tqdm(word_index.items()):

        if i >= max_features: continue

        embedding_vector = embeddings_index.get(word)

        word_sent = TextBlob(word).sentiment

        # Extra information we are passing to our embeddings

        extra_embed = [word_sent.polarity, word_sent.subjectivity]

        if embedding_vector is not None:

            embedding_matrix[i] = np.append(embedding_vector, extra_embed)

            continue

        key = word.lower()

        embedding_vector = embeddings_index.get(key)

        if embedding_vector is not None:

            embedding_matrix[i] = np.append(embedding_vector, extra_embed)

            continue

        key = word.upper()

        embedding_vector = embeddings_index.get(key)

        if embedding_vector is not None:

            embedding_matrix[i] = np.append(embedding_vector, extra_embed)

            continue

        key = word.capitalize()

        embedding_vector = embeddings_index.get(key)

        if embedding_vector is not None:

            embedding_matrix[i] = np.append(embedding_vector, extra_embed)

            continue

        key = ps.stem(word)

        embedding_vector = embeddings_index.get(key)

        if embedding_vector is not None:

            embedding_matrix[i] = np.append(embedding_vector, extra_embed)

            continue

        key = ls.stem(word)

        embedding_vector = embeddings_index.get(key)

        if embedding_vector is not None:

            embedding_matrix[i] = np.append(embedding_vector, extra_embed)

            continue

        key = ss.stem(word)

        embedding_vector = embeddings_index.get(key)

        if embedding_vector is not None:

            embedding_matrix[i] = np.append(embedding_vector, extra_embed)

            continue

        embedding_matrix[i,300:] = extra_embed

        count_found -= 1

    print('Total words ', nb_words)

    print("Got embedding for ",count_found," words.")

    return embedding_matrix
#np.stack(embeddings_index.values())

gc.collect()
#paragram_embedding = create_emb(word_index,load_paragram())

glove_embedding = create_emb(word_index,load_glove())
#final_matrix = np.concatenate([glove_embedding,paragram_embedding], 1)

final_matrix = glove_embedding
gc.collect()
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
class Attention(Layer):

    

    def __init__(self, 

                W_regulariser=None,b_regulariser=None,u_regulariser=None,

                W_constraint=None,b_constraint=None,u_constraint=None,

                biase=True, **kwargs):

        self.support_masking = True

        self.initializer = initializers.get('glorot_uniform')

        

        self.W_regulariser = regularizers.get(W_regulariser)

        self.b_regulariser = regularizers.get(b_regulariser)

        self.u_regulariser = regularizers.get(u_regulariser)

        

        self.W_constraint = constraints.get(W_constraint)

        self.b_constraint = constraints.get(b_constraint)

        self.u_constraint = constraints.get(u_constraint)

        

        self.biase = biase

        super(Attention,self).__init__(**kwargs)

        

    def build(self, input_shape):

        assert len(input_shape) == 3

        self.W = self.add_weight(shape=(input_shape[-1],input_shape[-1],),

                                 initializer=self.initializer,

                                 name='{}_W'.format(self.name),

                                 regularizer=self.W_regulariser,

                                 constraint=self.W_constraint)

        if self.biase:

            self.b =  self.add_weight(shape=(input_shape[-1],),

                                      initializer = 'zero',

                                      name='{}_b'.format(self.name),

                                      regularizer=self.b_regulariser,

                                      constraint=self.b_constraint

                                     )

        self.u = self.add_weight(shape=(input_shape[-1],),

                                 initializer=self.initializer,

                                 name='{}_u'.format(self.name),

                                 regularizer=self.u_regulariser,

                                 constraint=self.u_constraint)

        super(Attention, self).build(input_shape)

    

    def compute_mask(self, input, input_mask=None):

        # do not pass the mask to the next layers

        return None

    

    def call(self, x, mask=None):

        uit = dot_product(x, self.W)



        if self.biase:

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

        
def model_lstm_du(final_embedding):

    inp = Input(shape=(maxlen,))

    x = Embedding(max_features, embed_size, weights=[final_embedding], trainable = False)(inp)

    x = SpatialDropout1D(0.2)(x)

    

    x1 = Bidirectional(CuDNNLSTM(256, return_sequences=True))(x)

    x1 = Bidirectional(CuDNNLSTM(128, return_sequences=True))(x1)

    #x1 = Attention()(x1)

    x1 = Conv1D(64, kernel_size=3, padding = "valid", kernel_initializer = "he_uniform")(x1)

    

    x2 = Bidirectional(CuDNNLSTM(256, return_sequences=True))(x)

    x2 = Bidirectional(CuDNNGRU(128, return_sequences=True))(x2)

    #x2 = Attention()(x2)

    x2 = Conv1D(64, kernel_size=4, padding = "valid", kernel_initializer = "he_uniform")(x2)

    

    avg_pool1 = GlobalAveragePooling1D()(x1)

    max_pool1 = GlobalMaxPooling1D()(x1)

    

    avg_pool2 = GlobalAveragePooling1D()(x2)

    max_pool2 = GlobalMaxPooling1D()(x2)

    

    conc = concatenate([avg_pool1, max_pool1,avg_pool2, max_pool2])

    conc = Dense(128, activation='relu')(conc)

    conc = Dense(64, activation='relu')(conc)

    conc = Dropout(0.1)(conc)

    outp = Dense(1, activation='sigmoid')(conc)

    model = Model(inputs = inp, outputs = outp)

    model.compile(loss='binary_crossentropy', optimizer=Adam(lr = 1e-3, decay=0.0), metrics=['accuracy'])

    return model
model = model_lstm_du(final_matrix)

model.summary()
gc.collect()
def train_pred(model, epoch=2):

    filepath="weights_best.h5"

    checkpoint = ModelCheckpoint(filepath=filepath, monitor='val_loss', verbose=2, save_best_only=True, mode='min' )

    #reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.0001, patience=1, verbose=2, min_lr=0.0001)

    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.0001, verbose=1,patience=1, mode='min')

    callbacks = [checkpoint,early_stopping]#, reduce_lr]

    

    #for e in range(epoch):

    model.fit(x=X_train,y=Y_train, batch_size=512,epochs=epoch, callbacks=callbacks, validation_data=(X_val, Y_val))

    model.load_weights(filepath)

    pred_val_y = model.predict([X_val], batch_size=1024, verbose=0)

    pred_test_y = model.predict([X_test], batch_size=1024, verbose=0)

    return pred_val_y, pred_test_y
pred_val_y, pred_test_y = train_pred(model, epoch=8)
submission = pd.read_csv("../input/jigsaw-unintended-bias-in-toxicity-classification/sample_submission.csv")

submission['prediction'] = pred_test_y

submission.to_csv('submission.csv', index=False)