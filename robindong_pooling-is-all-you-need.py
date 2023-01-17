# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os



if not os.path.exists('obj'):

    os.mkdir('obj')

if not os.path.exists('models'):

    os.mkdir('models')

    

print(os.listdir('./models/'))

print(os.listdir("../input"))

print(os.listdir('../input/embeddings'))



# Any results you write to the current directory are saved as output.


import re

import sys

import pickle



from sklearn.model_selection import train_test_split

from tensorflow.keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from tqdm import tqdm

tqdm.pandas()



max_features = 99999

MAX_TEXT_LENGTH = 70



def clean_text(x):

    puncts = [',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&', '/', '[', ']', '>', '%', '=', '#', '*', '+', '\\', '•',  '~',

              '@', '£', '·', '_', '{', '}', '©', '^', '®', '`',  '<', '→', '°', '€', '™', '›',  '♥', '←', '×', '§', '″', '′', 'Â', '█', '½', 'à',

              '…', '“', '★', '”', '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶', '↑', '±', '¿', '▾', '═', '¦', '║', '―', '¥', '▓', '—', '‹', '─',

              '▒', '：', '¼', '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲', 'è', '¸', '¾', 'Ã', '⋅', '‘', '∞',

              '∙', '）', '↓', '、', '│', '（', '»', '，', '♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø', '¹', '≤', '‡', '√', ]

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

        "tryin'": "trying" }



def _get_mispell(mispell_dict):

    mispell_re = re.compile('(%s)' % '|'.join(mispell_dict.keys()))

    return mispell_dict, mispell_re



mispellings, mispellings_re = _get_mispell(mispell_dict)



def replace_typical_misspell(text):

    def replace(match):

        return mispellings[match.group(0)]

    return mispellings_re.sub(replace, text)

    

def save_obj(obj, name): 

    with open('obj/' + name + '.pkl', 'wb') as file:

        pickle.dump(obj, file, pickle.HIGHEST_PROTOCOL)

        

def load_obj(name):

    with open('obj/' + name + '.pkl', 'rb') as file:

        return pickle.load(file)



def load_data(data_path):

    data = pd.read_csv(data_path)

    data['question_text'] = data['question_text'].progress_apply(lambda x: x.lower())

    data['question_text'] = data['question_text'].progress_apply(lambda x: clean_text(x))

    data['question_text'] = data['question_text'].progress_apply(lambda x: clean_numbers(x))

    data['question_text'] = data['question_text'].progress_apply(lambda x: replace_typical_misspell(x))

    data['question_text'] = data['question_text'].fillna("_##_").values

    return data



def load_glove(word_index):

    EMBEDDING_FILE = '../input/embeddings/glove.840B.300d/glove.840B.300d.txt'

    def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')[:300]

    embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE))

    

    all_embs = np.stack(list(embeddings_index.values()))

    emb_mean,emb_std = all_embs.mean(), all_embs.std()

    embed_size = all_embs.shape[1]



    # word_index = tokenizer.word_index

    nb_words = min(max_features, len(word_index))

    # Why random embedding for OOV? what if use mean?

    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))

    #embedding_matrix = np.random.normal(emb_mean, 0, (nb_words, embed_size)) # std 0

    for word, i in word_index.items():

        if i >= max_features: continue

        embedding_vector = embeddings_index.get(word)

        if embedding_vector is not None: embedding_matrix[i] = embedding_vector

            

    return embedding_matrix 

    

def load_fasttext(word_index):    

    EMBEDDING_FILE = '../input/embeddings/wiki-news-300d-1M/wiki-news-300d-1M.vec'

    def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')

    embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE) if len(o)>100)



    all_embs = np.stack(list(embeddings_index.values()))

    emb_mean,emb_std = all_embs.mean(), all_embs.std()

    embed_size = all_embs.shape[1]



    # word_index = tokenizer.word_index

    nb_words = min(max_features, len(word_index))

    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))

    #embedding_matrix = np.random.normal(emb_mean, 0, (nb_words, embed_size))

    for word, i in word_index.items():

        if i >= max_features: continue

        embedding_vector = embeddings_index.get(word)

        if embedding_vector is not None: embedding_matrix[i] = embedding_vector



    return embedding_matrix



def load_para(word_index):

    EMBEDDING_FILE = '../input/embeddings/paragram_300_sl999/paragram_300_sl999.txt'

    def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')

    embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE, encoding="utf8", errors='ignore') if len(o)>100)



    all_embs = np.stack(list(embeddings_index.values()))

    emb_mean,emb_std = all_embs.mean(), all_embs.std()

    embed_size = all_embs.shape[1]



    # word_index = tokenizer.word_index

    nb_words = min(max_features, len(word_index))

    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))

    #embedding_matrix = np.random.normal(emb_mean, 0, (nb_words, embed_size))

    for word, i in word_index.items():

        if i >= max_features: continue

        embedding_vector = embeddings_index.get(word)

        if embedding_vector is not None: embedding_matrix[i] = embedding_vector

    

    return embedding_matrix



data = load_data('../input/train.csv')

data = data.sample(frac = 1.0).reset_index(drop = True)



tokenizer = Tokenizer(num_words = max_features)

tokenizer.fit_on_texts(data['question_text'])



train_df, val_df = train_test_split(data, test_size = 0.1, stratify = data['target'])



train_x = train_df['question_text']

train_y = train_df['target'].values

val_x = val_df['question_text']

val_y = val_df['target'].values



train_x = tokenizer.texts_to_sequences(train_x)

val_x = tokenizer.texts_to_sequences(val_x)



train_x = pad_sequences(train_x, MAX_TEXT_LENGTH)

val_x = pad_sequences(val_x, MAX_TEXT_LENGTH)



embedding_glove = load_glove(tokenizer.word_index)

embedding_fasttext = load_fasttext(tokenizer.word_index)

embedding_para = load_para(tokenizer.word_index)



save_obj(embedding_glove, 'glove')

save_obj(embedding_fasttext, 'fasttext')

save_obj(embedding_para, 'para')

save_obj(tokenizer, 'tokenizer')

save_obj(train_x, 'train_x')

save_obj(val_x, 'val_x')

save_obj(train_y, 'train_y')

save_obj(val_y, 'val_y')
os.listdir('obj/')
import os

import time

import math

import keras

import argparse



import tensorflow as tf



from keras.layers import Dense, Bidirectional, Dropout

from keras.layers import Activation, CuDNNGRU, Conv1D, Input, Conv2D

from keras.layers import GlobalMaxPooling1D, GlobalAveragePooling1D, MaxPooling2D

from keras.layers import Concatenate, concatenate, Embedding, Flatten

from keras.layers import SpatialDropout1D, Reshape, MaxPool2D, AveragePooling2D

from keras.optimizers import SGD, RMSprop, Adam

from keras.callbacks import Callback

from keras.models import load_model, Model

from keras.utils import Sequence

from keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.preprocessing.text import Tokenizer

from keras.callbacks import LearningRateScheduler

from keras import backend as K



from tqdm import tqdm

from sklearn.model_selection import train_test_split
EMBEDDING_LENGTH = 300

MAX_TEXT_LENGTH = 70

BATCH_SIZE = 256



MODEL_PATH = './models/'



MAX_EPOCHS = 4
def step_decay_func(args):

    def step_decay(epoch):

        initial_lrate = args.lr

        decay = 0.9

        lrate = initial_lrate * pow(decay, epoch)

        print('epoch:', epoch, ' lr:', lrate)

        return lrate

    return step_decay



def precision(y_true, y_pred):

    """Precision metric.

    Only computes a batch-wise average of precision.

    Computes the precision, a metric for multi-label classification of

    how many selected items are relevant.

    """

    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))

    precision = true_positives / (predicted_positives + K.epsilon())

    return precision



def recall(y_true, y_pred):

    """Recall metric.

    Only computes a batch-wise average of recall.

    Computes the recall, a metric for multi-label classification of

    how many relevant items are selected.

    """

    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))

    recall = true_positives / (possible_positives + K.epsilon())

    return recall



def fbeta_score(y_true, y_pred, beta=1):

    """Computes the F score.

    The F score is the weighted harmonic mean of precision and recall.

    Here it is only computed as a batch-wise average, not globally.

    This is useful for multi-label classification, where input samples can be

    classified as sets of labels. By only using accuracy (precision) a model

    would achieve a perfect score by simply assigning every class to every

    input. In order to avoid this, a metric should penalize incorrect class

    assignments as well (recall). The F-beta score (ranged from 0.0 to 1.0)

    computes this, as a weighted mean of the proportion of correct class

    assignments vs. the proportion of incorrect class assignments.

    With beta = 1, this is equivalent to a F-measure. With beta < 1, assigning

    correct classes becomes more important, and with beta > 1 the metric is

    instead weighted towards penalizing incorrect class assignments.

    """ 

    if beta < 0:

        raise ValueError('The lowest choosable beta is zero (only precision).')

        

    # If there are no true positives, fix the F score at 0 like sklearn.

    if K.sum(K.round(K.clip(y_true, 0, 1))) == 0:

        return 0



    p = precision(y_true, y_pred)

    r = recall(y_true, y_pred)

    bb = beta ** 2

    fbeta_score = (1 + bb) * (p * r) / (bb * p + r + K.epsilon())

    return fbeta_score



def fmeasure(y_true, y_pred):

    return fbeta_score(y_true, y_pred, beta = 1)
class CheckpointCallback(Callback):

    def __init__(self, path):

        self.path = path



    def on_epoch_end(self, epoch, logs = None):

        self.model.save(self.path + 'quora-' + str(epoch) + '.h5')

        

class DataSequence(Sequence):

    def __init__(self, train_x, train_y):

        pid = os.getpid()

        now = time.time()

        now = (now - int(now)) # Get fractional-part

        now = int(now * 1000)

        np.random.seed(pid * now)

        trn_idx = np.random.permutation(len(train_x))

        self.train_x = train_x[trn_idx]

        self.train_y = train_y[trn_idx]



    def __len__(self):

        n_batches = math.ceil(len(self.train_x) / BATCH_SIZE)

        return n_batches



    def __getitem__(self, index):

        texts = self.train_x[index * BATCH_SIZE : (index + 1) * BATCH_SIZE]

        return texts, self.train_y[index * BATCH_SIZE : (index + 1) * BATCH_SIZE]
def model_gru_pool(args, embedding_matrix):

    regular = keras.regularizers.l1(args.l1)



    inp = Input(shape = (MAX_TEXT_LENGTH,))



    net = Embedding(embedding_matrix.shape[0], EMBEDDING_LENGTH, weights=[embedding_matrix], trainable = True)(inp)

    net = SpatialDropout1D(args.dropout)(net)



    net = Bidirectional(CuDNNGRU(128, return_sequences = True, kernel_regularizer = regular, recurrent_regularizer = regular))(net)

    net = SpatialDropout1D(args.dropout)(net)

    avg_pool1 = GlobalAveragePooling1D(data_format = 'channels_last')(net)

    max_pool1 = GlobalMaxPooling1D(data_format = 'channels_last')(net)



    net = Bidirectional(CuDNNGRU(96, return_sequences = True, kernel_regularizer = regular, recurrent_regularizer = regular))(net)

    net = SpatialDropout1D(args.dropout)(net)

    avg_pool2 = GlobalAveragePooling1D(data_format = 'channels_last')(net)

    max_pool2 = GlobalMaxPooling1D(data_format = 'channels_last')(net)



    net = Bidirectional(CuDNNGRU(64, return_sequences = True, kernel_regularizer = regular, recurrent_regularizer = regular))(net)

    net = SpatialDropout1D(args.dropout)(net)

    avg_pool3 = GlobalAveragePooling1D(data_format = 'channels_last')(net)

    max_pool3 = GlobalMaxPooling1D(data_format = 'channels_last')(net)



    conc = concatenate([avg_pool1, max_pool1, avg_pool2, max_pool2, avg_pool3, max_pool3])

    conc = Dropout(args.dropout)(conc)



    conc = Dense(512, activation = 'relu')(conc)

    conc = Dropout(args.dropout)(conc)

    

    outp = Dense(1, activation = 'sigmoid')(conc)



    model = Model(inputs = inp, outputs = outp)

    return model



def build_model(args, embedding_matrix):

    model = model_gru_pool(args, embedding_matrix)

    #optimizer = SGD(momentum=0.0, nesterov=True)

    optimizer = Adam()

    model.compile(loss = 'binary_crossentropy',

                  optimizer = optimizer,

                  metrics = [fmeasure])

    return model
from collections import namedtuple



def main():

    glove_embeddings = load_obj('glove')

    paragram_embeddings = load_obj('para')

    fasttext_embeddings = load_obj('fasttext')



    embedding_matrix = np.mean([glove_embeddings, paragram_embeddings, fasttext_embeddings], axis=0)



    train_x = load_obj('train_x')

    val_x = load_obj('val_x')

    train_y = load_obj('train_y')

    val_y = load_obj('val_y')



    mg = DataSequence(train_x, train_y)

    args = namedtuple('Argument', 'lr l1 dropout')

    args.lr = 0.001

    args.l1 = 1e-6

    args.dropout = 0.33

    lr_callback = LearningRateScheduler(step_decay_func(args))

    ckpt_callback = CheckpointCallback('./models/')

    model = build_model(args, embedding_matrix)

    model.fit_generator(mg, epochs = MAX_EPOCHS, steps_per_epoch = len(train_x) / BATCH_SIZE,

                        validation_data = (val_x, val_y),

                        callbacks = [lr_callback, ckpt_callback],

                        verbose = True, max_queue_size = 64)

    

main()
def predict():

    data = load_data('../input/test.csv')

    test_x = data['question_text']



    tokenizer = load_obj('tokenizer')

    test_x = tokenizer.texts_to_sequences(test_x)

    test_x = pad_sequences(test_x, MAX_TEXT_LENGTH)

    print('Preprocessing completed')

    

    model = load_model('./models/quora-3.h5', custom_objects = {'fmeasure': fmeasure})

    print('Load model completed')

    

    output = model.predict(test_x)

    print('Predict completed')



    submission = data[['qid']].copy()

    submission['prediction'] = (output > 0.4).astype(int)

    submission.to_csv('submission.csv', index = False)



predict()
!head submission.csv