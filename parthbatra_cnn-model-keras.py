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
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import re

from tqdm import tqdm

from gensim.models import KeyedVectors

import tensorflow as tf

import tensorflow.keras

from tensorflow.keras import layers

from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score, roc_curve, classification_report

from tensorflow.keras.preprocessing.text import Tokenizer

from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras import layers,models

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
#Reading Data

df = pd.read_csv('../input/imdb-dataset-of-50k-movie-reviews/IMDB Dataset.csv')
df.columns
pd.options.display.max_colwidth = 150

tqdm.pandas()
df.head()
df.info()
sentiments = pd.get_dummies(df['sentiment'])

df = pd.concat([df,sentiments],axis = 1)
def clean_str(string):

    string = re.sub(r"[^A-Za-z0-9(),#!?\'\`]", " ", string)     

    string = re.sub(r"\'s", " \'s", string) 

    string = re.sub(r"\'ve", " \'ve", string) 

    string = re.sub(r"n\'t", " n\'t", string) 

    string = re.sub(r"\'re", " \'re", string) 

    string = re.sub(r"\'d", " \'d", string) 

    string = re.sub(r"\'ll", " \'ll", string) 

    string = re.sub(r",", " , ", string)

    string = re.sub(r"!", " ! ", string) 

    string = re.sub(r"\(", " ( ", string) 

    string = re.sub(r"\)", " ) ", string) 

    string = re.sub(r"\?", " ? ", string) 

    string = re.sub(r"\s{2,}", " ", string)

    string = re.sub(r' br ','',string)

    string = re.sub(r'\"',' \" ',string)

    return string.strip()
mispell_dict = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because",

                "could've": "could have", "couldn't": "could not", "didn't": "did not",  "doesn't": "does not",

                "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not", "he'd": "he would",

                "he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will",

                "how's": "how is",  "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have",

                "I'm": "I am", "I've": "I have", "i'd": "i would", "i'd've": "i would have", "i'll": "i will",

                "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would", 

                "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us",

                "ma'am": "madam", "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have",

                "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have",

                "o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not",

                "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have",

                "she's": "she is", "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have",

                "so've": "so have","so's": "so as", "this's": "this is","that'd": "that would", "that'd've": "that would have",

                "that's": "that is", "there'd": "there would", "there'd've": "there would have", "there's": "there is", 

                "here's": "here is","they'd": "they would", "they'd've": "they would have", "they'll": "they will",

                "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have", 

                "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will",

                "we'll've": "we will have", "we're": "we are", "we've": "we have", "weren't": "were not",

                "what'll": "what will", "what'll've": "what will have", "what're": "what are",

                "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", 

                "where'd": "where did", "where's": "where is", "where've": "where have", "who'll": "who will",

                "who'll've": "who will have", "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have",

                "will've": "will have", "won't": "will not", "won't've": "will not have", "would've": "would have", 

                "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would",

                "y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have","you'd": "you would", 

                "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have", "you're": "you are",

                "you've": "you have", 'colour': 'color', 'centre': 'center', 'favourite': 'favorite', 'travelling': 'traveling',

                'counselling': 'counseling', 'theatre': 'theater', 'cancelled': 'canceled', 'labour': 'labor',

                'organisation': 'organization', 'wwii': 'world war 2', 'citicise': 'criticize', 'youtu ': 'youtube ',

                'Qoura': 'Quora', 'sallary': 'salary', 'Whta': 'What', 'narcisist': 'narcissist', 'howdo': 'how do',

                'whatare': 'what are', 'howcan': 'how can', 'howmuch': 'how much', 'howmany': 'how many', 'whydo': 'why do', 

                'doI': 'do I', 'theBest': 'the best', 'howdoes': 'how does', 'mastrubation': 'masturbation', 'mastrubate': 'masturbate',

                "mastrubating": 'masturbating', 'pennis': 'penis', 'Etherium': 'Ethereum', 'narcissit': 'narcissist', 'bigdata': 'big data',

                '2k17': '2017', '2k18': '2018', 'qouta': 'quota', 'exboyfriend': 'ex boyfriend', 'airhostess': 'air hostess', "whst": 'what',

                'watsapp': 'whatsapp', 'demonitisation': 'demonetization', 'demonitization': 'demonetization', 'demonetisation': 'demonetization'}



def _get_mispell(mispell_dict):

    mispell_re = re.compile('(%s)' % '|'.join(mispell_dict.keys()))

    return mispell_dict, mispell_re



mispellings, mispellings_re = _get_mispell(mispell_dict)

def replace_typical_misspell(text):

    def replace(match):

        return mispellings[match.group(0)]

    return mispellings_re.sub(replace, text)
df['review'] = df['review'].apply(lambda x:replace_typical_misspell(x))

df['review'] = df['review'].apply(lambda x:clean_str(x))
df['review'][1]
!wget -P /root/input/ -c "https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz"
EMBEDDING_FILE = '/root/input/GoogleNews-vectors-negative300.bin.gz' # from above

word2vec = KeyedVectors.load_word2vec_format(EMBEDDING_FILE, binary=True)
def build_vocab(sentences, verbose =  True):

    """

    :param sentences: list of list of words

    :return: dictionary of words and their count

    """

    vocab = {}

    for sentence in tqdm(sentences, disable = (not verbose)):

        for word in sentence:

            try:

                vocab[word] += 1

            except KeyError:

                vocab[word] = 1

    return vocab
sentences = df["review"].apply(lambda x: x.split()).values

vocab = build_vocab(sentences)

print({k: vocab[k] for k in list(vocab)[:5]})
len(vocab)
import operator 



def check_coverage(vocab,embeddings_index):

    a = {}

    oov = {}

    k = 0

    i = 0

    for word in tqdm(vocab):

        try:

            a[word] = embeddings_index[word]

            k += vocab[word]

        except:



            oov[word] = vocab[word]

            i += vocab[word]

            pass



    print('Found embeddings for {:.2%} of vocab'.format(len(a) / len(vocab)))

    print('Found embeddings for  {:.2%} of all text'.format(k / (k + i)))

    sorted_x = sorted(oov.items(), key=operator.itemgetter(1))[::-1]



    return sorted_x
oov = check_coverage(vocab,word2vec)
df['review'] = df['review'].apply(lambda x: list(filter(None,x.split(' '))))

df['review'] = df['review'].apply(lambda x: ' '.join(x))
x_train,x_test,y_train,y_test = train_test_split(df['review'],df[['positive','negative']], test_size = 0.15)
x_train,x_val,y_train,y_val = train_test_split(x_train,y_train[['positive','negative']], test_size = 0.15)
df['sent_len'] = df['review'].apply(lambda x: len(x.split(' ')))

plt.figure()

sns.distplot(df['sent_len'],kde = False, norm_hist = False)

plt.show()

print('Mean Sentence Length: ', np.round(df['sent_len'].mean()))

print('Median Sentence Length: ',df['sent_len'].median())
NB_WORDS = 20000

MAX_LENGTH = 200
tokenizer = Tokenizer(num_words = NB_WORDS, split = ' ')
tokenizer.fit_on_texts(x_train)

X_train = tokenizer.texts_to_sequences(x_train)

X_val = tokenizer.texts_to_sequences(x_val)

X_test = tokenizer.texts_to_sequences(x_test)

X_train = pad_sequences(X_train, maxlen = MAX_LENGTH, padding = 'post')

X_val = pad_sequences(X_val, maxlen = MAX_LENGTH, padding = 'post')

X_test = pad_sequences(X_test, maxlen = MAX_LENGTH, padding = 'post')
tokenizer.word_index
embedding_dim = 300

embedding_matrix = np.zeros((NB_WORDS,embedding_dim),dtype = float)

for word, i in tokenizer.word_index.items():

    if i < NB_WORDS:

        if word in word2vec:

            embedding_vector = word2vec[word]



            embedding_matrix[i] = embedding_vector

        else:

            embedding_matrix[i] = np.random.uniform(-0.25,0.25,embedding_dim)

            
from tensorflow.keras import backend as K



def recall_m(y_true, y_pred):

    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))

    recall = true_positives / (possible_positives + K.epsilon())

    return recall



def precision_m(y_true, y_pred):

    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))

    precision = true_positives / (predicted_positives + K.epsilon())

    return precision



def f1_m(y_true, y_pred):

    precision = precision_m(y_true, y_pred)

    recall = recall_m(y_true, y_pred)

    return 2*((precision*recall)/(precision+recall+K.epsilon()))
num_filters = 100

inp_01 = layers.Input(shape = (MAX_LENGTH,))

embedding_layer = layers.Embedding(NB_WORDS,embedding_dim, weights = [embedding_matrix],input_length = MAX_LENGTH,trainable = False)(inp_01)



conv_01 = layers.Conv1D(filters = num_filters ,kernel_size = 3, activation='relu', kernel_regularizer = tf.keras.regularizers.l2(3) )(embedding_layer)

conv_02 = layers.Conv1D(filters = num_filters ,kernel_size = 4, activation='relu', kernel_regularizer = tf.keras.regularizers.l2(3) )(embedding_layer)

conv_03 = layers.Conv1D(filters = num_filters ,kernel_size = 5, activation='relu', kernel_regularizer = tf.keras.regularizers.l2(3) )(embedding_layer)



max_p01 = layers.MaxPooling1D(pool_size = MAX_LENGTH - 3 + 1)(conv_01)

max_p01 = layers.MaxPooling1D(pool_size = MAX_LENGTH - 4 + 1)(conv_02)

max_p02 = layers.MaxPooling1D(pool_size = MAX_LENGTH - 5 + 1)(conv_03)



concatenated = layers.Concatenate(axis = -1)([max_p01, max_p01, max_p02])



flatten = layers.Flatten()(concatenated)

dropout = layers.Dropout(0.5)(flatten)



CNN_pred_01 = layers.Dense(2, activation = 'softmax')(dropout)
CNN_model_01 = models.Model(inp_01, CNN_pred_01)

CNN_model_01.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy',precision_m, recall_m, f1_m])

CNN_model_01.summary()
history_CNN_01 = CNN_model_01.fit(X_train, y_train, batch_size = 32, epochs = 30, validation_data = (X_val,y_val))
plt.figure()

plt.plot(history_CNN_01.history['accuracy'])

plt.plot(history_CNN_01.history['val_accuracy'])

plt.show()



plt.figure()

plt.plot(history_CNN_01.history['loss'])

plt.plot(history_CNN_01.history['val_loss'])

plt.show()
num_filters = 100



inp_20 = layers.Input(shape = (MAX_LENGTH,))

embedding_layer = layers.Embedding(NB_WORDS,embedding_dim, weights = [embedding_matrix],input_length = MAX_LENGTH,trainable = False)(inp_20)



reshape = layers.Reshape((MAX_LENGTH, embedding_dim, 1))(embedding_layer)#Keras requires ndim=4. We reshape to ndim=3; Keras automatically adds batch size as 4th dimension.



conv_20 =  layers.Conv2D(filters = num_filters, kernel_size = (3,embedding_dim),kernel_initializer='normal', activation='relu', kernel_regularizer=tf.keras.regularizers.l2(3))(reshape)

conv_21 =  layers.Conv2D(filters = num_filters, kernel_size = (4,embedding_dim),kernel_initializer='normal', activation='relu', kernel_regularizer=tf.keras.regularizers.l2(3))(reshape)

conv_22 =  layers.Conv2D(filters = num_filters, kernel_size = (5,embedding_dim),kernel_initializer='normal', activation='relu', kernel_regularizer=tf.keras.regularizers.l2(3))(reshape)



max_p20 = layers.MaxPool2D(pool_size = (MAX_LENGTH - 3 + 1,1), padding = 'valid')(conv_20)

max_p21 = layers.MaxPool2D(pool_size = (MAX_LENGTH - 4 + 1,1), padding = 'valid')(conv_21)

max_p22 = layers.MaxPool2D(pool_size = (MAX_LENGTH - 5 + 1,1), padding = 'valid')(conv_22)



concatenate = layers.Concatenate(axis = -1)([max_p20, max_p21, max_p22])

flatten = layers.Flatten()(concatenate)



dropout = layers.Dropout(0.5)(flatten)



CNN_20_pred = layers.Dense(2, activation = 'softmax')(dropout)
CNN_model_20 = models.Model(inp_20, CNN_20_pred)

CNN_model_20.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy',precision_m, recall_m, f1_m])

CNN_model_20.summary()
hist_CNN_20 = CNN_model_20.fit(X_train, y_train, batch_size = 32, epochs = 30, validation_data = (X_val,y_val))
num_filters = 100

inp_21 = layers.Input(shape = (MAX_LENGTH,))

embedding_layer = layers.Embedding(NB_WORDS,embedding_dim, weights = [embedding_matrix],input_length = MAX_LENGTH,trainable = True)(inp_21)



conv_21 = layers.Conv1D(filters = num_filters ,kernel_size = 3, activation='relu', kernel_regularizer = tf.keras.regularizers.l2(3) )(embedding_layer)

conv_22 = layers.Conv1D(filters = num_filters ,kernel_size = 4, activation='relu', kernel_regularizer = tf.keras.regularizers.l2(3) )(embedding_layer)

conv_23 = layers.Conv1D(filters = num_filters ,kernel_size = 5, activation='relu', kernel_regularizer = tf.keras.regularizers.l2(3) )(embedding_layer)

max_p21 = layers.MaxPooling1D(pool_size = MAX_LENGTH - 3 + 1)(conv_21)

max_p21 = layers.MaxPooling1D(pool_size = MAX_LENGTH - 4 + 1)(conv_22)

max_p22 = layers.MaxPooling1D(pool_size = MAX_LENGTH - 5 + 1)(conv_23)



concatenated = layers.Concatenate(axis = -1)([max_p21, max_p21, max_p22])



flatten = layers.Flatten()(concatenated)

dropout = layers.Dropout(0.1)(flatten)



CNN_pred_21 = layers.Dense(2, activation = 'softmax')(dropout)
CNN_model_21 = models.Model(inp_21, CNN_pred_21)

CNN_model_21.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy',precision_m, recall_m, f1_m])

CNN_model_21.summary()
history_CNN_11 = CNN_model_11.fit(X_train, y_train, batch_size = 32, epochs = 30, validation_data = (X_val,y_val))
plt.figure()

plt.plot(history_CNN_11.history['accuracy'])

plt.plot(history_CNN_21.history['val_accuracy'])

plt.show()



plt.figure()

plt.plot(history_CNN_11.history['loss'])

plt.plot(history_CNN_11.history['val_loss'])

plt.show()
num_filters = 100



inp_21 = layers.Input(shape = (MAX_LENGTH,))

embedding_layer = layers.Embedding(NB_WORDS,embedding_dim, weights = [embedding_matrix],input_length = MAX_LENGTH,trainable = False)(inp_20)



reshape = layers.Reshape((MAX_LENGTH, embedding_dim, 1))(embedding_layer)#Keras requires ndim=4. We reshape to ndim=3; Keras automatically adds batch size as 4th dimension.



conv_210 =  layers.Conv2D(filters = num_filters, kernel_size = (3,embedding_dim),kernel_initializer='normal', activation='relu', kernel_regularizer=tf.keras.regularizers.l2(3))(reshape)

conv_211 =  layers.Conv2D(filters = num_filters, kernel_size = (4,embedding_dim),kernel_initializer='normal', activation='relu', kernel_regularizer=tf.keras.regularizers.l2(3))(reshape)

conv_212 =  layers.Conv2D(filters = num_filters, kernel_size = (5,embedding_dim),kernel_initializer='normal', activation='relu', kernel_regularizer=tf.keras.regularizers.l2(3))(reshape)



max_p210 = layers.MaxPool2D(pool_size = (MAX_LENGTH - 3 + 1,1), padding = 'valid')(conv_210)

max_p211 = layers.MaxPool2D(pool_size = (MAX_LENGTH - 4 + 1,1), padding = 'valid')(conv_211)

max_p212 = layers.MaxPool2D(pool_size = (MAX_LENGTH - 5 + 1,1), padding = 'valid')(conv_212)



concatenate = layers.Concatenate(axis = -1)([max_p210, max_p211, max_p212])

flatten = layers.Flatten()(concatenate)



dropout = layers.Dropout(0.5)(flatten)



CNN_21_pred = layers.Dense(2, activation = 'softmax')(dropout)
CNN_model_22 = models.Model(inp_21, CNN_21_pred)

CNN_model_22.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy',precision_m, recall_m, f1_m])

CNN_model_22.summary()
history_CNN_21 = CNN_model_22.fit(X_train, y_train, batch_size = 32, epochs = 30, validation_data = (X_val,y_val))