# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import csv
import pandas as pd
from keras.models import Model
import tensorflow as tf
import numpy as np
import json
import nltk
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords as sw
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer
import re
from sklearn.feature_extraction.text import TfidfVectorizer

import matplotlib.pyplot as plt
import keras
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, Flatten,BatchNormalization
from keras.layers import Bidirectional, GlobalMaxPool1D, SpatialDropout1D, concatenate
from keras.models import Model, Sequential, load_model
from keras import initializers, regularizers, constraints, optimizers, layers
from keras.callbacks import ModelCheckpoint
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.pooling import GlobalMaxPooling1D, GlobalAveragePooling1D
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re
from nltk.stem import WordNetLemmatizer
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)
#vocab_size = 10000
#embedding_dim = 64
trunc_type = "pre"
padding_type = "pre"
oov_tok = "<OOV>"

stopwords = sw.words("english")
stemmer = SnowballStemmer("english")
lemmatizer = WordNetLemmatizer()
print("Number of stop words:", len(stopwords))
def filter_words(sentence,stopwords=None):
    sentence = re.sub(r'[^\w\s]', '', str(sentence).lower().strip())
    sentence = sentence.split()
    if stopwords is not None:
        sentence = [word for word in sentence if word not in 
                    stopwords]
    return sentence
def Stemlemise(word):
    ps = nltk.stem.porter.PorterStemmer()
    lem = nltk.stem.wordnet.WordNetLemmatizer()
    ps.stem(word)
    lem.lemmatize(word)
    return word
sentences = []
labels = []
ps = nltk.stem.porter.PorterStemmer()
lem = nltk.stem.wordnet.WordNetLemmatizer()
with open("../input/news-category-dataset/News_Category_Dataset_v2.json", 'r') as file:
  for line in file:
    sentences.append( json.loads(line) )

dtf = pd.DataFrame(sentences)
dtf = dtf[["category","headline"]]
dtf = dtf.rename(columns={"category":"y", "headline":"text"})

dtf.sample(5)

import matplotlib.pyplot as plt


fig, ax =  plt.subplots()
fig.suptitle("y", fontsize=12)
dtf["y"].reset_index().groupby("y").count().sort_values(by= 
       "index").plot(kind="barh", legend=False, 
        ax=ax).grid(axis='x')
plt.show()
def utils_preprocess_text(text, flg_stemm=False, flg_lemm=True, lst_stopwords=None):
  
    text = re.sub(r'[^\w\s]', '', str(text).lower().strip())
            
    ## Tokenize (convert from string to list)
    lst_text = text.split()
    ## remove Stopwords
    if lst_stopwords is not None:
        lst_text = [word for word in lst_text if word not in 
                    lst_stopwords]
                
    ## Stemming (remove -ing, -ly, ...)
    if flg_stemm == True:
        ps = nltk.stem.porter.PorterStemmer()
        lst_text = [ps.stem(word) for word in lst_text]
                
    ## Lemmatisation (convert the word into root word)
    if flg_lemm == True:
        lem = nltk.stem.wordnet.WordNetLemmatizer()
        lst_text = [lem.lemmatize(word) for word in lst_text]
            
    ## back to string from list
    text = " ".join(lst_text)
    return text
lst_stopwords = nltk.corpus.stopwords.words("english")
lst_stopwords
dtf["text_clean"] = dtf["text"].apply(lambda x: 
          utils_preprocess_text(x, flg_stemm=False, flg_lemm=True, 
          lst_stopwords=lst_stopwords))
dtf.head()
dtf_train, dtf_test = model_selection.train_test_split(dtf, test_size=0.3)
## get target
y_train = dtf_train["y"]
y_test = dtf_test["y"]
labels=y_test.unique()
labels
corpus_sentences = list(map(str,dtf_train["text_clean"])) + list(map(str,dtf_test["text_clean"]))
text = utils_preprocess_text(corpus_sentences)

dtf_train["text_clean"]
y_train
Xy_train = pd.concat([dtf_train["text_clean"], y_train], axis=1)
max_words = 15000
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(list(text))
list_tokenized_train = tokenizer.texts_to_sequences(Xy_train["text_clean"])
list_tokenized_test = tokenizer.texts_to_sequences(dtf_test["text_clean"])
max_len = 80
X_train_final = pad_sequences(list_tokenized_train,maxlen=max_len)
X_test_final = pad_sequences(list_tokenized_test,maxlen=max_len)
train_dummies = pd.get_dummies(Xy_train['y'])
y_train_final = train_dummies.values

Xy_train
np.random.seed(226)
shuffle_indices = np.random.permutation(np.arange(len(X_train_final)))
X_trains = X_train_final[shuffle_indices]
y_trains = y_train_final[shuffle_indices]
phs = Xy_train.iloc[shuffle_indices]

y_train_final[0]
Xy_train
td = 500
vec = TfidfVectorizer(max_features=td, ngram_range=(1,2))
x_tfidf = vec.fit_transform(phs['text_clean']).toarray()
test_tfidf = vec.transform(dtf_test["text_clean"]).toarray()

embedding_path = "../input/fasttext-crawl-300d-2m/crawl-300d-2M.vec"
def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
embedding_index = dict(get_coefs(*o.strip().split(" ")) for o in open(embedding_path))

embed_size = 300
word_index = tokenizer.word_index
nb_words = max(max_words, len(word_index))
embedding_matrix = np.zeros((nb_words, embed_size))
for word, i in word_index.items():
    if i >= max_words: continue
    embedding_vector = embedding_index.get(word)
    if embedding_vector is not None: embedding_matrix[i] = embedding_vector
len(word_index)
nb_words 
print(embedding_matrix.shape)
def keras_dl(model, embed_size, batch_size, epochs):   
    inp = Input(shape = (max_len,), name = 'lstm')
    print(inp.shape)
    x = Embedding(max_words,embed_size,weights = [embedding_matrix], trainable = False)(inp)
    x1 = SpatialDropout1D(0.5)(x)
    
    x_lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences = True))(x1)
    x_lstm_c1d = Conv1D(64,kernel_size=3,padding='valid',activation='tanh')(x_lstm)
    x_lstm_c1d_gp = GlobalMaxPooling1D()(x_lstm_c1d)
    
    x_gru =tf.keras.layers.Bidirectional(tf.keras.layers.GRU(128, return_sequences = True))(x1)
    x_gru_c1d = Conv1D(64,kernel_size=2,padding='valid',activation='tanh')(x_gru)
    x_gru_c1d_gp = GlobalMaxPooling1D()(x_gru_c1d)

    inp2 = Input(shape = (td,), name = 'tfidf')
    x2 = BatchNormalization()(inp2)
    x2 = Dense(8, activation='tanh')(x2)
    
    x_f = concatenate([x_lstm_c1d_gp, x_gru_c1d_gp])
    x_f = BatchNormalization()(x_f)
    x_f = Dropout(0.4)(Dense(128, activation='tanh') (x_f))    
    x_f = BatchNormalization()(x_f)
    x_f = concatenate([x_f, x2])
    x_f = Dropout(0.4)(Dense(64, activation='tanh') (x_f))
    x_f = Dense(41, activation = "sigmoid")(x_f)
    model = Model(inputs = [inp, inp2], outputs = x_f)
       
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    return (model)
embed_size = 300
batch_size = 256
epochs = 30
model = Sequential()
file_path = "best_model.hdf5"

check_point = ModelCheckpoint(file_path, monitor = "val_loss", verbose = 1,
                              save_best_only = True, mode = "min")
early_stop = EarlyStopping(monitor = "val_loss", mode = "min", patience = 3)

firstmodel = keras_dl(model, embed_size, batch_size, epochs)

firstmodel.summary()
text_model = firstmodel.fit({'lstm': X_trains, 'tfidf': x_tfidf}, y_trains, batch_size=batch_size,epochs=epochs,verbose=0,
                            validation_split = 0.1,
                            callbacks = [check_point, early_stop])
firstmodel = load_model(file_path)
pred = firstmodel.predict([np.array(X_test_final), test_tfidf], verbose = 1)
pred2 = np.round(np.argmax(pred, axis=1)).astype(int)

sub = pd.DataFrame({'sentence':  dtf_test['text_clean'],
                   'category': labels[pred2]})

sub
