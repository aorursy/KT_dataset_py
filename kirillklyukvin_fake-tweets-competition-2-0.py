import numpy as np
from numpy import savetxt 
import pandas as pd 
import re
import gc
import random
import os
import tensorflow as tf
import torch
import transformers

import spacy
from spacy.lemmatizer import Lemmatizer
from spacy.lookups import Lookups
from spacy.lang.en.stop_words import STOP_WORDS
import codecs
from gensim.models import Word2Vec

import nltk
from nltk.corpus import stopwords as nltk_stopwords
from gensim.models import Word2Vec
from tqdm import notebook

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, classification_report, precision_recall_curve
from sklearn.metrics import plot_confusion_matrix, make_scorer
from sklearn.model_selection import train_test_split, cross_validate, cross_val_score, GridSearchCV
from sklearn.model_selection import RandomizedSearchCV, KFold, StratifiedShuffleSplit
from sklearn.neighbors import DistanceMetric
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OrdinalEncoder, LabelEncoder, Binarizer, OneHotEncoder

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from sklearn.linear_model import SGDClassifier, LogisticRegression
from catboost import CatBoostClassifier, Pool, cv 

from tensorflow import keras
from tensorflow.keras.optimizers import SGD, Adam, RMSprop
from tensorflow.keras.layers import Dense, Input, Dropout, GlobalAveragePooling1D, GRU, LSTM, Conv1D, MaxPooling1D, Embedding
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, Callback
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
train = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')
test = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')
samp_sub = pd.read_csv('/kaggle/input/nlp-getting-started/sample_submission.csv')
train.head()
corpus_train = train['text'].values
corpus_test = test['text'].values
def lower_case(corpus):
    for tweet in range(len(corpus)):
        corpus[tweet] = corpus[tweet].lower()

    return corpus
corpus_train = lower_case(corpus_train)
corpus_test = lower_case(corpus_test)
all_words = ' '.join(corpus_train).split(' ')
len(all_words)
unique_words = list(set(all_words))
unique_words
len(unique_words)
def text_processing_03(df):
    
    text = df['text'].values
    
    df_new = df.copy()

    for sen in range(0, len(text)):
      
        ## removing part
        
        # remove hyperlinks
        document = re.sub(r'http\S+', '', str(text[sen]))
        # remove hashtags symbols
        document = re.sub(r'#', '', document)        
        # remove 'b'
        document = re.sub(r'^b\s+', '', document)   
        
        #remove strange characters
        document = re.sub(r'ûó', '', document)
        document = re.sub(r'ûò', '', document)
        document = re.sub(r'åê', '', document)
        document = re.sub(r'iûªm', '', document)
        document = re.sub(r'0npzp', '', document)
        document = re.sub(r'rq', '', document)
        document = re.sub(r'û_', '', document)
        document = re.sub(r'ûª', '', document)
        document = re.sub(r'ûï', '', document)
        document = re.sub(r'û', '', document)
        document = re.sub(r'å', '', document)
        document = re.sub(r'å_', '', document)       
        document = re.sub(r'ââ', '', document)
        document = re.sub(r'ìü', '', document)
        document = re.sub(r'ìñ1', '', document)
        document = re.sub(r'ìñ', '', document)
        document = re.sub(r'åèmgn', '', document)
        document = re.sub(r'åè', '', document)
        document = re.sub(r'åç', '', document)        
        document = re.sub(r'è', '', document)
        document = re.sub(r'ç', '', document)
        document = re.sub(r'ã', '', document)
        document = re.sub(r'ì', 'i', document)
        
                
        # convert all letters to a lower case
        document = document.lower()
        
        ## replacing part
        document = re.sub(r'èmgn', 'emergency', document)
        document = re.sub(r"isn't", 'is not', document)
        document = re.sub(r"havn't", 'have not', document)
        document = re.sub(r"'s", ' is', document)
        document = re.sub(r"'m", ' am', document)
        document = re.sub(r"'d", ' had', document)
        document = re.sub(r"'ve", ' have', document)
        document = re.sub(r"'t", ' not', document)
        
        document = re.sub(r"e-bike", 'electro bike', document)
        document = re.sub(r'hwy', 'highway', document)
        document = re.sub(r'nsfw', 'not safe for work', document)
        document = re.sub(r'koz', 'because', document)       
        
                
        ### NEW IN THIS VERSION
        
        ## remove repeating characters
        pattern = re.compile(r'(.)\1{2,}', re.DOTALL) 
        document = pattern.sub(r"\1\1", document)
        ## remove usernames
        document = re.sub(r'@\S+', '', document)
        ## remove all digits and numbers
        document = re.sub(r'\d+', '', document)
        
        
        # remove special symbols 
        document = re.sub(r'\W', ' ', document)
        # replace few spaces to a single one
        document = re.sub(r'\s+', ' ', document, flags=re.I)
        
        
        ### Removed expressions from the previous version
        
        # remove individual symbols from the start of the tweet
        #document = re.sub(r'^[a-zA-Z]\s+', '', document)
        # remove individual symbols
        #document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)
        
        df_new.loc[sen, 'text_lemm'] = document
        
    return df_new
X_train = train.copy()

X_train = text_processing_03(X_train)
X_train.head(20)
X_test = test.copy()

X_test = text_processing_03(X_test)
X_test.head(20)
corpus_train_lemm = X_train['text_lemm'].values

all_words_processed = ' '.join(corpus_train_lemm).split(' ')
unique_words = list(set(all_words_processed))
len(unique_words)
serie = pd.Series(all_words_processed)
serie.value_counts()[-50:]
train_bert = X_train[['target','text']]
test_bert = X_test[['text']]

y_train = X_train['target']
model_class, tokenizer_class, pretrained_weights = (
    transformers.DistilBertModel, transformers.DistilBertTokenizer, 'distilbert-base-uncased')

tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
model = model_class.from_pretrained(pretrained_weights)
tokenized_train = train_bert['text'].apply((lambda x: tokenizer.encode(x, add_special_tokens=True)))
tokenized_test = test_bert['text'].apply((lambda x: tokenizer.encode(x, add_special_tokens=True)))
tokenized_train[1]
def bert_features(tokenized):

    max_len = 0
    for i in tokenized.values:
        if len(i) > max_len:
            max_len = len(i)

    padded = np.array([i + [0]*(max_len-len(i)) for i in tokenized.values])
    
    attention_mask = np.where(padded != 0, 1, 0)
    
    batch_size = 1
    embeddings = []
    for i in notebook.tqdm(range(padded.shape[0] // batch_size)):
            batch = torch.LongTensor(padded[batch_size*i:batch_size*(i+1)]) 
            attention_mask_batch = torch.LongTensor(attention_mask[batch_size*i:batch_size*(i+1)])
        
            with torch.no_grad():
                batch_embeddings = model(batch, attention_mask=attention_mask_batch)
        
            embeddings.append(batch_embeddings[0][:,0,:].numpy())
    
    features = np.concatenate(embeddings)
    
    return(features)
X_train_bert = bert_features(tokenized_train) 
X_test_bert = bert_features(tokenized_test)
del tokenized_train, tokenized_test
lens = []

for i in range(len(X_train)):
    len_row = len(X_train['text_lemm'][i])
    lens.append(len_row)
    
max(lens)
NUM_WORDS = 10000

tokenizer = Tokenizer(num_words=NUM_WORDS)
tokenizer.fit_on_texts(X_train['text_lemm'].values)

text_train_keras = tokenizer.texts_to_sequences(X_train['text_lemm'].values)
text_test_keras = tokenizer.texts_to_sequences(X_test['text_lemm'].values)

vocab_size = len(tokenizer.word_index) + 1

X_train_keras = pad_sequences(text_train_keras, padding='post', maxlen=148)
X_test_keras = pad_sequences(text_test_keras, padding='post', maxlen=148)

y_train = X_train['target']
X_train_keras[0]
optimizer = Adam(lr=0.0001)
try:
    del model3
except:
    print('yeaaaaahboiiiiii')

model3 = Sequential()
model3.add(Embedding(vocab_size, 1000, input_length=148, trainable=True))
model3.add(LSTM(100, return_sequences=True, dropout=0.2, recurrent_dropout=0.15))
#model3.add(LSTM(100,return_sequences=False, dropout=0.2, recurrent_dropout=0.15))
model3.add(Dense(50, activation='relu', kernel_initializer='lecun_uniform'))
model3.add(Dense(50, activation='relu', kernel_initializer='lecun_uniform'))
model3.add(Dense(32, activation='relu', kernel_initializer='lecun_uniform'))
model3.add(Dropout(rate=0.2))
model3.add(Dense(1, activation="sigmoid"))

model3.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
history = model3.fit(X_train_keras, y_train, epochs=20, batch_size=400, validation_split=0.1)
model3.evaluate(X_train_keras, y_train)
prediction = model3.predict(X_test_keras).round().astype('int')
submission = samp_sub.copy()
submission['target'] = prediction
    
submission.to_csv('/kaggle/working/ver_2_015.csv', index=False)
SEED=2202
X_train_sub, X_valid_sub, y_train_sub, y_valid_sub = train_test_split(X_train_bert, y_train, test_size=0.1, random_state=SEED)
cbc = CatBoostClassifier(loss_function='Logloss',
                    iterations=2000,
                    learning_rate=0.09,
                    depth=3,
                    subsample=0.8,
                    verbose=100, 
                    grow_policy='Depthwise',
                    random_state=SEED)

cbc.fit(X_train_sub, y_train_sub)
f1_score(y_valid_sub, cbc.predict(X_valid_sub)).round(4)
def plot_hist(history):

    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.grid()
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.grid()
    plt.show()
optimizer = Adam(lr=0.0001)
optimizer = SGD(lr=0.001)
try:
    del model
    print('refined')
except:
    print('next')

model = Sequential()

model.add(Dense(50, input_dim=768, activation='relu', kernel_initializer='lecun_uniform'))
model.add(Dense(50, activation='relu', kernel_initializer='lecun_uniform'))

model.add(Dense(1, activation='sigmoid', kernel_initializer='lecun_uniform'))

model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['acc'])
history = model.fit(X_train_bert, y_train, epochs=1000, validation_split=0.1, batch_size=300, verbose=0)
plot_hist(history)
X_train_bert.shape
model.summary()
features_train = X_train_bert.reshape(-1, 768, 1)
features_train
del model2

model2 = Sequential()

model2.add(Dense(50, input_dim=768, activation='relu', kernel_initializer='lecun_uniform'))
model2.add(Dropout(rate=0.2))
model2.add(LSTM(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.15))
model2.add(LSTM(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.15))
model2.add(Dropout(rate=0.2))
model2.add(Dense(10, activation='relu'))
model2.add(Dense(1, activation='sigmoid'))

model2.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['acc'])
model2.summary()
history = model2.fit(features_train, y_train, epochs=50, validation_split=0.1, batch_size=100, verbose=1)
plot_hist(history)


def submission(model, test):

    
    pred = model.predict(test)
    
    submission = samp_sub.copy()
    submission['target'] = pred
    
    submission.to_csv('/kaggle/working/ver_2_014.csv', index=False)
    
submission(cbc, X_test_bert)
def submission_keras(model, X_test):

    prediction = model.predict(X_test).round().astype('int')
    submission = samp_sub.copy()
    submission['target'] = prediction
    
    submission.to_csv('/kaggle/working/ver_2_006.csv', index=False)

submission_keras(model, X_test_bert)
