import pandas as pd

import numpy as np

import string

import matplotlib.pyplot as plt

import seaborn as sns

from gensim.models import Word2Vec

import multiprocessing 

import keras

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder

from keras.models import Model

from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding

from keras.optimizers import RMSprop

from keras.preprocessing.text import Tokenizer

from keras.preprocessing import sequence

from keras.utils import to_categorical

from keras.callbacks import EarlyStopping

import nltk

!pip install tweet-preprocessor

from nltk.corpus import stopwords

import preprocessor as p

from nltk.tokenize import sent_tokenize as sent

from nltk.tokenize import word_tokenize as word

#nltk.download('stopwords')

from nltk.stem import PorterStemmer 

import re

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



%matplotlib inline
df = pd.read_excel('/kaggle/input/personality-data/mbti9k_comments250.xlsx',nrows=50)

print(len(df))
REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')

BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')

STOPWORDS = set(stopwords.words('english'))

FLOAT=re.compile('[-+]?\d*\.\d+|\d+')

ps = PorterStemmer() 

def clean_text(text):

    """

        text: a string

        

        return: modified initial string

    """

    text = FLOAT.sub('', text)

    text = text.lower() # lowercase text

    text = REPLACE_BY_SPACE_RE.sub(' ', text) # replace REPLACE_BY_SPACE_RE symbols by space in text. substitute the matched string in REPLACE_BY_SPACE_RE with space.

    text = BAD_SYMBOLS_RE.sub('', text) # remove symbols which are in BAD_SYMBOLS_RE from text. substitute the matched string in BAD_SYMBOLS_RE with nothing. 

    

    text = text.replace('x', '')

#    text = re.sub(r'\W+', '', text)

    text = ' '.join(ps.stem(word) for word in text.split() if word not in STOPWORDS) # remove stopwors from text

    return text
raw_text=df['comment']

#raw_text=data['body']

cleaned_text=[]

word_text=[]

f_text=[]

#print(datetime.now(pytz.timezone('Asia/Calcutta')).strftime('%Y-%m-%d %H:%M:%S'))

for i in range(0,len(raw_text)):

    text1=p.clean(raw_text[i])

    sent_text=sent(text1)

    for sen in sent_text:

        text=clean_text(sen) 

        text=word(text) #Tokenize into words

        f_text.append(text)  #Preparing input for word to vector model

    #if i==2:

    #print(raw_text[i])

        #print('\n\n\n')

        #print(f_text)

        #break

print(len(f_text))
raw_label1=df['type']

raw_label=[y.lower() for y in raw_label1]

label=[]

for x in raw_label:

    if x=='intj':

     label.append(1)

    else:

     label.append(0)

        

    

print(len(label) )   
EMB_DIM=300

w2v=Word2Vec(f_text,size=EMB_DIM,window=5,min_count=5,negative=15,iter=10,sg=0,workers=multiprocessing.cpu_count())

models=w2v.wv
models.save_word2vec_format('model_sg.bin')

models.save_word2vec_format('model_sg.txt', binary=False)

result=models.similar_by_word('friend')

print(result)
data_val=[]

for i in range(0,len(raw_text)):

    text1=p.clean(raw_text[i])

    text=clean_text(text1) 

    data_val.append(text)

    
# The maximum number of words to be used. (most frequent)

MAX_NB_WORDS = 50000

# Max number of words in each complaint.

MAX_SEQUENCE_LENGTH = 400

# This is fixed.

EMBEDDING_DIM = 100

tokenizer = keras.preprocessing.text.Tokenizer(num_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)

tokenizer.fit_on_texts(data_val)

word_index = tokenizer.word_index

print('Found %s unique tokens.' % len(word_index))

X = tokenizer.texts_to_sequences(data_val)

X = sequence.pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)

print('Shape of data tensor:', X.shape)

print(X)
Y=label
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.15)
max_words = 1000

max_len = 150

tok = Tokenizer(num_words=max_words)

tok.fit_on_texts(X_train)

sequences = tok.texts_to_sequences(X_train)

sequences_matrix = sequence.pad_sequences(sequences,maxlen=max_len)
def RNN():

    inputs = Input(name='inputs',shape=[MAX_SEQUENCE_LENGTH])

    layer = Embedding(MAX_NB_WORDS,EMBEDDING_DIM,input_length=MAX_SEQUENCE_LENGTH)(inputs)

    layer = LSTM(64)(layer)

    layer = Dense(256,name='FC1')(layer)

    layer = Activation('relu')(layer)

    layer = Dropout(0.5)(layer)

    layer = Dense(1,name='out_layer')(layer)

    layer = Activation('sigmoid')(layer)

    model = Model(inputs=inputs,outputs=layer)

    return model
model = RNN()

model.summary()

model.compile(loss='binary_crossentropy',optimizer=RMSprop(),metrics=['accuracy'])
model.fit(X_train,Y_train,batch_size=128,epochs=10,

          validation_split=0.2,callbacks=[EarlyStopping(monitor='val_loss',min_delta=0.0001)])
accr = model.evaluate(X_test,Y_test)
print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))