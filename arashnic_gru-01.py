
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import numpy as np 
import pandas as pd 
import nltk
import os
import gc
from keras.preprocessing import sequence,text
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense,Dropout,Embedding,LSTM,Conv1D,GlobalMaxPooling1D,Flatten,MaxPooling1D,GRU,SpatialDropout1D,Bidirectional
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report,f1_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
#pd.set_option('display.max_colwidth',100)
pd.set_option('display.max_colwidth', -1)
from nltk import FreqDist
from nltk.stem import SnowballStemmer,WordNetLemmatizer
from nltk.tokenize import word_tokenize


train = pd.read_table("../input/movie-review-sentiment-analysis-kernels-only/train.tsv")
test = pd.read_table("../input/movie-review-sentiment-analysis-kernels-only/test.tsv")
sub = pd.read_csv("../input/movie-review-sentiment-analysis-kernels-only/sampleSubmission.csv")




def standardize_text(df, text_field):
    df[text_field] = df[text_field].str.replace(r"http\S+", "")
    df[text_field] = df[text_field].str.replace(r"http", "")
    df[text_field] = df[text_field].str.replace(r"@\S+", "")
    df[text_field] = df[text_field].str.replace(r"[^A-Za-z0-9(),!?@\'\`\"\_\n]", " ")
    df[text_field] = df[text_field].str.replace(r"@", "at")
    df[text_field] = df[text_field].str.lower()
    return df

train = standardize_text(train, "Phrase")
test = standardize_text(test, "Phrase")

train_text=train.Phrase.values
test_text=test.Phrase.values
target=train.Sentiment.values
y=to_categorical(target)
print(train_text.shape,target.shape,y.shape)

from sklearn.model_selection import train_test_split
X_train_text,X_val_text,y_train,y_val=train_test_split(train_text,y,test_size=0.2,stratify=y,random_state=123)
print(X_train_text.shape,y_train.shape)
print(X_val_text.shape,y_val.shape)

corpus = train.Phrase.tolist() + test.Phrase.tolist()
all_words=' '.join(corpus)
all_words=word_tokenize(all_words)
dist=FreqDist(all_words)
num_unique_word=len(dist)


max_features = 15000
max_words = 100
batch_size = 256
epochs = 3
num_classes=5

#corpus = list(X_train_text)+ list(X_train_text)

tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(corpus))
X_train = tokenizer.texts_to_sequences(X_train_text)
X_val = tokenizer.texts_to_sequences(X_val_text)
X_test = tokenizer.texts_to_sequences(test_text)

X_train = sequence.pad_sequences(X_train, maxlen=max_words)
X_val = sequence.pad_sequences(X_val, maxlen=max_words)
X_test = sequence.pad_sequences(X_test, maxlen=max_words)
embed_size = 100 # how big is each word vector
#max_features = 20000 # how many unique words to use (i.e num rows in embedding vector)
maxlen = 100 

EMBEDDING_FILE = "../input/glove-global-vectors-for-word-representation/glove.6B.100d.txt"



def get_coefs(word, *arr):
    return word, np.asarray(arr, dtype='float32')
    
def get_embed_mat(EMBEDDING_FILE, max_features,embed_dim):
    # word vectors
    embeddings_index = dict(get_coefs(*o.rstrip().rsplit(' ')) for o in open(EMBEDDING_FILE, encoding='utf8'))
    print('Found %s word vectors.' % len(embeddings_index))

    # embedding matrix
    word_index = tokenizer.word_index
    num_words = min(max_features, len(word_index) + 1)
    all_embs = np.stack(embeddings_index.values()) #for random init
    embedding_matrix = np.random.normal(all_embs.mean(), all_embs.std(), 
                                        (num_words, embed_dim))
    for word, i in word_index.items():
        if i >= max_features:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    max_features = embedding_matrix.shape[0]
    
    return embedding_matrix
        


    
embed_dim = 100 #word vector dim
embedding_matrix = get_embed_mat(EMBEDDING_FILE,max_features,embed_dim)
print(embedding_matrix.shape)    
    
#-------------------------------------------------------------------------------- GRU
model = Sequential()
model.add(Embedding(max_features, embed_dim,input_length=X_train.shape[1], weights=[embedding_matrix],trainable=True))
model.add(SpatialDropout1D(0.25))
model.add(Bidirectional(GRU(128,return_sequences=True)))
model.add(Bidirectional(GRU(64,return_sequences=False)))
model.add(Dropout(0.5))
model.add(Dense(5, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

model.fit(X_train, y_train, validation_data=(X_val, y_val),epochs=4, batch_size=batch_size, verbose=1)
y_pred=model.predict_classes(X_test, verbose=1)


# LSTM
model1=Sequential()
model1.add(Embedding(max_features,embed_size, weights=[embedding_matrix],))
#model1.add(LSTM(128,dropout=0.4, recurrent_dropout=0.4,return_sequences=True))
model1.add(LSTM(64,dropout=0.4, recurrent_dropout=0.4,return_sequences=True))
model1.add(LSTM(32,dropout=0.5, recurrent_dropout=0.5,return_sequences=False))
model1.add(Dense(5,activation='softmax'))
model1.compile(loss='categorical_crossentropy',optimizer=Adam(lr=0.001),metrics=['accuracy'])
model1.summary()
  

model1.fit(X_train, y_train, validation_data=(X_val, y_val),epochs=4, batch_size=batch_size, verbose=1)
y_pred1=model1.predict_classes(X_test, verbose=1)

sub_agg=pd.DataFrame({'model1':y_pred1,'model':y_pred})
pred=sub_agg.agg('mode',axis=1)[0].values
sub_agg.head()

pred=[int(i) for i in pred]
sub.Sentiment=pred
sub.to_csv('desole.csv',index=False)
sub.head()





