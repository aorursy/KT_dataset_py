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
train=pd.read_csv('/kaggle/input/twitter-sentiment-analysis-hatred-speech/train.csv')

test=pd.read_csv('/kaggle/input/twitter-sentiment-analysis-hatred-speech/test.csv')
def punc(df):

    df['tweet'] = df['tweet'].str.replace('[#]','')

    print(df)

punc(train)

punc(test)
import nltk

from nltk.tokenize import TweetTokenizer

def tokenizer(df):

    tknzr = TweetTokenizer(strip_handles=True)

    df['tweet']= df['tweet'].apply(lambda x: tknzr.tokenize(x))

    print(df)

    

tokenizer(test)

tokenizer(train)
import nltk

from nltk.corpus import stopwords

stop=stopwords.words("english")

def stop_words(df):

    df['tweet']=df['tweet'].apply(lambda x: [i.lower() for i in x if i not in stop])

    print(df)

stop_words(train)

stop_words(test)
import re

def clean(df):

    df['tweet']=df['tweet'].apply(lambda x: [i for i in x if not re.match('[^\w\s]',i) and len(i)>3])

    print(df)

clean(train)

clean(test)
from nltk.stem import PorterStemmer

from textblob import Word

st = PorterStemmer()

def stemnlemm(df):

    df['tweet']=df['tweet'].apply(lambda x: ' '.join([Word(st.stem(i)).lemmatize() for i in x]))

    print(df)

stemnlemm(train)

stemnlemm(test)
X_train=pd.DataFrame(train['tweet'])

Y_train=pd.DataFrame(train['label'])
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(X_train,Y_train,random_state = 0 , stratify = Y_train)
x_train=x_train['tweet']

x_train
y_train=y_train['label']

y_train

import transformers

from tokenizers import BertWordPieceTokenizer

# First load the real tokenizer

tokenizer = transformers.DistilBertTokenizer.from_pretrained('distilbert-base-uncased' , lower = True)

# Save the loaded tokenizer locally

tokenizer.save_pretrained('.')

# Reload it with the huggingface tokenizers library

fast_tokenizer = BertWordPieceTokenizer('vocab.txt', lowercase=True)

fast_tokenizer
def fast_encode(texts, tokenizer, chunk_size=256, maxlen=400):



    tokenizer.enable_truncation(max_length=maxlen)

    tokenizer.enable_padding(max_length=maxlen)

    all_ids = []

    

    for i in range(0, len(texts), chunk_size):

        text_chunk = texts[i:i+chunk_size].tolist()

        encs = tokenizer.encode_batch(text_chunk)

        all_ids.extend([enc.ids for enc in encs])

    

    return np.array(all_ids)
x_test=x_test['tweet']

x_test
y_test=y_test['label']

y_test
x_train = fast_encode(x_train.values, fast_tokenizer, maxlen=400)

x_test = fast_encode(x_test.values, fast_tokenizer, maxlen=400)
import tensorflow as tf

from keras.layers import LSTM,Dense,Bidirectional,Input

from keras.models import Model

from tensorflow.keras.preprocessing import sequence

from tensorflow.keras.preprocessing.text import Tokenizer

from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.optimizers import Adam



def build_model(transformer, max_len=400):

    

    input_word_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")

    sequence_output = transformer(input_word_ids)[0]

    cls_token = sequence_output[:, 0, :]

    out = Dense(1, activation='sigmoid')(cls_token)

    

    model = Model(inputs=input_word_ids, outputs=out)

    model.compile(Adam(lr=2e-5), loss='binary_crossentropy', metrics=['accuracy'])

    

    return model
bert_model = transformers.TFDistilBertModel.from_pretrained('distilbert-base-uncased')
model = build_model(bert_model, max_len=400)

model.summary()
x_train
history = model.fit(x_train,y_train,batch_size = 32 ,validation_data=(x_test,y_test),epochs = 3)

model.evaluate(x_test,y_test)[1]*100
pred=model.predict(x_test)

pred = np.round(pred).astype(int)
from sklearn.metrics import confusion_matrix, accuracy_score

cm = confusion_matrix(y_test,pred)

cm
score = accuracy_score( y_test, pred)

print(score)