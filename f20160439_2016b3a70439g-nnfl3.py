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
train_df = pd.read_csv('/kaggle/input/nnfl-lab-3-nlp/nlp_train.csv')
test_df = pd.read_csv('/kaggle/input/nnfl-lab-3-nlp/_nlp_test.csv')
np.random.seed(0)

import pandas as pd
import numpy as np
import nltk

from numpy import array
from numpy import asarray
from numpy import zeros
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers.core import Activation, Dropout, Dense
from keras.layers import Flatten
from keras.layers import GlobalMaxPooling1D
from tensorflow.python.keras.layers import Bidirectional
from tensorflow.python.keras.layers.recurrent import LSTM
from tensorflow.python.keras.layers.embeddings import Embedding
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report
from sklearn.utils import shuffle
import re
import string
def clean_tweet(text):
    ''' Pre process and convert texts to a list of words '''
    text = str(text)
    text = text.lower()

    # Clean the text
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"#(\w+)", ' ', text, flags=re.MULTILINE)
    text = re.sub(r"@(\w+)", ' ', text, flags=re.MULTILINE)
    text = re.sub(r"http\S+", "", text)

   

    return text


train_df['tweet']=train_df['tweet'].apply(lambda x: clean_tweet(str(x)))
test_df['tweet']=test_df['tweet'].apply(lambda x: clean_tweet(str(x)))
import spacy
nlp = spacy.load('en_core_web_lg')

def give_emb(x):
    doc = nlp(x)
    return doc.vector
X_train_emb = train_df['tweet'].apply(give_emb)

data=[]
for i in X_train_emb:
    data.append(i)
data = np.asarray(data)
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python.keras.layers.recurrent import LSTM
def build_model():
    model = keras.Sequential([
    #layers.Dense(128, activation='relu', input_shape=[300]),
    layers.Dense(64, activation='relu',input_shape=[300]),
    layers.Dense(32, activation='relu'),
    layers.Dense(1)
  ])


    optimizer = tf.keras.optimizers.Adam(0.0001)

    model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae', 'mse'])
    return model
model = build_model()

EPOCHS = 70

history = model.fit(
  data,train_df['offensive_language'].values,
  epochs=EPOCHS, validation_split = 0.2, verbose=1)
y_p = model.predict(data)
y_p = y_p.reshape(-1,)
y_p = [3 if x >= 3 else x for x in y_p]
y_p = [0 if x <= 0 else x for x in y_p]
from sklearn.metrics import mean_squared_error
print(mean_squared_error(y_p,train_df['offensive_language']))
X_test_emb = test_df['tweet'].apply(give_emb)

test = []
for i in X_test_emb:
    test.append(i)
test = np.asarray(test)
y_sub = model.predict(test)

pd.Series(y_sub.reshape(-1)).hist()
y_sub= y_sub.reshape(-1,)
y_sub = [3 if x >= 3 else x for x in y_sub]
#y_sub = [3 if x >= 2.8 else x for x in y_sub]
y_sub= [0 if x <= 0 else x for x in y_sub]
pd.Series(y_sub).hist()
sub = pd.read_csv('/kaggle/input/nnfl-lab-3-nlp/_nlp_test.csv')
sub['offensive_language'] = y_sub
sub.to_csv('sub1001.csv',index=False)
from IPython.display import FileLink
FileLink(r'sub1001.csv')
model.save_weights("modellab3.h5")
FileLink(r'modellab3.h5')
submission_df=sub
from IPython.display import HTML
import pandas as pd
import numpy as np
import base64
def create_download_link(df, title = "Download CSV file", filename = "data.csv"):
    csv = submission_df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode())
    payload = b64.decode()
    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'
    html = html.format(payload=payload,title=title,filename=filename)
    return HTML(html)
create_download_link(submission_df)
