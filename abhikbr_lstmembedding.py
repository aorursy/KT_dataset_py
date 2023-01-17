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
import matplotlib.pyplot as plt

import keras 
import re
train = pd.read_csv('../input/nlp-getting-started/train.csv')

test = pd.read_csv('../input/nlp-getting-started/test.csv')
train.head()
train.isnull().sum()
train.info()
def remove_url(text):

    url = re.compile(r"https?://\S+|www\.\S+")

    return url.sub(r"", text)

                     

def remove_html(text):

    html = re.compile(r"<.*?>")

    return html.sub(r"", text)                 
def remove_emoji(string):

    emoji_pattern = re.compile(

        "["

        u"\U0001F600-\U0001F64F"  # emoticons

        u"\U0001F300-\U0001F5FF"  # symbols & pictographs

        u"\U0001F680-\U0001F6FF"  # transport & map symbols

        u"\U0001F1E0-\U0001F1FF"

        u"\U00002702-\U000027B0"

        u"\U000024C2-\U0001F251"# flags (iOS)

        "]+",

        flags = re.UNICODE

    )

    return emoji_pattern.sub(r"", string)
import string



def remove_punct(text):

    table = str.maketrans("","",string.punctuation)

    return text.translate(table)
train['text'] = train.text.map(lambda x: remove_url(x))

train['text'] = train.text.map(lambda x: remove_html(x))

train['text'] = train.text.map(lambda x: remove_emoji(x))

train['text'] = train.text.map(lambda x: remove_punct(x))
from nltk.corpus import stopwords



stop = set(stopwords.words("english"))



def remove_stopwords(text):

    text = [word.lower() for word in text.split() if word.lower() not in stop]

    

    return " ".join(text)
train['text'] = train['text'].map(remove_stopwords)
train.text
from collections import Counter



def counter_word(text):

    count = Counter()

    for i in text.values:

        for word in i.split():

            count[word]+=1

            

    return count
text =  train.text

counter = counter_word(text)
len(counter)
counter
num_words = len(counter)

max_length = 20


train_size = int(train.shape[0]*0.8)

train_sentences = train.text[:train_size]

train_labels = train.target[:train_size]



test_sentences = train.text[train_size:]

test_labels = train.target[train_size:]
train_sentences[0]
from keras.preprocessing.text import Tokenizer



tokenizer = Tokenizer(num_words = num_words)

tokenizer.fit_on_texts(train_sentences)
word_index = tokenizer.word_index

word_index
train_sequences = tokenizer.texts_to_sequences(train_sentences)
train_sequences[0]
from keras.preprocessing.sequence import pad_sequences



train_padded = pad_sequences(

train_sequences,maxlen=max_length,padding="post",truncating="post"

)
test_sequences = tokenizer.texts_to_sequences(test_sentences)

test_padded = pad_sequences(

test_sequences,maxlen=max_length,padding="post",truncating="post"

)
train_padded[0]
print(train.text[0])

print(train_sequences[0])
reverse_word_index = dict([(value,key) for (key , value) in word_index.items()])
def decode(text):

    return " ".join([reverse_word_index.get(i,"2") for i in text])
decode(train_sequences[0])
train_padded.shape
test_padded.shape
from keras.models import Sequential

from keras.layers import Embedding, LSTM, Dense, Dropout

from keras.initializers import Constant

from keras.optimizers import Adam



model = Sequential()



model.add(Embedding(num_words, 32, input_length = max_length))

model.add(LSTM(64,dropout=0.1))

model.add(Dense(1,activation="sigmoid"))



optimizer = Adam(learning_rate=3e-4)



model.compile(loss = "binary_crossentropy",optimizer = optimizer , metrics=["accuracy"])
model.summary()
history = model.fit(train_padded,train_labels,epochs=20,validation_data=(test_padded,test_labels))