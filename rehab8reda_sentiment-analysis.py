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
import pandas as pd 

import numpy as np 

import matplotlib.pyplot as plt 

import re

from gensim.models import KeyedVectors

import tensorflow as tf

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras import layers,models 

from keras.callbacks import ModelCheckpoint, EarlyStopping

from sklearn.model_selection import train_test_split

from keras.models import Model

from keras.layers import Input ,Dense ,Flatten,Dropout,Embedding ,Conv1D ,MaxPooling1D

from keras.layers.merge import concatenate
#Reading Data

df = pd.read_csv('../input/imdb-dataset-of-50k-movie-reviews/IMDB Dataset.csv')
df.head()
df['sentiment']=df['sentiment'].apply(lambda x: 1 if x=='positive' else 0)

df['sentiment'][:5]
df['review'].iloc[0]
## check if there null data

df.isnull().sum()

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
df['review'][0]

max_len=max(df['review'].apply(lambda x:len(x)))

max_len
x_train,x_val,y_train,y_val = train_test_split(df['review'],df['sentiment'], test_size = 0.2)
x_val.iloc[0]


MAX_LENGTH = 180

tokenizer = Tokenizer()

tokenizer.fit_on_texts(df['review'])

x_train_sequences = tokenizer.texts_to_sequences(x_train)

x_val_sequences = tokenizer.texts_to_sequences(x_val)

x_train_padded = pad_sequences(x_train_sequences, maxlen = MAX_LENGTH, padding = 'post')

x_val_padded = pad_sequences(x_val_sequences, maxlen = MAX_LENGTH, padding = 'post')



x_train_padded[0]
vocab_size=len(tokenizer.word_index)+1

vocab_size
### define model 

length=180



# channel 1

inputs1 = Input(shape=(length,))

embedding1 = Embedding(vocab_size, 300)(inputs1)

conv1 = Conv1D(filters=32, kernel_size=4, activation='relu')(embedding1)

drop1 = Dropout(0.5)(conv1)

pool1 = MaxPooling1D(pool_size=2)(drop1)

flat1 = Flatten()(pool1)

# channel 2

inputs2 = Input(shape=(length,))

embedding2 = Embedding(vocab_size, 300)(inputs2)

conv2 = Conv1D(filters=32, kernel_size=6, activation='relu')(embedding2)

drop2 = Dropout(0.5)(conv2)

pool2 = MaxPooling1D(pool_size=2)(drop2)

flat2 = Flatten()(pool2)

# channel 3

inputs3 = Input(shape=(length,))

embedding3 = Embedding(vocab_size, 300)(inputs3)

conv3 = Conv1D(filters=32, kernel_size=8, activation='relu')(embedding3)

drop3 = Dropout(0.5)(conv3)

pool3 = MaxPooling1D(pool_size=2)(drop3)

flat3 = Flatten()(pool3)

# merge

merged = concatenate([flat1, flat2, flat3])

# interpretation

dense1 = Dense(10, activation='relu')(merged)

outputs = Dense(1, activation='sigmoid')(dense1)

model = Model(inputs=[inputs1, inputs2, inputs3], outputs=outputs)

# compile

model.compile(loss='binary_crossentropy', optimizer='RMSprop', metrics=['accuracy'])

# summarize

model.summary()
##plot model

from keras.utils.vis_utils import plot_model

plot_model(model, show_shapes=True, to_file='model.png')
history = model.fit([x_train_padded,x_train_padded,x_train_padded],np.array(y_train),epochs=5,validation_data=([x_val_padded,x_val_padded,x_val_padded],np.array(y_val)))


plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title('Model accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()



plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('Model loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper right')

plt.show()