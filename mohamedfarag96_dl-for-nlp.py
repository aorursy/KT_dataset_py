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
import re

import nltk

import spacy

import string
#Extracting training data

train_data = pd.read_csv('../input/nlp-getting-started/train.csv')
#To get the first 5 rows of our data

train_data.head()
#To get more insights about data

train_data.info()
#To know the number of positive and negative samples at the training dataset

train_data['target'].value_counts()
train_data['target'].hist()
#First we will drop keyword, location and id

train_data.drop(['keyword','location', 'id'], axis = 1,inplace = True)

train_data
train_data['text'] = train_data['text'].astype(str)

train_data.head()
train_data['text'] = train_data['text'].str.lower()

train_data.head()
# train_data.drop(["text_lower"], axis=1, inplace=True)

PUNCT_TO_REMOVE = string.punctuation



def remove_punctuation(text):

    """custom function to remove the punctuation"""

    return text.translate(str.maketrans('', '', PUNCT_TO_REMOVE))



train_data["text"] = train_data["text"].apply(lambda text: remove_punctuation(text))

train_data.head()
from nltk.corpus import stopwords

", ".join(stopwords.words('english'))
STOPWORDS = set(stopwords.words('english'))

def remove_stopwords(text):

    """custom function to remove the stopwords"""

    return " ".join([word for word in str(text).split() if word not in STOPWORDS])



train_data["text"] = train_data["text"].apply(lambda text: remove_stopwords(text))

train_data.head()
from collections import Counter

cnt = Counter()

for text in train_data["text"].values:

    for word in text.split():

        cnt[word] += 1

        

cnt.most_common(10)
FREQWORDS = set([w for (w, wc) in cnt.most_common(10)])

def remove_freqwords(text):

    """custom function to remove the frequent words"""

    return " ".join([word for word in str(text).split() if word not in FREQWORDS])



train_data["text"] = train_data["text"].apply(lambda text: remove_freqwords(text))

train_data.head()
from nltk.stem.porter import PorterStemmer





stemmer = PorterStemmer()

def stem_words(text):

    return " ".join([stemmer.stem(word) for word in text.split()])



train_data["text"] = train_data["text"].apply(lambda text: stem_words(text))

train_data.head()
n_rare_words = 10

RAREWORDS = set([w for (w, wc) in cnt.most_common()[:-n_rare_words-1:-1]])

def remove_rarewords(text):

    """custom function to remove the rare words"""

    return " ".join([word for word in str(text).split() if word not in RAREWORDS])



train_data["text"] = train_data["text"].apply(lambda text: remove_rarewords(text))

train_data.head()
from nltk.stem import WordNetLemmatizer



lemmatizer = WordNetLemmatizer()

def lemmatize_words(text):

    return " ".join([lemmatizer.lemmatize(word) for word in text.split()])



train_data["text"] = train_data["text"].apply(lambda text: lemmatize_words(text))

train_data.head()
from nltk.corpus import wordnet

from nltk.stem import WordNetLemmatizer



lemmatizer = WordNetLemmatizer()

wordnet_map = {"N":wordnet.NOUN, "V":wordnet.VERB, "J":wordnet.ADJ, "R":wordnet.ADV}

def lemmatize_words(text):

    pos_tagged_text = nltk.pos_tag(text.split())

    return " ".join([lemmatizer.lemmatize(word, wordnet_map.get(pos[0], wordnet.NOUN)) for word, pos in pos_tagged_text])



train_data["text"] = train_data["text"].apply(lambda text: lemmatize_words(text))

train_data.head()
def remove_emoji(string):

    emoji_pattern = re.compile("["

                           u"\U0001F600-\U0001F64F"  # emoticons

                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs

                           u"\U0001F680-\U0001F6FF"  # transport & map symbols

                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)

                           u"\U00002702-\U000027B0"

                           u"\U000024C2-\U0001F251"

                           "]+", flags=re.UNICODE)

    return emoji_pattern.sub(r'', string)



train_data["text"] = train_data["text"].apply(lambda text: remove_emoji(text))

train_data.head()
def remove_emoticons(text):

    emoticon_pattern = re.compile(u'(' + u'|'.join(k for k in EMOTICONS) + u')')

    return emoticon_pattern.sub(r'', text)



train_data["text"] = train_data["text"].apply(lambda text: remove_emoticons(text))

train_data.head()
def remove_urls(text):

    url_pattern = re.compile(r'https?://\S+|www\.\S+')

    return url_pattern.sub(r'', text)

train_data["text"] = train_data["text"].apply(lambda text: remove_urls(text))

train_data.head()

def remove_html(text):

    html_pattern = re.compile('<.*?>')

    return html_pattern.sub(r'', text)



train_data["text"] = train_data["text"].apply(lambda text: remove_html(text))

train_data.head()

chat_words_map_dict = {}

chat_words_list = []

for line in chat_words_str.split("\n"):

    if line != "":

        cw = line.split("=")[0]

        cw_expanded = line.split("=")[1]

        chat_words_list.append(cw)

        chat_words_map_dict[cw] = cw_expanded

chat_words_list = set(chat_words_list)



def chat_words_conversion(text):

    new_text = []

    for w in text.split():

        if w.upper() in chat_words_list:

            new_text.append(chat_words_map_dict[w.upper()])

        else:

            new_text.append(w)

    return " ".join(new_text)



train_data["text"] = train_data["text"].apply(lambda text: chat_words_conversion(text))

train_data.head()

#Split our target & samples from the data



x_train = train_data['text']



y_train = train_data['target']



x_train , y_train
y_train = np.array(y_train)

x_train = np.array(x_train)
print(x_train.shape)

print('\n\n')

print(x_train[20:50])
print(y_train)

print(y_train.shape)
import tensorflow as tf

import keras



from tensorflow.keras.preprocessing.text import Tokenizer

from tensorflow.keras.preprocessing.sequence import pad_sequences



vocab_size = 10000

embedding_dim = 32

max_length = 150

trunc_type='post'

oov_tok = "<UNK>"

punc = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
tokenizer = Tokenizer(num_words = vocab_size,oov_token=oov_tok,lower = True, filters = punc)

tokenizer.fit_on_texts(x_train)



word_index = tokenizer.word_index



print( 'The size of the bag of wrods is:' ,len(word_index))

print('\n\n')

print(word_index)
x_train_seq = tokenizer.texts_to_sequences(x_train)

x_train_pad = pad_sequences(x_train_seq,maxlen=max_length, truncating=trunc_type)

print(x_train_pad[0])

print(x_train_pad.shape)
model = tf.keras.Sequential([

    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length,),

    tf.keras.layers.Conv1D(16,3, strides = 1, padding = 'same'),

    tf.keras.layers.LSTM(64,return_sequences = True),

    tf.keras.layers.LSTM(64),

#     tf.keras.layers.GlobalAveragePooling1D(),

    tf.keras.layers.Dense(20, activation='relu'),

    tf.keras.layers.Dense(1, activation='sigmoid')

])

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

model.summary()
num_epochs = 20

model.fit(x_train_pad, y_train, epochs=num_epochs)
test = pd.read_csv('../input/nlp-getting-started/test.csv')

test

x_test = test['text']

x_test = np.array(list(x_test))

x_test.shape
x_test_seq = tokenizer.texts_to_sequences(x_test)

x_test_pad = pad_sequences(x_test_seq,maxlen=max_length, truncating=trunc_type)



x_test_pad.shape
prediction = []

predictions = model.predict(x_test_pad)
for i in predictions:

    if i >= 0.5:

        prediction.append(1)

    else:

        prediction.append(0)

        

len(prediction)   
submission =  pd.DataFrame({

        "id": test['id'],

        "target": prediction

    })



submission.to_csv('submission12.csv', index=False)