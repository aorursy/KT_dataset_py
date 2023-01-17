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
import pandas as pd;pd.set_option('display.max_column',300)

import numpy as np

import seaborn as sns

import pylab as plt

%matplotlib inline

from keras.preprocessing.text import Tokenizer

from keras.preprocessing import sequence

import re

from keras.models import Sequential

from keras.layers import LSTM,Dropout,Dense,Embedding,Flatten
MAX_FEATURE = 100000
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
train.head(5)
train.loc[train['target']>=0.5,'target'] = 1

train.loc[train['target']<0.5,'target'] = 0
train.isnull().sum()/len(train)
list_all = list(train['comment_text'])+list(test['comment_text'])

from tqdm import tqdm

length_list = []

word_all = []

for i in tqdm(list_all):

    length_list.append(len(i))

    for j in i.split():

        word_all.append(j)

set_all = set(word_all)
print("a sentence has max words:",max(length_list))

print("a sentence has min words:",min(length_list))

print('a sentence has average words:',int(sum(length_list)/len(length_list)))

print('there are total',len(set_all),'unique words')
list_1 = []

temp = ''

for k in set_all:

    common = 'qwertyuiopasdfghjklzxcvbnmQWERTYUIOPASDFGHJKLZXCVBNM'

    for l in common:

        k = k.replace(l,'')

    list_1.append(k)

# set(list_1)
str_1 = ''

for i in list_1:

    str_1+=i

set_1 = set(str_1)
punct_1 = ''

for i in set_1:

    punct_1+=i

punct_1
misspell_dict = {"aren't": "are not", "can't": "cannot", "couldn't": "could not","dom't":"do not",

                 "didn't": "did not", "does'nt": "does not","doesn't": "does not", "don't": "do not",

                 "hadn't": "had not", "hasn't": "has not", "haven't": "have not",

                 "he'd": "he would", "he'll": "he will", "he's": "he is","here's":"here is",

                 "i'd": "I had", "i'll": "I will", "i'm": "I am", "isn't": "is not",

                 "it's": "it is", "it'll": "it will", "i've": "I have", "let's": "let us",

                 "mightn't": "might not", "mustn't": "must not", "shan't": "shall not",

                 "she'd": "she would", "she'll": "she will", "she's": "she is",

                 "shouldn't": "should not", "that's": "that is", "that''s":"that is","there's": "there is",

                 "they'd": "they would", "they'll": "they will", "they're": "they are",

                 "they've": "they have", "we'd": "we would", "we're": "we are","wasn't":"was not",

                 "weren't": "were not", "we've": "we have", "what'll": "what will",

                 "what're": "what are", "what's": "what is", "what've": "what have",

                 "where's": "where is", "who'd": "who would", "who'll": "who will",

                 "who're": "who are", "who's": "who is", "who've": "who have",

                 "won't": "will not", "wouldn't": "would not", "you'd": "you would",

                 "you'll": "you will", "you're": "you are", "you've": "you have","opp's":"opps",

                 "'re": " are", "wasn't": "was not", "we'll": " will", "tryin'": "trying",

                "ican't":"I can not","are't":"are not","dind't":"did not","whataboutism":"what about ism",

                "ya'know":"you know","havent't":"have not","how'd":"how had"}
def preprocess_1(data):

    '''

    Credit goes to https://www.kaggle.com/gpreda/jigsaw-fast-compact-solution

    '''

    punct = punct_1.replace("'",'')#首先去掉除了单引号的其他字符。单引号在执行完误拼后再去除。

    def clean_special_chars(text, punct):

        for p in punct:

            text = text.replace(p, ' ')

        return text



    data = data.astype(str).apply(lambda x: clean_special_chars(x, punct))

    return data
def _get_misspell(misspell_dict):

    misspell_re = re.compile('(%s)' % '|'.join(misspell_dict.keys()))

    return misspell_dict, misspell_re





def replace_typical_misspell(text):

    misspellings, misspellings_re = _get_misspell(misspell_dict)



    def replace(match):

        return misspellings[match.group(0)]



    return misspellings_re.sub(replace, text)
def clean_numbers(x):

    return re.sub('\d+', ' ', x)
# lower

train['comment_text'] = train['comment_text'].str.lower()

test['comment_text'] = test['comment_text'].str.lower()

# clean numbers

import re

train['comment_text'] = train['comment_text'].apply(clean_numbers)

test['comment_text'] = test['comment_text'].apply(clean_numbers)

# clean the text

train['comment_text'] = preprocess_1(train['comment_text'])

test['comment_text'] = preprocess_1(test['comment_text'])

# clean misspellings

train['comment_text'] = train['comment_text'].apply(replace_typical_misspell)

test['comment_text'] = test['comment_text'].apply(replace_typical_misspell)
def preprocess_2(data):

    '''

    Credit goes to https://www.kaggle.com/gpreda/jigsaw-fast-compact-solution

    '''

    punct = "'"#首先去掉除了单引号的其他字符。单引号在执行完误拼后再去除。

    def clean_special_chars(text, punct):

        for p in punct:

            text = text.replace(p, ' ')

        return text



    data = data.astype(str).apply(lambda x: clean_special_chars(x, punct))

    return data
# clean the text again

train['comment_text'] = preprocess_2(train['comment_text'])

test['comment_text'] = preprocess_2(test['comment_text'])
train.head()
train['comment_text'].isnull().sum()
X_train = train['comment_text']

X_test = test['comment_text']

y_train = train['target']
tokenizer = Tokenizer(num_words=MAX_FEATURE)
tokenizer.fit_on_texts(list(X_train)+list(X_test))

X_train = tokenizer.texts_to_sequences(X_train)

X_test = tokenizer.texts_to_sequences(X_test)
X_train = sequence.pad_sequences(X_train,maxlen=220)

X_test = sequence.pad_sequences(X_test,maxlen=220)

X_train[:2]
model = Sequential()

model.add(Embedding(input_dim=MAX_FEATURE,output_dim=300))

model.add(LSTM(units=128,dropout=0.2))

model.add(Dense(1))

model.summary()
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

model.fit(X_train,y_train,batch_size=512)
predictions = model.predict(X_test)
df_submit = pd.read_csv('../input/sample_submission.csv')

df_submit.prediction = predictions

df_submit.to_csv('submission.csv', index=False)