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
!pip install textblob
# import NLP modules

import re

import nltk

from textblob import TextBlob

from nltk.corpus import stopwords

from nltk.stem import PorterStemmer

from textblob import Word

from nltk.util import ngrams

from wordcloud import WordCloud, STOPWORDS

from nltk.tokenize import word_tokenize
#Read train.csv file

df_train = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")
#Text Preprocessing

def clean_text(col):

  #convert words to lowercase

  col['text'] = col['text'].apply(lambda x: " ".join(x.lower() for x in x.split())) 

  #Removes unicode strings like "\u002c" and "x96"

  col['text'] = col['text'].str.replace(r'(\\u[0-9A-Fa-f]+)','')

  col['text'] = col['text'].str.replace(r'[^\x00-\x7f]','')

  #convert any url to URL

  col['text'] = col['text'].str.replace('((www\.[^\s]+)|(https?://[^\s]+)|(http?://[^\s]+))','URL')

  #Convert any @Username to "AT_USER"

  col['text'] = col['text'].str.replace('@[^\s]+','AT_USER')

  #Remove additional white spaces

  col['text'] = col['text'].str.replace('[\s]+',' ')

  col['text'] = col['text'].str.replace('[\n]+',' ')

  #Remove not alphanumeric symbols white spaces

  col['text'] = col['text'].str.replace(r'[^\w]', ' ')

  #Removes hastag in front of a word """

  col['text'] = col['text'].str.replace(r'#([^\s]+)', r'\1')

  #Removes:) or :(

  col['text'] = col['text'].str.replace(r':\)',"")

  col['text'] = col['text'].str.replace(r':\(',"")

  #remove numbers

  col['text'] = col['text'].apply(lambda x: " ".join(x for x in x.split() if not x.isdigit()))

  #remove multiple exclamation

  col['text'] = col['text'].str.replace(r"(\!)\1+", ' ')

  #remove multiple question marks

  col['text'] = col['text'].str.replace(r"(\?)\1+", ' ')

  #remove multistop

  col['text'] = col['text'].str.replace(r"(\.)\1+", ' ')

  #lemma

  from textblob import Word

  col['text'] = col['text'].apply(lambda x: " ".join(Word(word).lemmatize() for word in x.split())) 

  #Removes emoticons from text

  col['text'] = col['text'].str.replace(':\)|;\)|:-\)|\(-:|:-D|=D|:P|xD|X-p|\^\^|:-*|\^\.\^|\^\-\^|\^\_\^|\,-\)|\)-:|:\'\(|:\(|:-\(|:\S|T\.T|\.\_\.|:<|:-\S|:-<|\*\-\*|:O|=O|=\-O|O\.o|XO|O\_O|:-\@|=/|:/|X\-\(|>\.<|>=\(|D:', "r")

  #trim

  col['text'] = col['text'].str.strip('\'"')
#clean train data

input_train = df_train.copy()

clean_text(input_train)
#prepare train data to use in fasstext

input_train['target_str'] = input_train['target'].replace(1, 'disaster')

input_train['target_str'] = input_train['target_str'].replace(0, 'nodisaster')



col = ['target_str', 'text']

fft_input = input_train[col]
# prepare correct format text for fasstex

# __label__ text



fft_input['target_str'] = ['__label__'+ s for s in input_train['target_str']]
fft_input['target_str'].head(30)
# Get 70% of the data

n = (fft_input.shape[0] * 7)/10

n = int(round(n))



# Split the file into 70% train and 30% test

fft_train = fft_input[:n] 

fft_test = fft_input[n:] 

print(fft_train.shape, fft_test.shape)
# convert the train and the test into text files in order to work with fasttext

fft_train.to_csv(r'fft_train.txt', index=False, sep=' ', header=False)

fft_test.to_csv(r'fft_test.txt', index=False, sep=' ', header=False)
train = open("fft_train.txt", "r")
!pip install fasttext
# Apply fasttest to the train data in order to VECTORIZE the data 

import fasttext

model = fasttext.train_supervised("fft_train.txt", lr=0.1, dim=100, epoch=5, word_ngrams=2, loss='softmax')
# Evaluate the test data

result = model.test('fft_test.txt')

result
#check precision and recall

print('Precision:', result[1])

print('Recall:', result[2]) 
#Read test.csv file

df_test = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")
#clean test/unseen data

input_test = df_test.copy()

clean_text(input_test)
#Take only is and text column of the test.csv

df_submit = input_test[['id','text']]

df_submit.head(10)
#predict the test/unseen labels

targets=[]

for row in df_submit['text']:

    pred_label=model.predict(row, k=-1, threshold=0.5)[0][0]

    if (pred_label == '__label__nodisaster'):

      pred_label = 0

    else:

      pred_label = 1

    targets.append(pred_label)     



# you add the list to the dataframe, then save the datframe to new csv

df_submit['target']=targets

df_submit.to_csv('tweet_submission.csv',sep=',',index=False)