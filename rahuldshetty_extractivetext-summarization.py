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

path = "/kaggle/input/news-summary/news_summary_more.csv"

df = pd.read_csv(path,encoding = "ISO-8859-1")

print(df.head())

print(df.describe())

contraction_mapping = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not",



                           "didn't": "did not", "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not",



                           "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",



                           "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would",



                           "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would",



                           "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam",



                           "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have",



                           "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock",



                           "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have",



                           "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is",



                           "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as",



                           "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would",



                           "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have",



                           "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have",



                           "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are",



                           "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",



                           "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is",



                           "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have",



                           "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have",



                           "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all",



                           "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have",



                           "you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have",



                           "you're": "you are", "you've": "you have"}

import re

from nltk.corpus import stopwords



stop_words = stopwords.words('english')



def preprocess(text):

    text = text.lower() # lowercase

    text = text.split() # convert have'nt -> have not

    for i in range(len(text)):

        word = text[i]

        if word in contraction_mapping:

            text[i] = contraction_mapping[word]

    text = " ".join(text)

    text = text.split()

    newtext = []

    for word in text:

        if word not in stop_words:

            newtext.append(word)

    text = " ".join(newtext)

    text = text.replace("'s",'') # convert your's -> your

    text = re.sub(r'\(.*\)','',text) # remove (words)

    text = re.sub(r'[^a-zA-Z0-9. ]','',text) # remove punctuations

    text = re.sub(r'\.',' . ',text)

    return text



sample = "(hello) hi there .man tiger caller who's that isn't it ? WALL-E"

print(preprocess(sample))
df['headlines'] = df['headlines'].apply(lambda x:preprocess(x))

df['text'] = df['text'].apply(lambda x:preprocess(x))

print(df['headlines'][20],df['text'][20])
def get_out_vector(text,summary,n=40):

    new_vec  = np.zeros(n)

    for txt in summary.split():

        if txt in text:

            for i,word in enumerate(text.split()):

                if word == txt:

                    new_vec[i] = 1

    return new_vec



def get_summary(text,new_vec,thresh = 0.5):

    summary = []

    for i,word in enumerate(text.split()):

        if new_vec[i] >= thresh:

            summary.append(word)

    return " ".join(summary)



sample_text = "hola this is a hola hola hola okay test nice"

sample_summary = "hola nice okay"

vec = (get_out_vector(sample_text,sample_summary,15))

print( get_summary(sample_text,vec) )
print(df.shape)
# label creation

rows = df.shape[0]

no_words_per_sentence = 80 # max words 

y = np.zeros((rows,no_words_per_sentence))



for i in range(rows):

    text = df['text'][i]

    summary = df['headlines'][i]

    vec = get_out_vector(text,summary,no_words_per_sentence)

    y[i] = vec

print(y[0])

print(y[900])
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.feature_extraction.text import CountVectorizer
no_input_words = 80

x = np.zeros((rows,no_input_words))

vectorizer = TfidfVectorizer(use_idf=True,norm='l1')

vectorizer.fit_transform(df['text'])

names = vectorizer.get_feature_names()

idf = vectorizer.idf_



maps = { names[i]:idf[i] for i in range(len(names)) }



for j in range(rows):

    text = df['text'][j]

    vec = np.zeros(no_input_words)

    for i,word in enumerate(text.split()):

        if word in maps:

            vec[i] = maps[word]

    x[j] = vec

print(x[0],x[50])    
print(x.shape,y.shape)
rows = 98401

x = x.reshape((rows,no_input_words,1))

y = y.reshape((rows,no_words_per_sentence,1))
from keras.models import Sequential

from keras.models import Model

from keras.layers import LSTM

from keras.layers import Dense

from keras.layers import RepeatVector

from keras.layers import TimeDistributed





model = Sequential()

model.add(LSTM(128, activation='relu', input_shape=(no_input_words,1)))

model.add(RepeatVector(no_words_per_sentence))

model.add(LSTM(256, activation='relu',return_sequences=True))

model.add(LSTM(128, activation='relu', return_sequences=True))

model.add(TimeDistributed(Dense(1,activation='sigmoid')))

model.compile(optimizer='adam', loss='mse')

model.summary()
h = model.fit(x,y,validation_split = 0.2, epochs = 10, batch_size = 128)
%matplotlib inline

import matplotlib.pyplot as plt

plt.plot(h.history['acc'])

plt.plot(h.history['val_acc'])

plt.title('Model accuracy')

plt.show()



plt.plot(h.history['loss'])

plt.plot(h.history['val_loss'])

plt.title('Model Loss')

plt.show()
model.save('model.h5')
ids = 0

sample_text = df['text'][ids]

vec = x[ids]

res = model.predict(vec)[0]

res_row = res.shape[0]

res_col = res.shape[1]

res = res.reshape((res_row,res_col))

out = get_summary(sample_text,res)

print(out)