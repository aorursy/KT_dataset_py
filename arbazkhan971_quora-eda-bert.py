import os

import torch

from transformers import *

from transformers import BertTokenizer, BertModel,BertForSequenceClassification,AdamW

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from keras.preprocessing.sequence import pad_sequences

from sklearn.model_selection import train_test_split

from tqdm import tqdm, trange

import tensorflow as tf

import numpy as np

import matplotlib.pyplot as plt
from fastai.text import * 

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from subprocess import check_output



%matplotlib inline

import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls

import pandas as pd

import numpy as np

import nltk

from nltk.corpus import stopwords

from nltk.stem import SnowballStemmer

import re

from string import punctuation

print(check_output(["ls", "../input"]).decode("utf8"))



df = pd.read_csv("../input/quoraquestions/data.csv").fillna("")

df.head() 
df.info()
df.shape
df.groupby("is_duplicate")['id'].count().plot.bar()
df
dfs = df[0:2500]

dfs.groupby("is_duplicate")['id'].count().plot.bar()
df
stop_words = ['the','a','an','and','but','if','or','because','as','what','which','this','that','these','those','then',

              'just','so','than','such','both','through','about','for','is','of','while','during','to','What','Which',

              'Is','If','While','This']
data.dropna(inplace=True)
data
##
def text_to_wordlist(text, remove_stop_words=True, stem_words=False):

    # Clean the text, with the option to remove stop_words and to stem words.



    # Clean the text

    text = re.sub(r"[^A-Za-z0-9]", " ", text)

    text = re.sub(r"what's", "", text)

    text = re.sub(r"What's", "", text)

    text = re.sub(r"\'s", " ", text)

    text = re.sub(r"\'ve", " have ", text)

    text = re.sub(r"can't", "cannot ", text)

    text = re.sub(r"n't", " not ", text)

    text = re.sub(r"I'm", "I am", text)

    text = re.sub(r" m ", " am ", text)

    text = re.sub(r"\'re", " are ", text)

    text = re.sub(r"\'d", " would ", text)

    text = re.sub(r"\'ll", " will ", text)

    text = re.sub(r"60k", " 60000 ", text)

    text = re.sub(r" e g ", " eg ", text)

    text = re.sub(r" b g ", " bg ", text)

    text = re.sub(r"\0s", "0", text)

    text = re.sub(r" 9 11 ", "911", text)

    text = re.sub(r"e-mail", "email", text)

    text = re.sub(r"\s{2,}", " ", text)

    text = re.sub(r"quikly", "quickly", text)

    text = re.sub(r" usa ", " America ", text)

    text = re.sub(r" USA ", " America ", text)

    text = re.sub(r" u s ", " America ", text)

    text = re.sub(r" uk ", " England ", text)

    text = re.sub(r" UK ", " England ", text)

    text = re.sub(r"india", "India", text)

    text = re.sub(r"switzerland", "Switzerland", text)

    text = re.sub(r"china", "China", text)

    text = re.sub(r"chinese", "Chinese", text) 

    text = re.sub(r"imrovement", "improvement", text)

    text = re.sub(r"intially", "initially", text)

    text = re.sub(r"quora", "Quora", text)

    text = re.sub(r" dms ", "direct messages ", text)  

    text = re.sub(r"demonitization", "demonetization", text) 

    text = re.sub(r"actived", "active", text)

    text = re.sub(r"kms", " kilometers ", text)

    text = re.sub(r"KMs", " kilometers ", text)

    text = re.sub(r" cs ", " computer science ", text) 

    text = re.sub(r" upvotes ", " up votes ", text)

    text = re.sub(r" iPhone ", " phone ", text)

    text = re.sub(r"\0rs ", " rs ", text) 

    text = re.sub(r"calender", "calendar", text)

    text = re.sub(r"ios", "operating system", text)

    text = re.sub(r"gps", "GPS", text)

    text = re.sub(r"gst", "GST", text)

    text = re.sub(r"programing", "programming", text)

    text = re.sub(r"bestfriend", "best friend", text)

    text = re.sub(r"dna", "DNA", text)

    text = re.sub(r"III", "3", text) 

    text = re.sub(r"the US", "America", text)

    text = re.sub(r"Astrology", "astrology", text)

    text = re.sub(r"Method", "method", text)

    text = re.sub(r"Find", "find", text) 

    text = re.sub(r"banglore", "Banglore", text)

    text = re.sub(r" J K ", " JK ", text)

    

    # Remove punctuation from text

    text = ''.join([c for c in text if c not in punctuation])

    

    # Optionally, remove stop words

    if remove_stop_words:

        text = text.split()

        text = [w for w in text if not w in stop_words]

        text = " ".join(text)

    

    # Optionally, shorten words to their stems

    if stem_words:

        text = text.split()

        stemmer = SnowballStemmer('english')

        stemmed_words = [stemmer.stem(word) for word in text]

        text = " ".join(stemmed_words)

    

    # Return a list of words

    return(text)
data[1]
def process_questions(question_list, questions, question_list_name, dataframe):

    '''transform questions and display progress'''

    for question in questions:

        question_list.append(text_to_wordlist(question))

        if len(question_list) % 100000 == 0:

            progress = len(question_list)/len(dataframe) * 100

            print("{} is {}% complete.".format(question_list_name, round(progress, 1)))
data1 = []

process_questions(data1, data, 'train_question1', data)
data1
data = pd.Series(data1)
data
data
data_up = df[:20000]
data_up
df.columns
BATCH_SIZE=10
data_lm = (TextList.from_df(data_up)

           #Inputs: all the text files in path

            .split_by_rand_pct(0.15)

           #We randomly split and keep 10% for validation

            .label_for_lm()           

           #We want to do a language model so we label accordingly

            .databunch(bs=BATCH_SIZE))

data_lm.save('tmp_lm')
data_lm.show_batch()

learn = language_model_learner(data_lm, AWD_LSTM, drop_mult=0.5)

learn.fit_one_cycle(1, 1e-2)
learn.unfreeze()

learn.fit_one_cycle(1, 1e-3)
learn.predict("what is data", n_words=10)



!pip install pytorch-transformers
