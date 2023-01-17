import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('darkgrid')

import plotly.express as ex

import plotly.graph_objs as go

import plotly.figure_factory as ff

from wordcloud import WordCloud,STOPWORDS

import nltk as nlp

import string 

import re
p_data = pd.read_csv('/kaggle/input/poe-short-stories-corpuscsv/preprocessed_data.csv')

p_data.head(3)
title_language = []

text_language  = []



title_bow = {}

text_bow = {} 



for index,row in p_data.iterrows():

    title_language += row['title'].lower().split(' ')

    text_language += row['text'].lower().split(' ')



for index,row in p_data.iterrows():

    title =  row['title'].lower().split(' ')

    text  = row['text'].lower().split(' ')

    for te in text:

        text_bow[te] = text_bow.get(te,0) +1

    for ti in title:

        title_bow[ti] = title_bow.get(ti,0) +1

          

    

title_language = list(set(title_language))

text_language = list(set(text_language))



    
all_texts = ' '.join(p_data.text.values)
def generate_ngram(n,text):

    s = text.lower()

    s = re.sub(r'[^a-zA-Z0-9\s]', ' ', s)

    tokens = s.split(' ')

    ngrams = zip(*[tokens[i:] for i in range(n)])

    return [" ".join(ngram) for ngram in ngrams]



bigram = generate_ngram(2,all_texts)

def get_nword_probs(word):

    contains = [w.split(' ')[1] for w in bigram if w.split(' ')[0] == word and w.split(' ')[1] != '']

    cont_dic = {}

    for word in contains:

        cont_dic[word] = cont_dic.get(word,0)+1

    occ = len(contains)

    cont_dic = {word:cont_dic[word]/occ for word in cont_dic.keys()}

    return cont_dic    



def get_next_word(cur_word,alpha):

    prob_dic = get_nword_probs(cur_word)

    prob_dic_top_5 = sorted(prob_dic, key=prob_dic.get, reverse=True)[:5]

    if np.random.normal(0,1,1) > alpha and len(prob_dic_top_5)>4:

        return prob_dic_top_5[int(np.round(np.random.uniform(1,4,1)))]

    elif len(prob_dic_top_5) == 0:

        return list(STOPWORDS)[int(np.round(np.random.uniform(0,len(STOPWORDS)-1,1)))]

    else:

        return prob_dic_top_5[0]



def get_random_words(n_words):

    tsample = p_data.text.sample(int(np.sqrt(n_words)))

    words = []

    for i in tsample:

        words += i.split(' ')

    choice = np.round(np.random.uniform(0,len(words),n_words))

    return [words[int(i)] for i in choice]
words = get_random_words(1)

poem_length = 80

poem = ''

cur_word = words[0]

for i in range(0,poem_length):

    poem+= (' '+(get_next_word(cur_word,0.5)))

    if np.random.normal(0,1,1) >0.8:

        poem+='\n'

    elif np.random.normal(0,1,1) >0.7:

        poem+=','

    elif np.random.normal(0,1,1) >0.9:

        words = get_random_words(5)

        words = [word for word in words if word not in STOPWORDS]

        if len(words) == 0:

            cur_word = get_next_word(cur_word,0.5)

        else:

            cur_word = words[0]

    else:

        cur_word = get_next_word(cur_word,0.5)

print(poem)
words = get_random_words(1)

poem_length = 80

poem = ''

cur_word = words[0]

for i in range(0,poem_length):

    poem+= (' '+(get_next_word(cur_word,0.8)))

    if np.random.normal(0,1,1) >0.8:

        poem+='\n'

    elif np.random.normal(0,1,1) >0.7:

        poem+=','

    elif np.random.normal(0,1,1) >0.9:

        words = get_random_words(5)

        words = [word for word in words if word not in STOPWORDS]

        if len(words) == 0:

            cur_word = get_next_word(cur_word,0.8)

        else:

            cur_word = words[0]

    else:

        cur_word = get_next_word(cur_word,0.8)

print(poem)
words = get_random_words(1)

poem_length = 120

poem = ''

cur_word = words[0]

for i in range(0,poem_length):

    poem+= (' '+(get_next_word(cur_word,0.62)))

    if np.random.normal(0,1,1) >0.8:

        poem+='\n'

    elif np.random.normal(0,1,1) >0.7:

        poem+=','

    elif np.random.normal(0,1,1) >0.9:

        words = get_random_words(5)

        words = [word for word in words if word not in STOPWORDS]

        if len(words) == 0:

            cur_word = get_next_word(cur_word,0.62)

        else:

            cur_word = words[0]

    else:

        cur_word = get_next_word(cur_word,0.62)

print(poem)