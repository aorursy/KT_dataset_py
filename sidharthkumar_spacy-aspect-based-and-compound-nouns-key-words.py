import os

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import plotly.express as px

from wordcloud import WordCloud, STOPWORDS

from collections import Counter

from nltk.corpus import stopwords

from sklearn.feature_extraction.text import CountVectorizer

import warnings 

warnings.filterwarnings('ignore')



def count_ngrams(dataframe,column,begin_ngram,end_ngram):

    # adapted from https://stackoverflow.com/questions/36572221/how-to-find-ngram-frequency-of-a-column-in-a-pandas-dataframe

    word_vectorizer = CountVectorizer(ngram_range=(begin_ngram,end_ngram), analyzer='word')

    sparse_matrix = word_vectorizer.fit_transform(df['title'].dropna())

    frequencies = sum(sparse_matrix).toarray()[0]

    most_common = pd.DataFrame(frequencies, 

                               index=word_vectorizer.get_feature_names(), 

                               columns=['frequency']).sort_values('frequency',ascending=False)

    most_common['ngram'] = most_common.index

    most_common.reset_index()

    return most_common





def word_bar_graph_function(df,column,title):

    # adapted from https://www.kaggle.com/benhamner/most-common-forum-topic-words

    topic_words = [ z.lower() for y in

                       [ x.split() for x in df[column] if isinstance(x, str)]

                       for z in y]

    word_count_dict = dict(Counter(topic_words))

    popular_words = sorted(word_count_dict, key = word_count_dict.get, reverse = True)

    popular_words_nonstop = [w for w in popular_words if w not in stopwords.words("english")]

    plt.barh(range(50), [word_count_dict[w] for w in reversed(popular_words_nonstop[0:50])])

    plt.yticks([x + 0.5 for x in range(50)], reversed(popular_words_nonstop[0:50]))

    plt.title(title)

    plt.show()

    

df = pd.read_csv('/kaggle/input/CORD-19-research-challenge/2020-03-13/all_sources_metadata_2020-03-13.csv')  

three_gram = count_ngrams(df,'title',3,3)

words_to_exclude = ["my","to","at","for","it","the","with","from","would","there","or","if","it","but","of","in","as","and",'NaN','dtype']
plt.figure(figsize=(10,10))

word_bar_graph_function(df,'title','Most common words in the titles of the papers in the CORD-19 dataset')
fig = px.bar(three_gram.sort_values('frequency',ascending=False)[0:10], 

             x="frequency", 

             y="ngram",

             title='Most Common 3-Words in Titles of Papers in CORD-19 Dataset',

             orientation='h')

fig.show()
value_counts = df['journal'].value_counts()

value_counts_df = pd.DataFrame(value_counts)

value_counts_df['journal_name'] = value_counts_df.index

value_counts_df['count'] = value_counts_df['journal']

fig = px.bar(value_counts_df[0:20].sort_values('count'), 

             x="count", 

             y="journal_name",

             title='Most Common Journals in the CORD-19 Dataset',

             orientation='h')

fig.show()
value_counts = df['publish_time'].value_counts()

value_counts_df = pd.DataFrame(value_counts)

value_counts_df['which_year'] = value_counts_df.index

value_counts_df['count'] = value_counts_df['publish_time']

fig = px.bar(value_counts_df[0:5].sort_values('count'), 

             x="count", 

             y="which_year",

             title='Most Common Dates of Publication',

             orientation='h')

fig.show()
import numpy as np

import pandas as pd



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import re

import random

random.seed(2019)

import nltk

from nltk.corpus import stopwords

from wordcloud import WordCloud,STOPWORDS

import matplotlib

import matplotlib.pyplot as plt

%matplotlib inline



import matplotlib.pyplot as plt

from matplotlib.ticker import PercentFormatter

import seaborn as sns

import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls

from nltk.tokenize import RegexpTokenizer

from nltk.stem.porter import PorterStemmer

from gensim import corpora, models

import gensim



from bs4 import BeautifulSoup

import re

from nltk.tokenize import WordPunctTokenizer



import datetime



from collections import Counter

def freqx(l, a=10):    

    counter=Counter(l)

    #print(counter)

    #print(counter.values())

    #print(counter.keys())

    return counter.most_common(a)



import spacy

from spacy import displacy

import en_core_web_sm

from tqdm import tqdm

nlp = spacy.load('en_core_web_sm', parse=True, tag=True, entity=True)
df['text'] = df['title'].astype(str) + ". "+ df['abstract'].astype(str)
def asb(data):

    data = data.reset_index(drop = True)

    aspect_terms = []

    comp_terms = []

    for x in tqdm(range(len(data['text']))):

        amod_pairs = []

        advmod_pairs = []

        compound_pairs = []

        xcomp_pairs = []

        neg_pairs = []

        if len(str(data['text'][x])) != 0:

            lines = str(data['text'][x]).replace('*',' ').replace('-',' ').replace('so ',' ').replace('be ',' ').replace('are ',' ').replace('just ',' ').replace('get ','').replace('were ',' ').replace('When ','').replace('when ','').replace('again ',' ').replace('where ','').replace('how ',' ').replace('has ',' ').replace('Here ',' ').replace('here ',' ').replace('now ',' ').replace('see ',' ').replace('why ',' ').split('.')       

            for line in lines:

                doc = nlp(line)

                str1=''

                str2=''

                for token in doc:

                    if token.pos_ is 'NOUN':

                        for j in token.lefts:

                            if j.dep_ == 'compound':

                                compound_pairs.append((j.text+' '+token.text,token.text))

                            if j.dep_ is 'amod' and j.pos_ is 'ADJ': #primary condition

                                str1 = j.text+' '+token.text

                                amod_pairs.append(j.text+' '+token.text)

                                for k in j.lefts:

                                    if k.dep_ is 'advmod': #secondary condition to get adjective of adjectives

                                        str2 = k.text+' '+j.text+' '+token.text

                                        amod_pairs.append(k.text+' '+j.text+' '+token.text)

                                mtch = re.search(re.escape(str1),re.escape(str2))

                                if mtch is not None:

                                    amod_pairs.remove(str1)

                    if token.pos_ is 'VERB':

                        for j in token.lefts:

                            if j.dep_ is 'advmod' and j.pos_ is 'ADV':

                                advmod_pairs.append(j.text+' '+token.text)

                            if j.dep_ is 'neg' and j.pos_ is 'ADV':

                                neg_pairs.append(j.text+' '+token.text)

                        for j in token.rights:

                            if j.dep_ is 'advmod'and j.pos_ is 'ADV':

                                advmod_pairs.append(token.text+' '+j.text)

                    if token.pos_ is 'ADJ':

                        for j,h in zip(token.rights,token.lefts):

                            if j.dep_ is 'xcomp' and h.dep_ is not 'neg':

                                for k in j.lefts:

                                    if k.dep_ is 'aux':

                                        xcomp_pairs.append(token.text+' '+k.text+' '+j.text)

                            elif j.dep_ is 'xcomp' and h.dep_ is 'neg':

                                if k.dep_ is 'aux':

                                        neg_pairs.append(h.text +' '+token.text+' '+k.text+' '+j.text)



            pairs = list(set(amod_pairs+advmod_pairs+neg_pairs+xcomp_pairs))

            for i in range(len(pairs)):

                if len(compound_pairs)!=0:

                    for comp in compound_pairs:

                        mtch = re.search(re.escape(comp[1]),re.escape(pairs[i]))

                        if mtch is not None:

                            pairs[i] = pairs[i].replace(mtch.group(),comp[0])



        aspect_terms.append(pairs)

        comp_terms.append(compound_pairs)







    data['compound_nouns'] = comp_terms

    data['aspect_keywords'] = aspect_terms

    term1 = []

    for j in range(len(aspect_terms)):

        for i in aspect_terms[j]:

            if len(i)>1:

                term1.append(i)

    term2 = []

    for j in range(len(comp_terms)):

        for i in comp_terms[j]:

            if len(i[0])>1:

                term2.append(i[0])

    

    z1 = freqx(term1, 1000)

    z2 = freqx(term2, 1000)

    

    return(pd.DataFrame(z1), pd.DataFrame(z2))

df_ab, df_cn = asb(df)
df_ab.columns = ['Ab_words', 'frequency']   # aspect words

df_cn.columns = ['Cn_words', 'frequency']   #compound nouns
fig = px.bar(df_ab[0:20].sort_values('frequency'), 

             x="frequency", 

             y="Ab_words",

             title='Most common Aspect based keywords in Titles & Abstract of Papers in CORD-19 Dataset',

             orientation='h')

fig.show()
fig = px.bar(df_cn[0:20].sort_values('frequency'), 

             x="frequency", 

             y="Cn_words",

             title='Most common Compound nouns in Titles & Abstract of Papers in CORD-19 Dataset',

             orientation='h')

fig.show()