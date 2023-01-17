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
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')



train_char = pd.read_csv('../input/train.csv')

test_char = pd.read_csv('../input/test.csv')
import nltk

#nltk.download('popular')



import re

from nltk.corpus import stopwords

from nltk.stem import WordNetLemmatizer

from nltk.tokenize import word_tokenize

from nltk.tokenize import TweetTokenizer



# text cleaning & tokenization

def tokenize(text):

    

    # clean text

    text = text.encode('ascii', 'ignore').decode('ascii')

    text = text.lower()

    

    text = re.sub(r'\b(?:(?:https?|ftp)://)?\w[\w-]*(?:\.[\w-]+)+\S*', ' ', text) # remove hyperlink,subs charact in the brackets

    text = re.sub("[\r\n]", ' ', text) # remove new line characters

    #text = re.sub(r'[^\w\s]','',text)

    text = text.strip() ## convert to lowercase split indv words

    

    #tokens = word_tokenize(text)

    # use TweetTokenizer instead of word_tokenize -> to prevent splitting at apostrophies

    tknzr = TweetTokenizer()

    tokens = tknzr.tokenize(text)

    

    # retain tokens with at least two words

    tokens = [token for token in tokens if re.match(r'.*[a-z]{2,}.*', token)]

    

    # remove stopwords - optional

    # removing stopwords lost important information

    #if stop_set != None:

    stop_set = stopwords.words('english')

    tokens = [token for token in tokens if token not in stop_set]

    

    # lemmmatization - optional

    #if lemmatizer != None:

    lemmatizer = WordNetLemmatizer() 

    tokens = [lemmatizer.lemmatize(token) for token in tokens]

    return tokens



train['tokens'] = train['question_text'].map(lambda x: tokenize(x))

test['tokens'] = test['question_text'].map(lambda x: tokenize(x))
train
t1 = train[['qid','tokens','question_text']]

t2 = t1.append(test)

t2.head()
#######################################

#     Co-occurance words (bigram)     #

#######################################

import pandas as pd

import numpy as np

import nltk

from nltk.corpus import stopwords

import gensim

from gensim.models import LdaModel

from gensim import models, corpora, similarities

import re

from nltk.stem.porter import PorterStemmer

import time

from nltk import FreqDist

from scipy.stats import entropy

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style("darkgrid")

import nltk

from nltk import bigrams

from nltk.corpus import stopwords

import re

import networkx as nx

import warnings

import matplotlib.pyplot as plt

import itertools

import collections

import nltk

from nltk import bigrams

t3 = t2[['tokens']]

t4 = t3['tokens'].tolist()

terms_bigram = [list(bigrams(tweet)) for tweet in t4]

# View bigrams for the first tweet

#https://www.earthdatascience.org/courses/earth-analytics-python/using-apis-natural-language-processing-twitter/calculate-tweet-word-bigrams-networks-in-python/

terms_bigram[0]

bigrams = list(itertools.chain(*terms_bigram))



# Create counter of words in clean bigrams

bigram_counts = collections.Counter(bigrams)



bigram_counts.most_common(30)



bigram_df = pd.DataFrame(bigram_counts.most_common(30),

                             columns=['bigram', 'count'])



bigram_df



d = bigram_df.set_index('bigram').T.to_dict('records')

# Create network plot 

G = nx.Graph()



# Create connections between nodes

for k, v in d[0].items():

    G.add_edge(k[0], k[1], weight=(v * 10))



#G.add_node("dummy", weight=100)

fig, ax = plt.subplots(figsize=(25, 20))



pos = nx.spring_layout(G, k=16)



# Plot networks

nx.draw_networkx(G, pos,

                 font_size=16,

                 width=3,

                 edge_color='grey',

                 node_color='purple',

                 with_labels = False,

                 ax=ax)



# Create offset labels

for key, value in pos.items():

    x, y = value[0]+.135, value[1]+.045

    ax.text(x, y,

            s=key,

            bbox=dict(facecolor='red', alpha=0.25),

            horizontalalignment='center', fontsize=30)

    

plt.show()

##This plot displays the networks of co-occurring words in tweets

## plot bigram df

top_30 = bigram_df.head(20).sort_values(by=['count'])

top_30

ax = top_30.plot.barh(x='bigram', y='count', rot=0, color=(0.2, 0.4, 0.6, 0.6))
t5 = t2[['tokens']]

t5 = t5['tokens'].tolist()

t6  = list(itertools.chain(*t5))



#######################################

#        Most common Word             #

#######################################



fdist = FreqDist(t6)

top20_tweetcount = fdist.most_common(20)

top20_tweetcount = sorted(top20_tweetcount , key=lambda x: x[1])

list1 = []

list2 = []

for i in top20_tweetcount:

   list1.append(i[0])

   list2.append(i[1])



print(top20_tweetcount)

list

ef = pd.DataFrame({'Vocubulary':list1, 'Most common word':list2})

bx = ef.plot.barh(x='Vocubulary', y='Most common word', rot=0, color=(0.2, 0.4, 0.6, 0.6))
def tokenize_character(text):

    

    # clean text

    text = text.encode('ascii', 'ignore').decode('ascii')

    text = text.lower()

    

    text = re.sub(r'\b(?:(?:https?|ftp)://)?\w[\w-]*(?:\.[\w-]+)+\S*', ' ', text) # remove hyperlink,subs charact in the brackets

    text = re.sub("[\r\n]", ' ', text) # remove new line characters

    #text = re.sub(r'[^\w\s]','',text)

    text = text.strip() ## convert to lowercase split indv words

    

    #tokens = word_tokenize(text)

    # use TweetTokenizer instead of word_tokenize -> to prevent splitting at apostrophies

    tknzr = TweetTokenizer()

    tokens = tknzr.tokenize(text)

    

    # retain tokens with at least two words

    tokens = [token for token in tokens if re.match(r'.*[a-z]{2,}.*', token)]

    

    return tokens



train_char['tokens'] = train_char['question_text'].map(lambda x: tokenize_character(x))

test_char['tokens'] = test_char['question_text'].map(lambda x: tokenize_character(x))
train_char

p1 = train_char

p1 = p1[["target", "tokens"]]

p1["word_count"] = p1.tokens.apply(lambda x: len(x))

p1.head()
def score_to_numeric(x):

    if x==0:

        return "Insincere"

    if x==1:

        return "Sincere"

p1['target_class'] = p1['target'].apply(score_to_numeric)

p1.head()
import seaborn as sns

sns.set(style="whitegrid")
ax = sns.boxplot(x="target_class", y="word_count", data=p1,                  

                 order=["Insincere", "Sincere"])
p1_zero = p1[p1["target"]==0]

p1_zero = p1_zero[["tokens", "word_count"]]



p1_one = p1[p1["target"]==1]

p1_one = p1_one[["tokens", "word_count"]]



p1_zero.head()
p2_zero = p1_zero[['tokens']]

p2_zero = p2_zero['tokens'].tolist()

p2_zero  = list(itertools.chain(*p2_zero))



stop_set = stopwords.words('english')

p2_zero = [token for token in p2_zero if token not in stop_set]



fdist = FreqDist(p2_zero)

top20_tweetcount = fdist.most_common(20)

top20_tweetcount = sorted(top20_tweetcount , key=lambda x: x[1])

list1 = []

list2 = []

for i in top20_tweetcount:

   list1.append(i[0])

   list2.append(i[1])



print(top20_tweetcount)

list

ef = pd.DataFrame({'Insincere Vocubulary':list1, 'Most common word':list2})

bx = ef.plot.barh(x='Insincere Vocubulary', y='Most common word', rot=0, color=(0.2, 0.4, 0.6, 0.6))
p2_one = p1_one[['tokens']]

p2_one = p2_one['tokens'].tolist()

p2_one  = list(itertools.chain(*p2_one))



stop_set = stopwords.words('english')

p2_one = [token for token in p2_one if token not in stop_set]



fdist = FreqDist(p2_one)

top20_tweetcount = fdist.most_common(20)

top20_tweetcount = sorted(top20_tweetcount , key=lambda x: x[1])

list1 = []

list2 = []

for i in top20_tweetcount:

   list1.append(i[0])

   list2.append(i[1])



print(top20_tweetcount)

list

ef = pd.DataFrame({'Sincere Vocubulary':list1, 'Most common word':list2})

bx = ef.plot.barh(x='Sincere Vocubulary', y='Most common word', rot=0, color=(0.2, 0.4, 0.6, 0.6))