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
import nltk
import pandas as pd
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd;
import numpy as np;
import scipy as sp;
import sklearn;
import sys;
from nltk.corpus import stopwords, wordnet
import nltk;
from nltk.stem import WordNetLemmatizer
from gensim.models import ldamodel
from gensim.models.hdpmodel import HdpModel
from gensim.models import CoherenceModel
from gensim import matutils, models
import gensim.corpora;
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer;
from sklearn.decomposition import NMF;
from sklearn.preprocessing import normalize;
import scipy.sparse
import string
import pickle;
import re;
from nltk import pos_tag, word_tokenize
data = pd.read_csv('../input/unstructured-l0-nlp-hackathon/data.csv')
def data_cleansing(text):
    '''remove puctuation'''
    text = re.sub(r'[.,"?!:/[;\)/(*%&^]',' ',text)
    '''remove stopwords, numbers and small words'''
    stop_words = stopwords.words('english')
    text = ' '.join([x.lower() for x in text.split() if x not in stop_words and x.isalpha() and len(x)>2])
    text = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)","",text).split())
    return(text)

data['clean_text'] = data['text'].apply(data_cleansing)
data['chunks'] = data['clean_text'].apply(lambda x: ' '.join([xx[0] for xx in pos_tag(word_tokenize(x)) \
                                             if xx[1].startswith('NN') or xx[1].startswith('J')]))
data['chunks'] = data['chunks'].apply(data_cleansing)
data_list = [xx.lower().split(' ') for xx in data['chunks']]
id2word = gensim.corpora.Dictionary(data_list)
corpus = [id2word.doc2bow(text) for text in data_list]
# id2word_count = dict((v, k) for k, v in vectorizer.vocabulary_.items())
def get_lda_topics(model, num_topics):
    word_dict = {}
    topics = model.show_topics(num_topics,20)
    word_dict = {'Topic '+str(i):[x.split('*') for x in words.split('+')] \
                 for i,words in model.show_topics(10,20)}
    return pd.DataFrame.from_dict(word_dict)
lda_mod = models.LdaModel(corpus=corpus,
                          num_topics=5,
                          id2word=id2word,
                          random_state=1,
                          passes=50,
                          eta = 1)
get_lda_topics(lda_mod,20)
scored_corpus = lda_mod[corpus]
final_top = []
for i in range(len(scored_corpus)):
    v=dict(scored_corpus[i])
    for top, score in v.items():
        if score == max(v.values()):
            final_top.append(top)
data["topic"] = final_top
data.head()
topic_dict = {}
topic_dict[0] = 'Automobiles'
topic_dict[1] = 'tech_newsv'
topic_dict[2] = 'sports_news'
topic_dict[3] = 'room_rental'
topic_dict[4] = 'glassdoor_reviews'
data['topic'] = data['topic'].apply(lambda x: topic_dict[x])
data[['Id','topic']].to_csv('output.csv')