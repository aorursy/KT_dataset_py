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

data = pd.read_csv("/kaggle/input/unstructured-l0-nlp-hackathon/data.csv")

data.head()
def get_lda_topics(model, num_topics):

    word_dict = {}

    topics = model.show_topics(num_topics,20)

    word_dict = {'Topic '+str(i):[x.split('*') for x in words.split('+')] \

                 for i,words in lda.show_topics(10,20)}

    return pd.DataFrame.from_dict(word_dict)
from nltk.corpus import stopwords

import re

stopwords_list = stopwords.words('english')



data["clean_text"] = data.apply(lambda row: " ".join(list(set(row["text"].split(" "))-set(stopwords_list))) if str(row["text"])!="nan" else "",axis=1) 

data["clean_text"] = data["clean_text"].apply(lambda x: ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)","",x).split()))

data["clean_text"] = data["clean_text"].str.lower()

data["clean_text"].head()
from gensim.models import ldamodel

import gensim.corpora;



data["clean_text_list"] = data["clean_text"].apply(lambda x: x.lower().split(" "))

data_reviews = [value[0] for value in data[["clean_text_list"]].iloc[0:].values]



id2word = gensim.corpora.Dictionary(data_reviews)

corpus = [id2word.doc2bow(text) for text in data_reviews]

# corpus

lda = ldamodel.LdaModel(corpus=corpus, id2word=id2word, num_topics=5)

get_lda_topics(lda, 5)
def get_nmf_topics(model, n_top_words):

    

    #the word ids obtained need to be reverse-mapped to the words so we can print the topic names.

    feat_names = vectorizer.get_feature_names()

    

    word_dict = {};

    for i in range(num_topics):

        

        #for each topic, obtain the largest values, and add the words they map to into the dictionary.

        words_ids = model.components_[i].argsort()[:-20 - 1:-1] 

        #get the top 20 word ids for each topic 

        words = [feat_names[key] for key in words_ids] #obtain the word using the id

        word_dict['Topic ' + '{:01d}'.format(i+1)] = words;

    

    return pd.DataFrame(word_dict);
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer;

from sklearn.decomposition import NMF;

from sklearn.preprocessing import normalize;



num_topics = 5

data_reviews_sent = data["clean_text"].to_list()

vectorizer = TfidfVectorizer(analyzer='word', max_features=5000)

x_counts = vectorizer.fit_transform(data_reviews_sent)

# print(x_counts)

transformer = TfidfTransformer(smooth_idf=False)

x_tfidf = transformer.fit_transform(x_counts)



xtfidf_norm = normalize(x_tfidf, norm='l1', axis=1)



model = NMF(n_components=5, init='nndsvd', max_iter=500, solver="cd", alpha=0.0, tol=1e-4, random_state=42);

model.fit(x_counts)



get_nmf_topics(model, 5)
df = pd.DataFrame(model.transform(vectorizer.transform(data_reviews_sent)))

df["topic"] = df.apply(lambda x: list(x).index(x.max()), axis=1)

df["topic"]
lda[corpus][0]
scored_corpus = lda[corpus]

final_top = []

for i in range(len(scored_corpus)):

    v=dict(scored_corpus[i])

    for top, score in v.items():

        if score == max(v.values()):

            final_top.append(top)
# data["topic"] = final_top

# data.head()
# NMF

data["topic"] = df["topic"]

data.head()
data.replace({'topic': {4: "sports_news",

                                1: "glassdoor_reviews",

                                0: "Automobiles",

                                2: "room_rentals",

                                3: "tech_news"}},inplace=True)

data.head()
data[["Id","topic"]].to_csv("sample_submission.csv",index=False)
from gensim.models.hdpmodel import HdpModel

Hdpmodel = HdpModel(corpus=corpus,id2word=id2word)

def get_hdp_topics(model, num_topics):

    word_dict = {}

    topics = model.show_topics(num_topics,50)

    word_dict = {'Topic '+str(i):[x.split('*') for x in words.split('+')] \

                 for i,words in lda.show_topics(5,50)}

    return pd.DataFrame.from_dict(word_dict)

get_hdp_topics(Hdpmodel, 5)