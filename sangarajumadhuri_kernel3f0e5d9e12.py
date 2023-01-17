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
scored_corpus = lda[corpus]

final_top = []

for i in range(len(scored_corpus)):

    v=dict(scored_corpus[i])

    for top, score in v.items():

        if score == max(v.values()):

            final_top.append(top)
data["topic"] = final_top

data.head()
data.replace({'topic': {2: "sports_news",

                                3: "glassdoor_reviews",

                                4: "Automobiles",

                                1: "room_rentals",

                                0: "tech_news"}},inplace=True)

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