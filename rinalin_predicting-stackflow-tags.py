# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import nltk # natural language processing

import re # regular expression

from bs4 import BeautifulSoup #scraping HTML

from nltk.corpus import stopwords

import seaborn as sns # visualization

import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfTransformer

from string import punctuation









# nltk workspace



stop = set(stopwords.words('english'))



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
# read in raw data sets

biology = pd.read_csv("../input/biology.csv")

cooking = pd.read_csv("../input/cooking.csv")

crypto = pd.read_csv("../input/crypto.csv")

diy = pd.read_csv("../input/diy.csv")

robotics = pd.read_csv("../input/robotics.csv")

travel = pd.read_csv("../input/travel.csv")

test = pd.read_csv("../input/test.csv")
## Concatenate datasets

#raw = pd.concat([biology,cooking,crypto,diy,robotics,travel],axis = 0, ignore_index =True)

raw = test
# Simple Statistics

print("This dataset has in total {} rows.".format(raw.shape[0]))

#print("out of which, {} rows come from train dataset and {} rows come from test dataset.".format(raw[raw['source']=='train'].shape[0],raw[raw['source']=='test'].shape[0]))
def parse_content(s):

    emphasize = []

    header = []

    link = []

    content = ""

    soup = BeautifulSoup(s,'html.parser')

    content = soup.get_text()

    return pd.Series({'content_parsed':content})
# Apply parse_content onto dataframe.

raw = pd.concat([raw,raw['content'].apply(parse_content)],axis = 1)
test[1:5]
## Study the tags Word count distribution

#raw['tags'] = raw['tags'].apply(lambda x: x.split(" "))

#raw['tags_wc'] = raw['tags'].apply(len)

#sns.barplot(x='tags_wc',y='tags_wc',data=raw,estimator=lambda x: len(x),palette='Blues')
##study the tags: semantic structure

#raw['tags_token'] = raw['tags'].apply(str).apply(nltk.word_tokenize)

#raw['tags_pos'] = raw['tags'].apply(nltk.pos_tag)
# What is the distribution of different semantics?

#semantics = pd.DataFrame({'semantics' : [pair[1] for col in raw['tags_pos'].tolist() for pair in col]})

#semantics['count'] = 1

#fig,axs = plt.subplots()

#sns.barplot(x='semantics',y='count',data=semantics,estimator=lambda x: len(x),palette=sns.cubehelix_palette(8, start=.5, rot=-.75),ax=axs)

#axs.set_title('semantic distribution')

#axs.set_ylabel('frequency')
from string import punctuation

def strip_punctuation(s):

    return ''.join(c for c in s if c not in punctuation)

#Building NLTK pipelines

def td_idf_matrix(dataset):

    dataset['all_text'] = dataset['title'] + dataset['content_parsed'] 

    dataset['all_text'] = dataset['all_text'].apply(lambda x: str.lower(x).replace('\n',' '))

    mydoclist = [strip_punctuation(doc) for doc in dataset['all_text'].tolist()]

    count_vectorizer = CountVectorizer(stop_words='english',lowercase=True,analyzer='word')

    term_freq_matrix = count_vectorizer.fit_transform(mydoclist)

    tfidf = TfidfTransformer(norm="l2")

    tfidf.fit(term_freq_matrix)

    tf_idf_matrix = tfidf.transform(term_freq_matrix)

    pos_to_word = dict([[v,k] for k,v in count_vectorizer.vocabulary_.items()])

    return tf_idf_matrix, pos_to_word



tf_idf_matrix, pos_to_word = td_idf_matrix(raw)


def importance_list_row(sparse_row,n_importance):

    importance_list = [0]*n_importance

    for i in range(0,n_importance): 

        ind =  sparse_row.indices[sparse_row.data.argmax(axis=0)] if sparse_row.nnz else 0

        importance_list[i] = pos_to_word[ind]

        sparse_row[0,ind] = 0

    return importance_list





def importance_list(sparse_matrix,n_importance):

    n_row = sparse_matrix.shape[0]

    importance_lists = [0]*n_row

    for row in range(0,n_row):

        importance_lists[row] = importance_list_row(sparse_matrix[row],n_importance)

    return importance_lists   
#n_importance = 2

#predict = importance_list(tf_idf_matrix,n_importance)

#predict_vs_actual = pd.DataFrame({'predict':predict})

#predict_vs_actual['predict'] = predict_vs_actual['predict'].apply(lambda x: "".join(chr+" ") for char in x)

#predict_vs_actual[0:50]
#tokenize and tag texts. 

lemmatizer = nltk.stem.WordNetLemmatizer()

raw['all_text'] = raw['all_text'].apply(strip_punctuation)

raw['text_token'] = raw['all_text'].apply(nltk.word_tokenize)

raw['text_token'] = raw['all_text'].apply(nltk.word_tokenize)

raw['text_token'] = raw['text_token'].apply(lambda x:[lemmatizer.lemmatize(t) for t in x])

raw['text_pos'] = raw['text_token'].apply(nltk.pos_tag)

raw['text_nouns'] = raw['text_pos'].apply(lambda x: [pair[0] for pair in x if pair[1] in ("NN","NNS","JJ")])
raw['text_bigram'] = raw['text_pos'].apply(nltk.bigrams)

raw['text_bigram'] = raw['text_bigram'].apply(list)
raw['word_pair'] = raw['text_bigram'].apply(findPair)
raw[0:5]
def findPair(l):

    result = []

    for pair in l:

        if pair[1][1] in ('NN','NNS') and pair[0][1] in ('NN','NNS','JJ'):

            result.append(pair[0][0]+" "+pair[1][0])

    return result
mydoclist = raw['text_nouns'].apply(" ".join).tolist()

#mydoclist[0:5]

count_vectorizer = CountVectorizer(stop_words='english',lowercase=True,analyzer='word',ngram_range=(1,1))

term_freq_matrix = count_vectorizer.fit_transform(mydoclist)

tfidf = TfidfTransformer(norm="l2")

tfidf.fit(term_freq_matrix)

tf_idf_matrix = tfidf.transform(term_freq_matrix)

pos_to_word = dict([[v,k] for k,v in count_vectorizer.vocabulary_.items()])
n_importance = 3

predict = importance_list(tf_idf_matrix,n_importance)

predict_vs_actual = pd.DataFrame({'tags':predict,'id':raw['id']})

predict_vs_actual['tags'] = predict_vs_actual['tags'].apply(" ".join)
predict_vs_actual[0:100]
predict_vs_actual.to_csv("predicted.csv",index=False)