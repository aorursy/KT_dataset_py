# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import gensim

import nltk.sentiment

import re

import tqdm



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv('/kaggle/input/blog-authorship-corpus/blogtext.csv')

data.head()
sentence_splitter=re.compile("""[.?!]['"]*\s+""",re.UNICODE)

SplitWords=re.compile("""\W+""",re.UNICODE)

def tokenize(document):

    """For a document, returns a list of tokens.

       For a corpus (iterable of documents), returns a generator of tokenized documents."""

    result = []

    for sentence in sentence_splitter.split(document.lower()):

        words=SplitWords.split(sentence)

        result.extend(words)

    return result



class Tokenizer(object):

    def __init__(self,corpus):

        self.corpus = corpus

    

    def __iter__(self):

        for document in tqdm.tqdm(self.corpus):

            yield tokenize(document)



class bow(object):

    def __init__(self,corpus):

        self.tokens = Tokenizer(corpus)

        self.dictionary = gensim.corpora.dictionary.Dictionary(self.tokens)

    

    def __iter__(self):

        for doc in self.tokens:

            yield self.dictionary.doc2bow(doc)



print("Building dictionary")

frequencies = bow(data['text'].values)

print("Training LogEntropy model")

weights = gensim.models.logentropy_model.LogEntropyModel(frequencies)



print("Training LSI model")

lsi = gensim.models.LsiModel(weights[frequencies],256,frequencies.dictionary)



def transform(corpus):

    return gensim.matutils.corpus2dense(lsi[weights[corpus]],

                                       256).T



vader=nltk.sentiment.vader.SentimentIntensityAnalyzer()
print("Transforming corpus")

vectors = pd.DataFrame(transform(frequencies),

                      index = pd.MultiIndex.from_tuples(data['id'].items()))

vectors.index.names=('uri','id')

print("Calculating sentiments")

sentiments = pd.Series([sum((emotion['compound'] 

                             for emotion in (vader.polarity_scores(sentence)

                                             for sentence in sentence_splitter.split(document))))

                        for document in data['text'].values],

                       index = pd.MultiIndex.from_tuples(data['id'].items()))

sentiments.index.names=('uri','id')

print("Calculating opinions")

opinions = vectors.mul(sentiments,

                      axis='index').groupby(level='id').sum()
norms = opinions.apply(np.linalg.norm,

                      axis=1,

                      result_type='reduce')

def similarity(opinions,norms):

    result = []

    for user in tqdm.tqdm(opinions.index):

        others = opinions.loc[opinions.index>user]

        dotprod = others.mul(opinions.loc[user],

                            axis='columns').sum(axis=1)

        cosine = dotprod/(norms.loc[norms.index>user]*norms[user])

        cosine.index=[(user,other) for other in cosine.index]

        result.append(cosine[cosine<0])

    return pd.concat(result)



similar_users = similarity(opinions,norms)

similar_users.nsmallest(10)
def show_posts(user):

    for (uid,row) in data.loc[data['id']==user].iterrows():

        print(uid,row['text'])

        

show_posts(3963763)
show_posts(3294597)
def common_ground(user0,user1):

    mask = (opinions.loc[user0]*opinions.loc[user1]).apply(lambda x:1.0 if x>0 else 0)

    targets = vectors.xs(user1,level='id')**2.0

    return (targets.dot(mask)/targets.sum(axis=1)).sort_values()

common_ground(3963763,3294597)
data.loc[576267,'text']
common_ground(3294597,3963763)
data.loc[206750,'text']