#Importing packages

import pandas as pd



import re

import nltk

from nltk.corpus import stopwords

from nltk.stem.wordnet import WordNetLemmatizer



import gensim

from gensim.corpora import Dictionary

from gensim.models import ldamodel

from gensim import corpora



import numpy

%matplotlib inline



import warnings

warnings.filterwarnings(action='ignore', category=UserWarning)  # To ignore all warnings that arise here to enhance clarity
#Reading data

data=pd.read_csv("fake.csv")

data.head()
#Checking to see the labels

data['type'].unique()
#Checking the languages

data['language'].unique()
#Filtering the english language news and taking only text column

EnNews=data[data['language']=='english']

texts=EnNews['text']

test=texts.tolist()
#Checking 2 first rows of data

test[:2]
texts=[]

for text in test:

    text=str(text)

    texts.append(text)
documents = [re.sub("[^a-zA-Z]+", " ", text) for text in texts]

texts = [[word for word in text.lower().split() ] for text in documents]

# stemming words: having --> have; friends --> friend

lmtzr = WordNetLemmatizer()

texts = [[lmtzr.lemmatize(word) for word in text ] for text in texts]

# tokenize

# remove common words 

stoplist = stopwords.words('english')

texts = [[word for word in text if word not in stoplist] for text in texts]

#remove short words

texts = [[ word for word in tokens if len(word) >= 3 ] for tokens in texts]

extra_stopwords = ['will', 'need', 'think', 'well','going', 'can', 'know', 'com', 'get','make','www','http', 'want',

                'like','say','got','said','something','now', 'news','back','want', 

                'many','along','things','day','also','first', 'great', 'take', 'good', 'much', 'would', 'thing',

                'talk', 'talking', 'thank', 'does', 'give']

extra_stoplist = extra_stopwords

texts = [[word for word in text if word not in extra_stoplist] for text in texts]
# this is text processing required for topic modeling with Gensim

dictionary = Dictionary(texts)



## Remove rare and common tokens.

# ignore words that appear in less than 5 documents or more than 80% documents (remove too frequent & infrequent words) - an optional step

dictionary.filter_extremes(no_below=2, no_above=0.4) 

dictionary.save('fakedata.dict')  # store the dictionary, for future reference



# convert words to vetors or integers

corpus = [dictionary.doc2bow(text) for text in texts]

corpora.MmCorpus.serialize('fakedata.mm', corpus)  # store to disk, for later use
numpy.random.seed(1) # setting random seed to get the same results each time. 

model = ldamodel.LdaModel(corpus, id2word=dictionary, num_topics=35, passes=20)

# Result of the model

model.show_topics()
#Finding top 5 topics in terms of coherence

num_topics = 35

top_topics = model.top_topics(corpus, topn=5)

# Average topic coherence is the sum of topic coherences of all topics, divided by the number of topics.

avg_topic_coherence = sum([t[1] for t in top_topics]) / num_topics

print('Average topic coherence: %.4f.' % avg_topic_coherence)



from pprint import pprint

pprint(top_topics)
#LSI model

from gensim.models import lsimodel

LSImodel = lsimodel.LsiModel(corpus, id2word=dictionary, num_topics=35)

LSImodel.show_topics()
# NMF model

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from sklearn.decomposition import NMF

tfidf_vectorizer = TfidfVectorizer(max_df=0.40, min_df=2, stop_words='english')

tf = tfidf_vectorizer.fit_transform(documents)

nmf = NMF(n_components=35, random_state=1, alpha=.1, l1_ratio=.5).fit(tf)

def print_top_words(model, feature_names, n_top_words):

    for topic_idx, topic in enumerate(model.components_):

        message = "Topic #%d: " % topic_idx

        message += " ".join([feature_names[i]

                             for i in topic.argsort()[:-n_top_words - 1:-1]])

        print(message)

    print()

tfidf_feature_names = tfidf_vectorizer.get_feature_names()

print_top_words(nmf, tfidf_feature_names, 5)