from nltk.tokenize import RegexpTokenizer

from stop_words import get_stop_words

from nltk.stem.wordnet import WordNetLemmatizer

from gensim import corpora, models

import pandas as pd

import gensim

import pyLDAvis.gensim

pattern = r'\b[^\d\W]+\b'

tokenizer = RegexpTokenizer(pattern)

en_stop = get_stop_words('en')

lemmatizer = WordNetLemmatizer()
from nltk.corpus import stopwords 

remove_words = list(stopwords.words('english'))
# remove_words
!ls ../input/news-category-dataset/
# Input from csv

df = pd.read_json('../input/news-category-dataset/News_Category_Dataset_v2.json',lines = True)



# sample data

df.head()
df.shape
df = df.sample(frac = 0.1)
df["Description"] = df["headline"]+". " +df["short_description"]
# list for tokenized documents in loop

texts = []



# loop through document list

for i in df['Description'].iteritems():

    # clean and tokenize document string

    raw = str(i[1]).lower()

    tokens = tokenizer.tokenize(raw)



    # remove stop words from tokens

    stopped_tokens = [raw for raw in tokens if not raw in en_stop]

    

    # remove stop words from tokens

    stopped_tokens_new = [raw for raw in stopped_tokens if not raw in remove_words]

    

    # lemmatize tokens

    lemma_tokens = [lemmatizer.lemmatize(tokens) for tokens in stopped_tokens_new]

    

    # remove word containing only single char

    new_lemma_tokens = [raw for raw in lemma_tokens if not len(raw) == 1]

    

    # add tokens to list

    texts.append(new_lemma_tokens)



# sample data

print(texts[0])
# turn our tokenized documents into a id <-> term dictionary

dictionary = corpora.Dictionary(texts)

# convert tokenized documents into a document-term matrix

corpus = [dictionary.doc2bow(text) for text in texts]
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=15, id2word = dictionary, passes=20)

import pprint

pprint.pprint(ldamodel.top_topics(corpus,topn=5))
pyLDAvis.enable_notebook()

pyLDAvis.gensim.prepare(ldamodel, corpus, dictionary)