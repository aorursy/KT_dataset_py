import os



# NPM

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



# Text mining modules

import spacy

from sklearn.feature_extraction.text import TfidfVectorizer

from spacy.lemmatizer import Lemmatizer

#LDA

from gensim.models.ldamodel import LdaModel

from gensim.matutils import Sparse2Corpus

from gensim.corpora.dictionary import Dictionary

import pyLDAvis

import pyLDAvis.gensim

pyLDAvis.enable_notebook()

# pd.set_option('display.max_colwidth', -1)
print(os.listdir("../input"))
nRows2Test = 1500 # specify 'None' if want to read whole file

n_components=7
# Ancillary methods

def explore_topic(lda_model, topic_number, topn):

    """

    accept a ldamodel, atopic number and topn vocabs of interest

    prints a formatted list of the topn terms

    """

    terms = []

    for term, frequency in lda_model.show_topic(topic_number, topn=topn):

        terms += [term]

        print(u'{:20} {:.3f}'.format(term, round(frequency, 3)))

    

    return terms



def print_lda_model(lda_model, num_topics):

    print(u'{:20} {}'.format(u'term', u'frequency') + u'\n')

    for i in range(num_topics):

        print('\n')

        print('Topic '+str(i)+' |---------------------\n')

        tmp = explore_topic(lda_model,topic_number=i, topn=10)

        print
df = pd.read_csv('../input/Consumer_Complaints.csv', delimiter=',', nrows = nRows2Test)

df.columns
df.head()
df.rename(index=str, columns={'Consumer complaint narrative' : 'complaints'}, inplace=True)

df = df[['complaints']]

df['complaints'].isna().sum()
df.dropna(subset=['complaints'], inplace=True)
df['complaints'].isna().sum()
len(df[df['complaints'] == ''])
%%time



# Preprocess:

# lowercase

# df.complaints = df.complaints.str.lower()

# lemmatization

nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner', 'textcat'])

%time df.complaints = df.complaints.apply(lambda row: ' '.join([w.lemma_ for w in nlp(row)]))



# numbers

df.complaints = df.complaints.str.replace(r'\d+', ' number ')

# urls

df.complaints = df.complaints.str.replace(r'(http|https)://[^\s]*', ' httpaddr ')

# email adresses

df.complaints = df.complaints.str.replace(r'[^\s]+@[^\s]+', ' emailaddr ')
df.head()
%%time

tfidf_vectoriser = TfidfVectorizer(max_df=0.95, min_df=0.05, stop_words='english')

complaints_tfidf_vectors = tfidf_vectoriser.fit_transform(df.complaints)
# LDA clustering

%time corpus = Sparse2Corpus(complaints_tfidf_vectors)

%time id2word = {v:k for k,v in tfidf_vectoriser.vocabulary_.items()}

%time dictionary = Dictionary.from_corpus(corpus, id2word=id2word)
%%time

lda = LdaModel(corpus, id2word=id2word, num_topics=n_components)

print_lda_model(lda, min(n_components, 10))
from gensim.models.coherencemodel import CoherenceModel

cm = CoherenceModel(model=lda, corpus=corpus, coherence='u_mass')

cm.get_coherence()
cm.get_coherence_per_topic()
pyLDAvis.gensim.prepare(lda, corpus, dictionary)
def compute_coherence_values(dictionary, corpus, limit, start=2, step=3):

    """

    Compute c_v coherence for various number of topics



    Parameters:

    ----------

    dictionary : Gensim dictionary

    corpus : Gensim corpus

    texts : List of input texts

    limit : Max num of topics



    Returns:

    -------

    model_list : List of LDA topic models

    coherence_values : Coherence values corresponding to the LDA model with respective number of topics

    """

    coherence_values = []

    model_list = []

    for num_topics in range(start, limit, step):

        model=LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics)

        model_list.append(model)

        coherencemodel = CoherenceModel(model=model, corpus=corpus, coherence='u_mass')

        coherence_values.append(coherencemodel.get_coherence())



    return model_list, coherence_values
start=2

limit=50

step=4



model_list, coherence_values = compute_coherence_values(dictionary=dictionary, corpus=corpus, start=start, limit=limit, step=step)

# Show graph

import matplotlib.pyplot as plt

x = range(start, limit, step)

plt.plot(x, coherence_values)

plt.xlabel("Num Topics")

plt.ylabel("Coherence score")

plt.legend(("coherence_values"), loc='best')

plt.show()