#The jupyter notebook can be downloaded using the link at top right corner of the kaggle webpage if not fully rendered in the webpage. 



import numpy as np

import gensim

import os

import re



from gensim.utils import simple_preprocess

from gensim.parsing.preprocessing import STOPWORDS

from gensim import corpora



from gensim.models.ldamulticore import LdaMulticore



import pandas as pd
!pip install scispacy
!pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.2.4/en_core_sci_md-0.2.4.tar.gz
#df = pd.read_csv('metadata.csv')

bucket = 'coviddata'

file = 'metadata.csv'

gcs_url = 'https://%(bucket)s.storage.googleapis.com/%(file)s' % {'bucket':bucket, 'file':file}

df = pd.read_csv(gcs_url)

df.head()
df2 = df.drop(columns = ['sha', 'source_x', 'pmcid', 'license', 'Microsoft Academic Paper ID', 'WHO #Covidence', 'has_full_text'])
df2.head()
df2.shape
df3 = df2.dropna(subset=['abstract'])
df3.shape
df3.head()
import en_core_sci_md

nlp = en_core_sci_md.load(disable=["tagger", "parser", "ner"])

nlp.max_length = 2000000
import spacy
from spacy.tokenizer import Tokenizer
def tokenize(doc):

    

    return [token.text for token in nlp(doc) if not token.is_stop and not token.is_punct and not token.pos == 'PRON']
data = df3['abstract'].apply(tokenize)
data
vect = [nlp(doc).vector for doc in df3['abstract']]
from sklearn.neighbors import NearestNeighbors
nn = NearestNeighbors(n_neighbors=25, algorithm='ball_tree')

nn.fit(vect)
query = "chloroquine hydroxycholoroquine HCoV-19 SARS-CoV-2 coronavirus covid-19 treatment"
query_vect = nlp(query).vector
#find 10 most similar abstracts as the above query

similar_abstracts = nn.kneighbors([query_vect])[1]
for abstract in similar_abstracts:

    print(df3['abstract'].iloc[abstract])
output = pd.DataFrame((df3['abstract'].iloc[abstract]))

pd.set_option('display.max_colwidth', 0)

output.head(25)

#Output of the top 25 abstracts matching the query with index numbers
#From the above abstracts, abstracts 28684, 8950, 7683 appear relevant to our search for newer treatments.

#Abstract 4935 and 34889 is relevant to chloroquine in treating covid-19. 

#Abstract 18811 is relevant to a monoclonal antibody treatment against covid-19.

#Abstract 30643 is relevant to a new target Abelson tyrosine-protein kinase 2 (Abl2) against covid-19.

#Abstract 43973 is important in discussing various approaches towards developing a vaccine and treatments against covid-19.



#Next step will be to inspect the detailed papers for these abstracts.

#Let us inspect the abstracts first and raw some conclusions.
#pd.set_option('display.max_colwidth', 0)

query1 = output.iloc[ 10, : ]

query1.head()
query2 = output.iloc[ 19, : ]

query2.head()
#The full text for this abstract can be accessed here : https://github.com/bs3537/DS-Unit-4-Sprint-1-NLP/blob/master/1-s2.0-S2090123220300540-main.pdf
#NLP Topic Modeling for the selected abstract above
from sklearn.feature_extraction.text import TfidfVectorizer
vect = TfidfVectorizer(stop_words='english', tokenizer = tokenize, ngram_range=(1,2))
tf = vect.fit_transform(output['abstract'])
from sklearn.decomposition import LatentDirichletAllocation
lda = LatentDirichletAllocation(n_components=50, random_state=0, n_jobs=-1)
lda.fit(tf)
def print_top_words(model, feature_names, n_top_words):

    for topic_idx, topic in enumerate(model.components_):

        message = "\nTopic #%d: " % topic_idx

        message += " ".join([feature_names[i]

                             for i in topic.argsort()[:-n_top_words - 1:-1]])

        print(message)

    print()
tfidf_feature_names = vect.get_feature_names()

top_words = print_top_words(lda, tfidf_feature_names, 25)

top_words
!pip install pyLDAvis
import pyLDAvis.gensim



pyLDAvis.enable_notebook()
data = output['abstract'].apply(tokenize)
id2word = corpora.Dictionary(data)
corpus = [id2word.doc2bow(token) for token in data]
lda2 = LdaMulticore(corpus = corpus,

                   id2word = id2word,

                   random_state = 42,

                   num_topics = 15,

                   passes = 10,

                   workers = 4)
lda2.print_topics()
import re

words = [re.findall(r'"([^"]*)"',t[1]) for t in lda2.print_topics()]
topics = [' '.join(t[0:10]) for t in words]
for id, t in enumerate(topics): 

    print(f"------ Topic {id} ------")

    print(t, end="\n\n")
pyLDAvis.gensim.prepare(lda2, corpus, id2word)
#On hovering over 2019-nCoV, this word is most commonly present in topics 1 and 10. 

#COVID-19 word is most commonly present in topic 1. 