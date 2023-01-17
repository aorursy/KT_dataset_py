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
data = pd.read_csv('../input/consumer_complaints.csv')
data
verbatim_product = data[['consumer_complaint_narrative','product']]
verbatim_product.head(5)
filtered_verbatim = verbatim_product.dropna()
filtered_verbatim.head(2)
len(filtered_verbatim.consumer_complaint_narrative)
filtered_verbatim['product'].value_counts()
filtered_verbatim['product'].value_counts().plot(kind='bar')
complaint = filtered_verbatim.iloc[1]['consumer_complaint_narrative']
pd.options.display.max_colwidth = 1000
print(complaint)
import spacy #for our NLP processing
import nltk #to use the stopwords library
import string # for a list of all punctuation
from nltk.corpus import stopwords # for a list of stopwords
nlp = spacy.load('en_core_web_sm')
text = nlp(complaint)
text
tokens = [tok for tok in text]
tokens.head(5)
tokens = [tok.lemma_ for tok in text]
tokens
tokens = [tok.lemma_.lower().strip() for tok in text]
tokens
tokens = [tok.lemma_.lower().strip() for tok in text if tok.lemma_ != '-PRON-']
tokens
stop_words = stopwords.words('english')
punctuations = string.punctuation
stop_words
tokens = [tok for tok in tokens if tok not in stop_words and tok not in punctuations]
tokens
def cleanup_text(complaint):
    doc = nlp(complaint, disable=['parser', 'ner'])
    tokens = [tok.lemma_.lower().strip() for tok in doc if tok.lemma_ != '-PRON-']
    tokens = [tok for tok in tokens if tok not in stop_words and tok not in punctuations]
    return tokens

limit = 100
doc_sample = filtered_verbatim.consumer_complaint_narrative
print('tokenized and lemmatized document: ')

for idx, complaint in enumerate(doc_sample):
    print(cleanup_text(complaint))
    if idx == limit:
        break
    
doc_sample = doc_sample[0:10000]
#doc_sample = doc_sample[:]
processed_docs = doc_sample.map(cleanup_text)
import gensim
dictionary = gensim.corpora.Dictionary(processed_docs)
dictionary.filter_extremes(no_below=10, no_above=0.5, keep_n=100000)
bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]
bow_doc_4310 = bow_corpus[4310]

for i in range(len(bow_doc_4310)):
    print("Word {} (\"{}\") appears {} time.".format(bow_doc_4310[i][0], 
                                                     dictionary[bow_doc_4310[i][0]], 
                                                     bow_doc_4310[i][1]))
lda_model = gensim.models.LdaMulticore(bow_corpus, num_topics=10, id2word=dictionary, passes=2 )
for idx, topic in lda_model.print_topics(-1):
    print('Topic: {} \nWords: {}'.format(idx, topic))
import pyLDAvis
import pyLDAvis.gensim as gensimvis
vis_data = gensimvis.prepare(lda_model, bow_corpus, dictionary)
pyLDAvis.display(vis_data)

