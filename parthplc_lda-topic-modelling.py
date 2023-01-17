# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
doc1 = "Sugar is bad to consume. My sister likes to have sugar, but not my father."

doc2 = "My father spends a lot of time driving my sister around to dance practice."

doc3 = "Doctors suggest that driving may cause increased stress and blood pressure."

doc4 = "Sometimes I feel pressure to perform well at school, but my father never seems to drive my sister to do better."

doc5 = "Health experts say that Sugar is not good for your lifestyle."



# compile documents

doc_complete = [doc1, doc2, doc3, doc4, doc5]
from nltk.corpus import stopwords

from nltk.stem.wordnet import WordNetLemmatizer

import string

stop = set(stopwords.words('english'))

exclude = set(string.punctuation)

lemma = WordNetLemmatizer()
def clean(doc):

    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])

    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)

    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())

    return normalized



doc_clean = [clean(doc).split() for doc in doc_complete] 
doc_clean
# Importing Gensim

import gensim

from gensim import corpora



# Creating the term dictionary of our courpus, where every unique term is assigned an index. 

dictionary = corpora.Dictionary(doc_clean)



# Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above.

doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]
print(dictionary)
print(doc_term_matrix)
# Creating the object for LDA model using gensim library

Lda = gensim.models.ldamodel.LdaModel



# Running and Trainign LDA model on the document term matrix.

ldamodel = Lda(doc_term_matrix, num_topics=3, id2word = dictionary, passes=50)
print(ldamodel.print_topics(num_topics=3, num_words=2))
