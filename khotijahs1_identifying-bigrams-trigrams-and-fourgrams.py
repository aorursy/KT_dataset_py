#download necessacy packages

!pip install gensim
import nltk

from gensim.models import Phrases

import gensim.models.keyedvectors as word2vec
from nltk import ngrams

Sentences="i love machine learning and deep learning"

bigram=list(ngrams(Sentences.lower().split(),2))

trigram=list(ngrams(Sentences.lower().split(),3))

fourgram=list(ngrams(Sentences.lower().split(),4))
print(bigram)
print(trigram)
print(fourgram)
print("Total pairs generated are:",len(bigram+trigram+fourgram))