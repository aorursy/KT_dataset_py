import nltk



nltk.download("gutenberg")
hamlet_raw = nltk.corpus.gutenberg.raw('shakespeare-hamlet.txt')

print(hamlet_raw[:1000])
from nltk.tokenize import sent_tokenize



sentences = sent_tokenize(hamlet_raw)



print(sentences[:10])

from nltk.tokenize import word_tokenize



words = word_tokenize(sentences[0])



print(words)
from nltk.corpus import stopwords



stopwords_list = stopwords.words('english')



print(stopwords_list)
non_stopwords = [w for w in words if not w.lower() in stopwords_list]

print(non_stopwords)
import string

punctuation = string.punctuation

print(punctuation)
from nltk import pos_tag



pos_tags = pos_tag(words)#com o texto original



print(pos_tags)
from nltk.stem.snowball import SnowballStemmer



stemmer = SnowballStemmer('english')



sample_sentence = "He has already gone"

sample_words = word_tokenize(sample_sentence)



stems = [stemmer.stem(w) for w in sample_words]



print(stems)
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer

from nltk.corpus import wordnet



lemmatizer = WordNetLemmatizer() #precisa do POS



pos_tags = nltk.pos_tag(sample_words)



lemmas = []

for w in pos_tags:

    if w[1].startswith('J'):

        pos_tag = wordnet.ADJ

    elif w[1].startswith('V'):

        pos_tag = wordnet.VERB

    elif w[1].startswith('N'):

        pos_tag = wordnet.NOUN

    elif w[1].startswith('R'):

        pos_tag = wordnet.ADV

    else:

        pos_tag = wordnet.NOUN

        

    lemmas.append(lemmatizer.lemmatize(w[0], pos_tag))

    

print(lemmas)
from nltk import word_tokenize



frase = 'o cachorro correu atrás do gato'





ngrams = ["%s %s %s" % (nltk.word_tokenize(frase)[i], \

                      nltk.word_tokenize(frase)[i+1], \

                      nltk.word_tokenize(frase)[i+2]) \

          for i in range(len(nltk.word_tokenize(frase))-2)]#para cada palavra dentor das palavras a té a antepenúltima



print(ngrams)

non_punctuation = [w for w in words if not w.lower() in punctuation]



n_grams_3 = ["%s %s %s"%(non_punctuation[i], non_punctuation[i+1], non_punctuation[i+2]) for i in range(0, len(non_punctuation)-2)]



print(n_grams_3)
from sklearn.feature_extraction.text import CountVectorizer



count_vect = CountVectorizer(ngram_range=(3,3))#de trigramas a trigramas



import numpy as np



#arr = np.array(sentences[:10])#as 10 primeiras

arr = np.array(sentences[:10])#todas



print(arr)



n_gram_counts = count_vect.fit_transform(arr)#extrai os n-gramas e conta a frequência



print(n_gram_counts.toarray()[:10])#lista de contagens



print(count_vect.vocabulary_)#vocabulário

#cada linha é uma sentença

#cada coluna é se há trigrama na sentença, se houver 2 "1" na mesma coluna este trigrama está em duas sentenças

#a posição do 1 é o índice do trigrama que está impresso na lista de contagens

#se houver o mesmo trigrama na mesma sentença, na linha, ele soma
arr = np.array(sentences)



n_gram_counts = count_vect.fit_transform(arr)



print(n_gram_counts.toarray()[:20])



print([k for k in count_vect.vocabulary_.keys()][:20])
import nltk

from nltk.tokenize import word_tokenize

from nltk.tokenize import sent_tokenize

from nltk.stem import WordNetLemmatizer

from nltk.corpus import wordnet

from operator import itemgetter

import pandas as pd

import operator 

import itertools

nltk.download('wordnet')

from collections import Counter

from nltk.corpus import stopwords

stopwords_list = stopwords.words('english')

import string
words = [w for w in (sent_tokenize(nltk.corpus.reuters.raw()))]
tokenized = []

for w in words:

    tokenized.extend(word_tokenize(w))

tokenized[:100]
pos_tags = nltk.pos_tag(tokenized)

pos_tags[:100]
from nltk.stem import WordNetLemmatizer

from nltk.corpus import wordnet



lemmatizer = WordNetLemmatizer()



lemmas = []

for w in pos_tags:       

    

    if w[1].startswith('J'):

        pos_tag = wordnet.ADJ

    elif w[1].startswith('V'):

        pos_tag = wordnet.VERB

    elif w[1].startswith('N'):

        pos_tag = wordnet.NOUN

    elif w[1].startswith('R'):

        pos_tag = wordnet.ADV

    else:

        pos_tag = wordnet.NOUN



    lemmas.append(lemmatizer.lemmatize(w[0], pos_tag))

    
non_stopwords = [w for w in lemmas if not w[0].lower() in stopwords_list]

    

non_punctuation = [w for w in non_stopwords if not w[0] in string.punctuation]
words_to_count = (word for word in non_punctuation if word[:1].isupper())

c = Counter(words_to_count)

print(c.most_common(10))