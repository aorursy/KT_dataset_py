import numpy as np 

import pandas as pd 

data1 = pd.read_csv("../input/SW_EpisodeIV.txt",delim_whitespace = True,header = 0,escapechar='\\')

data2 = pd.read_csv("../input/SW_EpisodeV.txt",delim_whitespace = True,header = 0,escapechar='\\')

data3 = pd.read_csv("../input/SW_EpisodeVI.txt",delim_whitespace = True,header = 0,escapechar='\\')

data = pd.concat([data1,data2,data3],axis = 0)

data.head()
import re

import nltk

from nltk.corpus import stopwords #To Remove the StopWords like "the","in" ect

from nltk.stem import PorterStemmer

from nltk.stem import WordNetLemmatizer #lemmatize the word for example "studies" and "studying" will be converted to "study"
print(PorterStemmer().stem("trouble"))

print(PorterStemmer().stem("troubling"))

print(PorterStemmer().stem("troubled"))
def unigram(data):

    text = " ".join(data)

    CleanedText = re.sub(r'[^a-zA-Z]'," ",text)

    CleanedText = " ".join([WordNetLemmatizer().lemmatize(word) for word in nltk.word_tokenize(CleanedText) if word not in stopwords.words("english") and len(word) > 3])

    return CleanedText
CleanedText = unigram(data['dialogue'])
from wordcloud import WordCloud

%matplotlib inline

import matplotlib.pyplot as plt

wordcloud = WordCloud(random_state=21).generate(CleanedText)

plt.figure(figsize = (30,15))

plt.imshow(wordcloud,interpolation = 'bilinear')

plt.axis("off")

plt.show()
def ngrams(data,n):

    text = " ".join(data)

    text1 = text.lower()

    text2 = re.sub(r'[^a-zA-Z]'," ",text1)

    text3 = " ".join([WordNetLemmatizer().lemmatize(word) for word in nltk.word_tokenize(text2) if word not in stopwords.words("english") and len(word) > 2])

    words = nltk.word_tokenize(text3)

    ngram = list(nltk.ngrams(words,n))

    return ngram
ngram = ngrams(data['dialogue'],2)

ngram[1:10]
"_".join(ngram[0])
for i in range(0,len(ngram)):

    ngram[i] = "_".join(ngram[i])
Bigram_Freq = nltk.FreqDist(ngram)
bigram_wordcloud = WordCloud(random_state = 21).generate_from_frequencies(Bigram_Freq)

plt.figure(figsize = (50,25))

plt.imshow(bigram_wordcloud,interpolation = 'bilinear')

plt.axis("off")

plt.show()
ngram = ngrams(data['dialogue'],3)
for i in range(0,len(ngram)):

    ngram[i] = "_".join(ngram[i])
Trigram_Freq = nltk.FreqDist(ngram)
trigram_wordcloud = WordCloud(random_state = 21).generate_from_frequencies(Trigram_Freq)

plt.figure(figsize = (50,25))

plt.imshow(trigram_wordcloud,interpolation = 'bilinear')

plt.axis("off")

plt.show()
lda_data = []

for i in range(0,len(data)):

    lda_data.append(data.iloc[i,]['dialogue'])

    
import string

exclude = set(string.punctuation)

def clean_doc(doc):

    stop_free = " ".join([i for i in doc.lower().split() if i not in stopwords.words("english")])

    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)

    normalized = " ".join([WordNetLemmatizer().lemmatize(word) for word in nltk.word_tokenize(punc_free)])

    return normalized
doc_clean = [clean_doc(doc).split() for doc in lda_data] 

doc_clean[0]
import gensim

from gensim import corpora

#Creating the term dictionary of our courpus, where every unique term is assigned an index

dictionary = corpora.Dictionary(doc_clean)

#Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above

dtm = [dictionary.doc2bow(doc) for doc in doc_clean]
# Creating the object for LDA model using gensim library

Lda = gensim.models.ldamodel.LdaModel

#num_topics = number of topics you want to extract from the corpus

ldamodel = Lda(dtm, num_topics=5, id2word = dictionary, passes=50)
print(ldamodel.print_topics(num_topics=5, num_words=5))