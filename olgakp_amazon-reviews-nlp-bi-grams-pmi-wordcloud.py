import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import nltk

import os

import pandas

#load the dataset

df_review = pandas.read_csv('../input/amazon_alexa.tsv', sep='\t')

#view first five records

df_review.head()
#remove columns

df_review1 = df_review.drop(['date','feedback','rating','variation'], axis = 1)

df_review1 = df_review1.astype(str)

df_review1.head()



#create new text file with reviews

df_review1.to_csv('reviewtext.txt', sep='\t', index=False)



#load new text file with reviews

f = open('reviewtext.txt', encoding="utf8")

text = f.read()

#print(text)
from nltk import FreqDist



reviewtokens = nltk.word_tokenize(text)

reviewwords = [w.lower( ) for w in reviewtokens]

print(len(reviewtokens))



#define st0p words

nltkstopwords = nltk.corpus.stopwords.words('english')

morestopwords = ['could','would','might','must','need','sha','wo','y',"'s","'d","'ll","'t","'m","'re","'ve","n't"]

stopwords = nltkstopwords + morestopwords

stoppedwords = [w for w in reviewwords if not w in stopwords]

#print(len(stoppedorphanswords))



import re

#function that takes a word and returns true if it consists only

#of non-alphabetic characters  (assumes import re)

def alpha_filter(w):

  # pattern to match word of non-alphabetical characters

  pattern = re.compile('^[^a-z]+$')

  if (pattern.match(w)):

    return True

  else:

    return False

# apply the function to stoppedwords

alphawords = [w for w in stoppedwords if not alpha_filter(w)]

#frequency distribution after stopwords filter

ndist = FreqDist(alphawords)

nitems = ndist.most_common(50)

print(nitems)
mystring = ",".join(alphawords)

from wordcloud import WordCloud, STOPWORDS 

import matplotlib.pyplot as plt 

import pandas as pd 

wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(mystring)

plt.figure()

plt.imshow(wordcloud, interpolation="bilinear")

plt.axis("off")

plt.show()
# Bigrams and Bigram frequency distribution

reviewbigrams = list(nltk.bigrams(reviewwords))



#setup for bigrams and bigram measures

from nltk.collocations import *

bigram_measures = nltk.collocations.BigramAssocMeasures()



# create the bigram finder and score the bigrams by frequency

finder = BigramCollocationFinder.from_words(reviewwords)

scored = finder.score_ngrams(bigram_measures.raw_freq)

# apply a filter to remove non-alphabetical tokens 

finder.apply_word_filter(alpha_filter)

scored = finder.score_ngrams(bigram_measures.raw_freq)

#for bscore in scored[:30]:

#    print (bscore)



# apply a filter to remove stop words

finder.apply_word_filter(lambda w: w in stopwords)

scored = finder.score_ngrams(bigram_measures.raw_freq)

for bscore in scored[:50]:

    print (bscore)
### pointwise mutual information

finder3 = BigramCollocationFinder.from_words(reviewwords)

scored = finder3.score_ngrams(bigram_measures.pmi)

#for bscore in scored[:50]:

#    print (bscore)



# to get good results, must first apply frequency filter = 5

finder.apply_freq_filter(5)

scored = finder.score_ngrams(bigram_measures.pmi)

for bscore in scored[:50]:

    print (bscore)