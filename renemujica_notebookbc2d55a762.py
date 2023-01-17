import numpy as np

import csv

from bs4 import BeautifulSoup

from nltk.corpus import stopwords

from nltk.tokenize import RegexpTokenizer

import re

import pickle

from nltk.collocations import BigramCollocationFinder

from nltk.metrics import BigramAssocMeasures

import nltk

from nltk.sentiment.vader import SentimentIntensityAnalyzer
tokenizer = RegexpTokenizer(r"\w+")

stop = set(stopwords.words("english"))

def process_review(review):

    review = review.lower()

    tokens = tokenizer.tokenize(review)

    return filter(lambda token: token not in stop or len(token) > 3, tokens)

    
corpus_dict = dict()

corpus = []

review_corpus = []

#with open("./corpusdict.pkl", "wb") as corpusdictfile, open("./corpus.pkl", "wb") as corpusfile:

with open("../input/Reviews.csv") as csvfile:

    reader = csv.reader(csvfile)

    for row in reader:

        soup = BeautifulSoup(row[9], "html.parser")

        review = soup.get_text()

        filtered = process_review(review)

        for word in filtered:

            try:

                corpus_dict[word] += 1

            except KeyError:

                corpus_dict[word] = 0

            finally:

                corpus.append(word)



        review_corpus.append(review)

    

    #pickle.dump(corpus, corpusfile)

    #pickle.dump(corpus_dict, corpusdictfile)
# el largo del corpus procesado

print(len(corpus))
threshold = 3

corpus_prunned = [word for word in corpus if corpus_dict[word] > threshold]

print(len(corpus_prunned))
#with open("./prunned.pkl", "wb") as prunnedfile:

#    pickle.dump(corpus_prunned, prunnedfile)
#corpus_prunned = []

#with open("./prunned.pkl", "rb") as prunnedfile:

#    corpus_prunned = pickle.load(prunnedfile)

print(len(corpus_prunned))

bcf = BigramCollocationFinder.from_words(corpus_prunned)

print("Top 30 collocations:")

print(bcf.nbest(BigramAssocMeasures.likelihood_ratio, 30))



corpus_prunned = corpus_prunned[:int(len(corpus_prunned)*0.01)]

print("PoS tagging")

tagged = nltk.pos_tag(corpus_prunned)

print("NER tagging")

ner = nltk.chunk.ne_chunk(tagged)[:100]

print(ner)

print("Sentiment analysis")

review_analyzed = dict()

sid = SentimentIntensityAnalyzer()

for review in review_corpus:

    review_analyzed[review] = sid.polarity_scores(review)

list(review_analyzed.values())[:30]