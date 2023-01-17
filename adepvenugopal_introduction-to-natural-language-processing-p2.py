#http://www.nltk.org/book/

#Import nltk packages

import nltk

from nltk.tokenize import sent_tokenize, word_tokenize

from nltk.corpus import stopwords

from nltk.stem import PorterStemmer

from nltk.tokenize import PunktSentenceTokenizer

import matplotlib.pyplot as plt
#The Brown Corpus was the first million-word electronic corpus of English, created in 1961 at Brown University. 

from nltk.corpus import brown

brown.categories()
#Find words in the category "news" in Brown Corpus

brown.sents(categories='news')
#Find words in the category "news" in Brown Corpus

brown.words(categories='news')
from nltk.corpus import brown

news_text = brown.words(categories='news')

fdist = nltk.FreqDist(w.lower() for w in news_text)

fdist
#The Brown Corpus is a convenient resource for studying systematic differences between genres, a kind of linguistic inquiry known as stylistics. 

#Let's compare genres in their usage of modal verbs. The first step is to produce the counts for a particular genre.

modals = ['can', 'could', 'may', 'might', 'must', 'will']

for m in modals:

    print(m + ':', fdist[m], end=' ')
#Inaugural Address Corpus

#the corpus is actually a collection of 55 texts, one for each presidential address. An interesting property of this collection is its time dimension:

from nltk.corpus import inaugural

inaugural.fileids()[50:]
[fileid[:4] for fileid in inaugural.fileids()][:5]
#Let's look at how the words America and citizen are used over time. 

plt.figure(figsize=(20,10))

cfd = nltk.ConditionalFreqDist((target, fileid[:4])

                               for fileid in inaugural.fileids()

                               for w in inaugural.words(fileid)

                               for target in ['america', 'citizen']

                               if w.lower().startswith(target))

cfd.plot()
#Corpora in Other Languages (hindi)

nltk.corpus.indian.words('hindi.pos')
#Gender Prediction using Naive Bayes Classifer

import nltk.classify.util

from nltk.classify import NaiveBayesClassifier

from nltk.corpus import names



def gender_features(word): 

    return {'last_letter': word[-1]} 

 

# Load data and training 

names = ([(name, 'male') for name in names.words('male.txt')] + 

  [(name, 'female') for name in names.words('female.txt')])

 

featuresets = [(gender_features(n), g) for (n,g) in names] 

train_set = featuresets

classifier = nltk.NaiveBayesClassifier.train(train_set) 

 

# Predict

print(classifier.classify(gender_features('John')))
from nltk.corpus import wordnet as wn
wn.synsets('dog')
wn.synsets('dog', pos=wn.VERB)
wn.synset('dog.n.01')
print(wn.synset('chase.v.01').definition())
print(wn.synset('chase.v.01').examples())
wn.synset('dog.n.01').lemmas()
dog = wn.synset('dog.n.01')

dog.hypernyms()
dog.hypernyms()
dog.hyponyms()
good = wn.synset('good.a.01')

good.lemmas()
good.lemmas()[0].name()
good.lemmas()[0].antonyms()[0].name()
from nltk.corpus import wordnet

word1 = "weapon"

synArray = wordnet.synsets(word1)

synArray

woi =  synArray[0]

print("Word ==>",word1)

print("Word Synset==>",woi)

print("Word Definition ==>",woi.definition())

print("Word Name ==>",woi.name())

print("Word Parts of Speech ==>",woi.pos())

print("Word Hypernym ==>",woi.hypernyms())

print("Word Hyponym ==>",woi.hyponyms())

print("Word Hyponym ==>",woi.hyponyms()[1])

#print(woi2 = woi.hyponyms()[1])
#For Chatbot please refer to YouTube video

#https://www.youtube.com/watch?v=EMDKOk5FeCA&