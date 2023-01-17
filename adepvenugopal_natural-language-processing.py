#http://www.nltk.org/book/

#Import nltk packages

import nltk

from nltk.tokenize import sent_tokenize, word_tokenize

from nltk.corpus import stopwords

from nltk.stem import PorterStemmer

from nltk.tokenize import PunktSentenceTokenizer

import matplotlib.pyplot as plt
#Define the text string

data = "All work and no play makes jack dull boy. This video is about NLTK programming."

data
#Convert the given text into sentences

sentences = sent_tokenize(data)

print(sentences)
#Convert the given text into words

words = word_tokenize(data)

print(words)
stopWords = set(stopwords.words('english'))

stopWords
wordsFiltered = []

for w in words:

    if w not in stopWords:

        wordsFiltered.append(w)



print("\nExtracted words ==>",words)

print("\nWords after removing the stop words ==>",wordsFiltered)

#print("\nList of stop words ==>",stopWords)
ps = PorterStemmer()

sentence = "gaming, the gamers play games"

words = word_tokenize(sentence)

for word in words:

    print(word + ":" + ps.stem(word))
text = 'She sells, sea shells on the sea shore'

sentences = nltk.sent_tokenize(text)

for sent in sentences:

    print(nltk.pos_tag(nltk.word_tokenize(sent)))
#Import books from nltk corpora

from nltk.book import *
#Display the tile of 1st book (Moby Dick by Herman Melville 1851)

text1
#Display the tile of 2nd book (#Display the tile of 2nd book)

text2
#Searching text

text1.concordance("monstrous")
#What other words appear in a similar range of contexts

text1.similar("monstrous")
#What other words appear in a similar range of contexts

text2.similar("monstrous")
#Plot the word location of the given words in the book

text4.dispersion_plot(["citizens", "democracy", "freedom", "duties", "America"])
#Find the count of words in the 3rd book (The Book of Genesis)

len(text3)
#Find the unique words in the given book (The Book of Genesis)

uniq_words = list(set(text3))

type(uniq_words)

sorted(uniq_words[:10])
#Count the number of unique words in the given book

len(set(text3))
#Find the Lexical Richness of the given book

len(set(text3)) / len(text3)
#Find frequency distribution of the words in book1

fdist1 = FreqDist(text1) 

print(fdist1)
#Find most common used words along with its word count

fdist1.most_common(10)
#Find count of word whale

fdist1['whale']
#Plot frequency distribution of most common used words

fdist1.plot(20)

fdist1.plot(20, cumulative=True)
fdist1.plot(20, cumulative=True)
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

print(classifier.classify(gender_features('Jonas')))
from nltk.corpus import wordnet as wn
#Synsets = Synonym sets.  Find all the synonmyms of 'dog'

wn.synsets('cat')
#Synsets = Synonym sets.  Find all the VERB synonmyms of 'dog'

wn.synsets('dog', pos=wn.VERB)
#Select the synset

wn.synset('dog.n.01')
#Find the Definition of the given synset

print(wn.synset('chase.v.01').definition())
#Find the Examples of the given synset

print(wn.synset('chase.v.01').examples())
#Lemmas = Collection of synonym words 

wn.synset('dog.n.01').lemmas()
#Find the upper entity in hierarchy

dog = wn.synset('dog.n.01')

dog.hypernyms()
dog.hypernyms()
#Find the below entity in hierarchy

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