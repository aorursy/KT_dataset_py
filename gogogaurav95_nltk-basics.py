import nltk

from nltk.tokenize import word_tokenize, sent_tokenize 
sent = "Mary had a little lamb. Her fleece was white as snow."

sents = sent_tokenize(sent)

print(sents)
words = [word_tokenize(sent) for sent in sents]

print(words)
from nltk.corpus import stopwords

from string import punctuation
customStopWords = set(stopwords.words('english') + list(punctuation))

print(customStopWords)
wordsWOStopWords = [word for word in word_tokenize(sent) if word not in customStopWords]

print(wordsWOStopWords)
from nltk.collocations import *

finder = BigramCollocationFinder.from_words(wordsWOStopWords)
sorted(finder.ngram_fd.items())
text = "Mary closed on closing night when she was in the mood to close."
from nltk.stem.lancaster import LancasterStemmer

stemmer = LancasterStemmer()

stemmedWords = [stemmer.stem(word) for word in word_tokenize(text)]

print(stemmedWords)
nltk.pos_tag(word_tokenize(text))

from nltk.corpus import wordnet as wn

for ss in wn.synsets('bass'):

    print(ss, ss.definition())
from nltk.wsd import lesk

sense = lesk(word_tokenize("She stays closed the  store"), 'close')

print(sense, sense.definition())