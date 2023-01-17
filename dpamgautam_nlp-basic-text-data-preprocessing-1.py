import nltk

from nltk.corpus import brown
brown.words()
# number of words in the brown corpus



len(brown.words())
len(brown.sents())
brown.sents()
brown.sents(fileids='ca01')
len(brown.fileids())
print(brown.fileids()[:100])
print(brown.raw('ca02').strip()[:1000])
from nltk.corpus import webtext
webtext.fileids()
# printing line by line



for i,line in enumerate(webtext.raw('singles.txt').split('\n')):

    if i>10:

        break

    print(str(i) + ':\t' + line)
single_10 = webtext.raw('singles.txt').split('\n')[8]

print(single_10)
from nltk import sent_tokenize, word_tokenize



print(sent_tokenize(single_10))
for sent in sent_tokenize(single_10):

    print(word_tokenize(sent))
print(sent_tokenize(single_10))

print(word_tokenize(sent_tokenize(single_10)[0].lower()))



for sent in sent_tokenize(single_10):

    print([ word.lower() for word in word_tokenize(sent)])
print(word_tokenize(single_10))
from nltk.corpus import stopwords

stop_en = stopwords.words('english')

print(stop_en)
single_10_lower = list(map(str.lower, word_tokenize(single_10)))

print(single_10_lower)
stop_en = set(stop_en)

print([word for word in single_10_lower if not word in stop_en])
from string import punctuation

punct = set(punctuation)

print(punct)
stop_punct = stop_en.union(punct)

print(stop_punct)
print([word for word in single_10_lower if not word in stop_punct])
from nltk.stem import PorterStemmer

porter = PorterStemmer()



for word in ['walking', 'walks', 'walked']:

    print(porter.stem(word))
from nltk.stem import WordNetLemmatizer

wnl = WordNetLemmatizer()



for word in ['walking', 'walks', 'walked']:

    print(wnl.lemmatize(word))