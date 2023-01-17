!ls  ../input
# load data

filename = '../input/metamorphosis_clean.txt'

file = open(filename, 'rt')

text = file.read()

file.close()

# split into words

from nltk.tokenize import word_tokenize

tokens = word_tokenize(text)

print(tokens[:100])
from nltk import sent_tokenize

sentences = sent_tokenize(text)
# load data

file = open(filename, 'rt')

text = file.read()

file.close()

# split into words

from nltk.tokenize import word_tokenize

tokens = word_tokenize(text)

# remove all tokens that are not alphabetic

words = [word for word in tokens if word.isalpha()]

print(words[:100])
from nltk.corpus import stopwords

stop_words = stopwords.words('english')

print(stop_words)
# load data

file = open(filename, 'rt')

text = file.read()

file.close()

# split into words

from nltk.tokenize import word_tokenize

tokens = word_tokenize(text)

# convert to lower case

tokens = [w.lower() for w in tokens]

# remove punctuation from each word

import string

table = str.maketrans('', '', string.punctuation)

stripped = [w.translate(table) for w in tokens]

# remove remaining tokens that are not alphabetic

words = [word for word in stripped if word.isalpha()]

# filter out stop words

from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

words = [w for w in words if not w in stop_words]

print(words[:100])
# load data

file = open(filename, 'rt')

text = file.read()

file.close()

# split into words

from nltk.tokenize import word_tokenize

tokens = word_tokenize(text)

# stemming of words

from nltk.stem.porter import PorterStemmer

porter = PorterStemmer()

stemmed = [porter.stem(word) for word in tokens]

print(stemmed[:100])