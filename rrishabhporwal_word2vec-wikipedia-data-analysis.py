import nltk

import urllib

import bs4 as bs

import re

from gensim.models import Word2Vec

from nltk.corpus import stopwords
#!pip install beautifulsoup4

#!pip install lxml
# Gettings the data source

source = urllib.request.urlopen('https://en.wikipedia.org/wiki/Global_warming').read()
# Parsing the data/ creating BeautifulSoup object

soup = bs.BeautifulSoup(source,'lxml')

# Fetching the data

text = ""

for paragraph in soup.find_all('p'):

    text += paragraph.text
# Preprocessing the data

text = re.sub(r'\[[0-9]*\]',' ',text)

text = re.sub(r'\s+',' ',text)

text = text.lower()

text = re.sub(r'\d',' ',text)

text = re.sub(r'\s+',' ',text)
# Preparing the dataset

sentences = nltk.sent_tokenize(text)



sentences = [nltk.word_tokenize(sentence) for sentence in sentences]



for i in range(len(sentences)):

    sentences[i] = [word for word in sentences[i] if word not in stopwords.words('english')]
# Training the Word2Vec model

model = Word2Vec(sentences, min_count=1)



words = model.wv.vocab
# Finding Word Vectors

vector = model.wv['global']
# Most similar words

similar = model.wv.most_similar('global')
# Word2Vec model visualization



# Install gensim - pip install gensim

from gensim.models import KeyedVectors



filename = 'GoogleNews-vectors-negative300.bin'



model = KeyedVectors.load_word2vec_format(filename, binary=True)



model.wv.most_similar('king')



model.wv.most_similar(positive=['king','woman'], negative= ['man'])