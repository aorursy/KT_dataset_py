import numpy as np

import pandas as pd

import os

import string

import re



import matplotlib.pyplot as plt



%matplotlib inline
import nltk

from nltk.tokenize import word_tokenize

from nltk.corpus import stopwords

from nltk.stem import WordNetLemmatizer

from spacy.lemmatizer import Lemmatizer

from spacy.lookups import Lookups



from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.decomposition import TruncatedSVD



from wordcloud import WordCloud

stop_words = stopwords.words('english')



from gensim.models.ldamodel import LdaModel
DATA_PATH = '/kaggle/input/nlp-topic-modelling'
data = pd.read_csv(os.path.join(DATA_PATH, 'Reviews.csv'))

print("Data has {} rows and {} columns".format(data.shape[0], data.shape[1]))
text = data.sample(5000).reset_index()['Text']
# Looking at sample text

print(text[1])

print(text[132])
print(word_tokenize(text[1]))

print(text[1].split(" "))
translator=str.maketrans('','',string.punctuation) # To remove punctuation

lemmatizer = WordNetLemmatizer()

def clean_text(sent):

    return ' '.join([lemmatizer.lemmatize(word.lower()) for word in word_tokenize(sent.translate(translator))\

     if word not in stop_words and len(word)>2 and word.isalpha()])
cleanText = text.apply(lambda x : clean_text(x))
print(text[23])

print("-------")

print(cleanText[23])
tfidfVectorizer = TfidfVectorizer(max_features=500)

tfidfVectors = tfidfVectorizer.fit_transform(cleanText)
tfidfVectors.shape
print(tfidfVectors[234:244, 198:212].todense())
tfidfVectors.shape
lsa = TruncatedSVD(n_components=5, algorithm='randomized', n_iter=3, random_state=18)

lsa_out = lsa.fit_transform(tfidfVectors)
lsa.explained_variance_ratio_
lsa.singular_values_
lsa_out.shape
for i, topic in enumerate(lsa_out[0]):

    print("Topic ", i, " : ", topic*100)
lsa.components_.shape
vocab = tfidfVectorizer.get_feature_names()

topic_content = []
for v in lsa.components_:

    sorted_vocab = sorted(zip(vocab, v), key=lambda x : x[1], reverse=True)

    topic_content.append({x:y for x, y in sorted_vocab})
print("Top 5 words in topic 1 : ", list(topic_content[0].keys())[:5])

print("Top 5 words in topic 2 : ", list(topic_content[1].keys())[:5])

print("Top 5 words in topic 3 : ", list(topic_content[2].keys())[:5])

print("Top 5 words in topic 4 : ", list(topic_content[3].keys())[:5])

print("Top 5 words in topic 5 : ", list(topic_content[4].keys())[:5])
wc1= WordCloud(background_color="black", max_words=500)

wc1.generate_from_frequencies(topic_content[0])



fig = plt.figure(1, figsize=(15, 15))

plt.imshow(wc1, interpolation="bilinear")

plt.title("Topic 1")

plt.axis("off")

plt.show()
wc2= WordCloud(background_color="black", max_words=500)

wc2.generate_from_frequencies(topic_content[1])



fig = plt.figure(1, figsize=(15, 15))

plt.imshow(wc2, interpolation="bilinear")

plt.title("Topic 2")

plt.axis("off")

plt.show()
wc3= WordCloud(background_color="black", max_words=500)

wc3.generate_from_frequencies(topic_content[2])



fig = plt.figure(1, figsize=(15, 15))

plt.imshow(wc3, interpolation="bilinear")

plt.title("Topic 3")

plt.axis("off")

plt.show()
wc4= WordCloud(background_color="black", max_words=500)

wc4.generate_from_frequencies(topic_content[3])



fig = plt.figure(1, figsize=(15, 15))

plt.imshow(wc4, interpolation="bilinear")

plt.title("Topic 4")

plt.axis("off")

plt.show()
wc5= WordCloud(background_color="black", max_words=500)

wc5.generate_from_frequencies(topic_content[4])



fig = plt.figure(1, figsize=(15, 15))

plt.imshow(wc5, interpolation="bilinear")

plt.title("Topic 5")

plt.axis("off")

plt.show()
from gensim import corpora, models

import gensim
splitText = cleanText.apply(lambda x:word_tokenize(x))
dictionary = corpora.Dictionary(splitText)

corpus = [dictionary.doc2bow(text) for text in splitText]
lda = LdaModel(corpus, num_topics=5)
import pyLDAvis.gensim

print(lda.print_topics(num_topics=5, num_words=3))
lda.print_topics()[0]
import pyLDAvis.gensim

pyLDAvis.enable_notebook()

news = pyLDAvis.gensim.prepare(lda,corpus, dictionary)
news
# https://github.com/cemoody/lda2vec

# https://github.com/huseinzol05/NLP-Models-Tensorflow/tree/master/topic-model