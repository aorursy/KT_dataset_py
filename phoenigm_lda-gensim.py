!pip install pymorphy2



import nltk as nltk

from nltk.corpus import stopwords

import pandas as pd

import numpy as np

import pymorphy2

import string



from gensim import corpora

from gensim.models.coherencemodel import CoherenceModel

from gensim.models.ldamodel import LdaModel
analyzer = pymorphy2.MorphAnalyzer()

extra = '«»\–—'

extra_stop = ['...', 'фильм', 'это', 'который', 'весь', 'свой', 'человек']

print(stopwords.words('russian'))



def get_normal_form_of_single_text(text):

    words = nltk.word_tokenize(text)

    

    words_array = []

    

    for word in words:

        token = analyzer.parse(word)[0].normal_form

        if token in stopwords.words('russian') or token in string.punctuation or token in extra or token in extra_stop:

            continue

        words_array.append(token)

    return words_array
# df = pd.read_csv("../input/allreviews/all-reviews.csv", names=["name", "review", "label"], skiprows=lambda i: i % 10 != 0 or i == 0)

df = pd.read_csv("../input/allreviews/all-reviews.csv", names=["name", "review", "label"], skiprows=1)

df['label'] = df['label'].map(lambda x: "1" if x not in ["-1", "0", "1"] else x)



df['review'] = df['review'].map(lambda x: get_normal_form_of_single_text(x))

print(df['review'])
dictionary = corpora.Dictionary(df['review'])

print(dictionary)
corpus = [dictionary.doc2bow(text) for text in df['review']]
words_count = 15

topics_count = 10

passes = 1
model = LdaModel(corpus, num_topics=topics_count, id2word=dictionary, passes=passes)

topics = model.print_topics(num_words=words_count)

for topic in topics:

    words = topic[1]

    print(list(map(lambda x: x[x.index("\"")+1:-2], words.split('+'))))
cm = CoherenceModel(model=model, texts=df['review'], dictionary=dictionary)

coherence = cm.get_coherence()

print(coherence)