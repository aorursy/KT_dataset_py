# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('/kaggle/input/songlyrics/songdata.csv')

queen_df = df[df.artist == 'Queen']

queen_df.head()
import nltk

def count_words(matrix):

  words_g = []

  tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')

  stop_words = set(nltk.corpus.stopwords.words('english'))

  stemmer = nltk.stem.SnowballStemmer('english')

  for lyric in matrix:

    for word in tokenizer.tokenize(lyric):

      if not word in stop_words:

        words_g.append(word)

  f = nltk.FreqDist(words_g)

  words = []

  values = []

  for word, value in f.most_common(20):

    words.append(word)

    values.append(value)

  return words_g, words, values

words_g, words, values = count_words(queen_df.text.values)

import plotly.graph_objs as go

plot = go.Figure(go.Bar(x=words, y=values, marker_color=['orange']*4+['#cc6600']*4+['#b35900']*4+['#994d00']*4+['#804000']*4))

plot.update_layout(title='Palabras mas frecuentes')

plot.show()
from textblob import TextBlob

def sentence_sentiment(lyrics):

    sentiment = []

    means = []

    for lyric in lyrics:

        blob = TextBlob(lyric)

        sentiment.append(blob.sentiment.polarity)

    return sentiment
means = sentence_sentiment(queen_df.text.values.tolist())
sentiments = []

for i in means:

    if i > 0:

        sentiments.append('Positivo')

    elif i == 0:

        sentiments.append('Neutral')

    elif i < 0:

        sentiments.append('Negativo')
queen_df['sentiment'] = sentiments

labels = queen_df.sentiment.value_counts().index.tolist()

values = queen_df.sentiment.value_counts().values.tolist()
plot = go.Figure(go.Pie(labels=labels, values=values, marker_colors=['lightgreen', 'darkred', 'darkorange']))

plot.update_layout(title='Distribución de sentimientos')



plot.show()
import spacy



nlp = spacy.load('en_core_web_sm')

def verbs(text_matrix):

    verbs = []

    penalties = ['\'s', '\'m', '\'re', '\'ve']

    for liryc in text_matrix:

        doc = nlp(liryc)

        for token in doc:

            if token.pos_ == 'VERB':

                if not token.text in penalties:

                    if not token.is_stop:

                        verbs.append(token.text)

    return verbs
verbs = verbs(queen_df.text.values.tolist())
f = nltk.FreqDist(verbs)

wo = []

va = []

for w, v in f.most_common(20):

    wo.append(w)

    va.append(v)

    

plot = go.Figure(go.Bar(x=wo, y=va, marker_color=['orange']*4+['#cc6600']*4+['#b35900']*4+['#994d00']*4+['#804000']*4))

plot.update_layout(title='Verbos Mas Frecuentes')

plot.show()
def persons_parser(text_matrix):

    persons = []

    organizations = []

    geo_political = []

    penalties = ['\'s', '\'m', '\'re', '\'ve']

    for liryc in text_matrix:

        doc = nlp(liryc)

        for token in doc.ents:

            if token.label_ == 'PERSON':

                if not token.text in penalties:

                    persons.append(token.text)

            elif token.label_ == 'ORG':

                organizations.append(token.text)

            elif token.label_== 'GPE':

                geo_political.append(token.text)

    return persons, organizations, geo_political
persons, organizations, geo_political = persons_parser(queen_df.text.values.tolist())
f = nltk.FreqDist(persons)

wo = []

va = []

for w, v in f.most_common(10):

    wo.append(w)

    va.append(v)

    

plot = go.Figure(go.Bar(x=wo, y=va, marker_color=['orange']*4+['#cc6600']*4+['#b35900']*4+['#994d00']*4+['#804000']*4))

plot.update_layout(title='Referencias a supuestas personas')



plot.show()
f = nltk.FreqDist(organizations)

wo = []

va = []

for w, v in f.most_common(10):

    wo.append(w)

    va.append(v)

    

plot = go.Figure(go.Bar(x=wo, y=va, marker_color=['orange']*4+['#cc6600']*4+['#b35900']*4+['#994d00']*4+['#804000']*4))

plot.update_layout(title='Palabras reconocidas como organizaciones')

plot.show()
f = nltk.FreqDist(geo_political)

wo = []

va = []

for w, v in f.most_common(10):

    wo.append(w)

    va.append(v)

    

plot = go.Figure(go.Bar(x=wo, y=va, marker_color=['orange']*4+['#cc6600']*4+['#b35900']*4+['#994d00']*4+['#804000']*4))

plot.update_layout(title='Palabras reconocidas como geopoliticas')

plot.show()
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from sklearn.decomposition import NMF, LatentDirichletAllocation



tfv = TfidfVectorizer(stop_words='english')

cv = CountVectorizer(stop_words='english')
words = tfv.fit_transform(queen_df.text.values)

lda = NMF(n_components=5).fit(words)
feature_names = tfv.get_feature_names()
for i_topico, topico in enumerate(lda.components_):

    print(f'Topico Numero #{i_topico+1}')

    print(" ".join([feature_names[i] for i in topico.argsort()[-20:-1]]))

    

topic_dict = {0:'Supuestamente Dolor', 1:'Supuestamente El Mismo', 2:'Supuestamente Disfrutar La Vida', 3:'Supuestamente Experiencias Adrenalinicas', 4:'Terceros'}
import numpy as np

def classify_topic(document):

    return topic_dict[np.argmax(lda.transform(tfv.transform([document])))]
queen_df['topic'] = queen_df.text.apply(classify_topic)


labels = queen_df.topic.value_counts().index.tolist()

values = queen_df.topic.value_counts().values.tolist()


plot = go.Figure(go.Bar(x=labels, y=values, marker_color=['orange']*4+['#cc6600']*4+['#b35900']*4+['#994d00']*4+['#804000']*4))

plot.update_layout(title='Frecuencia De Los Topicos')

plot.show()