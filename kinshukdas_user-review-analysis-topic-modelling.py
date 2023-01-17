import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
data = pd.read_csv('../input/googleplaystore_user_reviews.csv')
data.head()
data.App.unique()
data.isnull().sum()
len(data)
data = data.dropna()
data.isnull().sum()
len(data)
data.columns
data.Sentiment.unique()
print('No. of Positive: ', (data.Sentiment=='Positive').sum())

print('No. of Neutral: ', (data.Sentiment=='Neutral').sum())

print('No. of Negative: ', (data.Sentiment=='Negative').sum())
sns.countplot(x='Sentiment', data=data)

plt.xlabel('Sentiment')

plt.ylabel('Count')
from wordcloud import WordCloud
# Word Cloud for positive reviews



text = data[data['Sentiment']=='Positive']['Translated_Review']

text = str(text)

wordcloud = WordCloud(max_font_size=100, max_words=200, background_color="white").generate(text)



plt.imshow(wordcloud, interpolation='bilinear')

plt.axis("off")
# Word Cloud for the negative reviews



text = data[data['Sentiment']=='Negative']['Translated_Review']

text = str(text)

wordcloud = WordCloud(max_font_size=100, max_words=200, background_color="white").generate(text)



plt.imshow(wordcloud, interpolation='bilinear')

plt.axis("off")
# Word Cloud for the Neutral reviews



text = data[data['Sentiment']=='Neutral']['Translated_Review']

text = str(text)

wordcloud = WordCloud(max_font_size=100, max_words=200, background_color="white").generate(text)



plt.imshow(wordcloud, interpolation='bilinear')

plt.axis("off")
import re

import pyLDAvis.gensim

from nltk.tokenize import RegexpTokenizer

from nltk.corpus import stopwords

from stemming.porter2 import stem

from nltk.stem import WordNetLemmatizer

from gensim import corpora

from gensim.models.ldamodel import LdaModel

from string import punctuation
negative_reviews = data[(data.Sentiment=='Negative') | (data.Sentiment=='Neutral')]['Translated_Review']
negative_reviews.head()
tokenizer = RegexpTokenizer(r'\w+')

en_stop = set(stopwords.words('english'))

en_stop.add('error')

exclude = set(punctuation)

lemma = WordNetLemmatizer()



texts = []



for rev in negative_reviews:

    review = re.sub('[^a-zA-Z]', ' ', rev)

    raw = review.lower()

    tokens = tokenizer.tokenize(raw)

    stop_free = [rev for rev in tokens if not rev in en_stop]

    punc_free = [rev for rev in stop_free if not rev in exclude]

    normalized = [lemma.lemmatize(rev) for rev in punc_free]

    stemmed_tokens = [stem(rev) for rev in normalized]

    texts.append(stemmed_tokens)
dictionary = corpora.Dictionary(texts)



corpus = [dictionary.doc2bow(text) for text in texts]



lda = LdaModel(corpus, num_topics=5, id2word=dictionary, passes=50, iterations=20, minimum_probability=0.5)



vis_data = pyLDAvis.gensim.prepare(lda, corpus, dictionary, mds='tsne')

pyLDAvis.display(vis_data)
positive_reviews = data[data.Sentiment=='Positive']['Translated_Review']
tokenizer = RegexpTokenizer(r'\w+')

en_stop = set(stopwords.words('english'))

en_stop.add('error')

exclude = set(punctuation)

lemma = WordNetLemmatizer()



texts = []



for rev in positive_reviews:

    review = re.sub('[^a-zA-Z]', ' ', rev)

    raw = review.lower()

    tokens = tokenizer.tokenize(raw)

    stop_free = [rev for rev in tokens if not rev in en_stop]

    punc_free = [rev for rev in stop_free if not rev in exclude]

    normalized = [lemma.lemmatize(rev) for rev in punc_free]

    stemmed_tokens = [stem(rev) for rev in normalized]

    texts.append(stemmed_tokens)
pyLDAvis.enable_notebook()
dictionary = corpora.Dictionary(texts)



corpus = [dictionary.doc2bow(text) for text in texts]



lda = LdaModel(corpus, num_topics=5, id2word=dictionary, passes=50, iterations=20, minimum_probability=0.5)



vis_data = pyLDAvis.gensim.prepare(lda, corpus, dictionary, mds='tsne')

pyLDAvis.display(vis_data)