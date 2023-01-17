import pandas as pd

import numpy as np



import matplotlib.pyplot as plt

plt.style.use('seaborn')

import seaborn as sns



import warnings

warnings.filterwarnings("ignore")



import time

import datetime
df = pd.read_csv('../input/berlin-airbnb-data/reviews_summary.csv')



print("The dataset has {} rows and {} columns.".format(*df.shape))



print("It contains {} duplicates.".format(df.duplicated().sum()))
df.head()
print("The dataset has {} rows and {} columns.".format(*df.shape))
df.isna().sum()
df.dropna(inplace=True)

df.isna().sum()
df.shape
from langdetect import detect
def language_detection(text):

    try:

        return detect(text)

    except:

        return None
%%time

df['language'] = df['comments'].apply(language_detection)
df.language.value_counts().head(10)
ax = df.language.value_counts(normalize=True).head(6).sort_values().plot(kind='barh', figsize=(9,5));
df_eng = df[(df['language']=='en')]

df_de  = df[(df['language']=='de')]

df_fr  = df[(df['language']=='fr')]
from nltk.corpus import stopwords

from wordcloud import WordCloud

from collections import Counter

from PIL import Image



import re

import string
def plot_wordcloud(wordcloud, language):

    plt.figure(figsize=(12, 10))

    plt.imshow(wordcloud, interpolation = 'bilinear')

    plt.axis("off")

    plt.title(language + ' Comments\n', fontsize=18, fontweight='bold')

    plt.show()
wordcloud = WordCloud(max_font_size=None, max_words=200, background_color="lightgrey", 

                      width=3000, height=2000,

                      stopwords=stopwords.words('english')).generate(str(df_eng.comments.values))



plot_wordcloud(wordcloud, 'English')
wordcloud = WordCloud(max_font_size=None, max_words=150, background_color="powderblue",

                      width=3000, height=2000,

                      stopwords=stopwords.words('german')).generate(str(df_de.comments.values))



plot_wordcloud(wordcloud, 'German')
wordcloud = WordCloud(max_font_size=200, max_words=150, background_color="lightgoldenrodyellow",

                      #width=1600, height=800,

                      width=3000, height=2000,

                      stopwords=stopwords.words('french')).generate(str(df_fr.comments.values))



plot_wordcloud(wordcloud, 'French')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
# assign it to another name to make it easier to use

analyzer = SentimentIntensityAnalyzer()
# getting only the negative score

def negative_score(text):

    negative_value = analyzer.polarity_scores(text)['neg']

    return negative_value



# getting only the neutral score

def neutral_score(text):

    neutral_value = analyzer.polarity_scores(text)['neu']

    return neutral_value



# getting only the positive score

def positive_score(text):

    positive_value = analyzer.polarity_scores(text)['pos']

    return positive_value



# getting only the compound score

def compound_score(text):

    compound_value = analyzer.polarity_scores(text)['compound']

    return compound_value
%%time



df_eng['sentiment_neg'] = df_eng['comments'].apply(negative_score)

df_eng['sentiment_neu'] = df_eng['comments'].apply(neutral_score)

df_eng['sentiment_pos'] = df_eng['comments'].apply(positive_score)

df_eng['sentiment_compound'] = df_eng['comments'].apply(compound_score)
df_eng.head()
df_pos = df_eng.loc[df_eng.sentiment_compound >= 0.7]



pos_comments = df_pos['comments'].tolist()
df_neg = df_eng.loc[df_eng.sentiment_compound < 0.0]



neg_comments = df_neg['comments'].tolist()
from nltk.corpus import stopwords

from nltk.stem.wordnet import WordNetLemmatizer

import string
stop = set(stopwords.words('english'))

exclude = set(string.punctuation)

lemma = WordNetLemmatizer()
def clean(doc):

    stop_free = " ".join([word for word in doc.lower().split() if word not in stop])

    punc_free = "".join(token for token in stop_free if token not in exclude)

    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())

    return normalized



doc_clean = [clean(comment).split() for comment in pos_comments] 
from gensim import corpora

dictionary = corpora.Dictionary(doc_clean)

corpus = [dictionary.doc2bow(text) for text in doc_clean]
import gensim

#find 3 topics

ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=3, id2word=dictionary, passes=15)





topics = ldamodel.print_topics(num_words=4)

for topic in topics:

    print(topic)
# find 5 topics

ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=5, id2word=dictionary, passes=15)



topics = ldamodel.print_topics(num_words=4)

for topic in topics:

    print(topic)
# 10 topics

ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=10, id2word=dictionary, passes=15)



topics = ldamodel.print_topics(num_words=4)

for topic in topics:

    print(topic)
doc_clean = [clean(comment).split() for comment in neg_comments]
dictionary = corpora.Dictionary(doc_clean)

corpus = [dictionary.doc2bow(text) for text in doc_clean]

# find 3 topics

ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=3, id2word=dictionary, passes=15)



topics = ldamodel.print_topics(num_words=4)

for topic in topics:

    print(topic)
#  find 5 topics

ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=5, id2word=dictionary, passes=15)



topics = ldamodel.print_topics(num_words=4)

for topic in topics:

    print(topic)
# 10 topics

ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=10, id2word=dictionary, passes=15)



topics = ldamodel.print_topics(num_words=4)

for topic in topics:

    print(topic)