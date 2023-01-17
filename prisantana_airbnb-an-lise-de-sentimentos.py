import pandas as pd

from re import sub

import numpy as np



from numpy import asarray

import matplotlib.pyplot as plt

plt.style.use('seaborn')

import seaborn as sns



import warnings

warnings.filterwarnings("ignore")





import time

import datetime

import nltk.data



from nltk.corpus import stopwords

from nltk.tokenize import RegexpTokenizer

from nltk.stem import PorterStemmer 

from nltk import download

 





from nltk.classify import NaiveBayesClassifier

from nltk.corpus import subjectivity



download('stopwords')

nltk.download('wordnet')





rw = pd.read_csv('http://data.insideairbnb.com/brazil/rj/rio-de-janeiro/2019-11-22/data/reviews.csv.gz')

li = pd.read_csv('http://data.insideairbnb.com/brazil/rj/rio-de-janeiro/2019-11-22/data/listings.csv.gz')





rw.head()
rw.info()

rw.nunique()

rw.isnull().sum()
li.head(4)
li.shape
li.info()

df = pd.merge(rw, li[['neighbourhood_cleansed', 'host_id', 'latitude', 'host_name',

                          'longitude', 'number_of_reviews', 'id', 'property_type']], 

              left_on='listing_id', right_on='id', how='left')



df.rename(columns = {'id_x':'id', 'neighbourhood_cleansed':'neighbourhood'}, inplace=True)

df.drop(['id_y'], axis=1, inplace=True)
df.head(3)
print("O datasete tem {} linhas e {} colunas.".format(*df.shape))
properties_per_host = pd.DataFrame(df.groupby('host_id',)['listing_id'].nunique())



# sort unique values descending and show the Top10

properties_per_host.sort_values(by=['listing_id'], ascending=False, inplace=True)

properties_per_host.head(10)



properties_per_host = pd.DataFrame(df.groupby('host_id')['listing_id'].nunique())



# sort unique values descending and show the Top10

properties_per_host.sort_values(by=['listing_id'], ascending=False, inplace=True)

properties_per_host.head(3)
top1_host = df.host_id == 81876389

df[top1_host].neighbourhood.value_counts()



pd.DataFrame(df[top1_host].groupby('neighbourhood')['listing_id'].nunique())
pd.DataFrame(df[top1_host].groupby('property_type')['listing_id'].nunique())
top2_host = df.host_id == 1982737

df[top2_host].neighbourhood.value_counts()



pd.DataFrame(df[top2_host].groupby('neighbourhood')['listing_id'].nunique())
pd.DataFrame(df[top1_host].groupby('property_type')['listing_id'].nunique())
top3_host = df.host_id == 31275569

df[top3_host].neighbourhood.value_counts()



pd.DataFrame(df[top3_host].groupby('neighbourhood')['listing_id'].nunique())
pd.DataFrame(df[top3_host].groupby('property_type')['listing_id'].nunique())
df.isna().sum()

df.isna().sum()

df.dropna(inplace=True)

df.isna().sum()
df.shape
from nltk.sentiment.vader import SentimentIntensityAnalyzer

from nltk.sentiment.util import *

import nltk.data

nltk.download('vader_lexicon')

nltk.download('punkt')



        
analyzer = SentimentIntensityAnalyzer()

def print_sentiment_scores(sentence):

    snt = analyzer.polarity_scores(sentence)

    print("{:-<40} {}".format(sentence, str(snt)))
print_sentiment_scores("O dia é muito bom.")

print_sentiment_scores("The book is good.")

print_sentiment_scores("The book is GOOD!")

print_sentiment_scores("the book is very GOOD!!")

print_sentiment_scores("This book is really GOOD! But the price is horrible.")

print_sentiment_scores("The book is not good.")

print_sentiment_scores("The book is dirty.")
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
negative_score("This book is really GOOD! But the price is horrible.")
neutral_score("This book is really GOOD! But the price is horrible.")
positive_score("This book is really GOOD! But the price is horrible.")

compound_score("This book is really GOOD! But the price is horrible.")
%%time



df['sentiment_neg'] = df['comments'].apply(negative_score)

df['sentiment_neu'] = df['comments'].apply(neutral_score)

df['sentiment_pos'] = df['comments'].apply(positive_score)

df['sentiment_compound'] = df['comments'].apply(compound_score)
fig, axes = plt.subplots(2, 2, figsize=(10,8))





df.hist('sentiment_neg', bins=25, ax=axes[0,0], color='lightcoral', alpha=0.6)

axes[0,0].set_title('Negative Sentiment Score')

df.hist('sentiment_neu', bins=25, ax=axes[0,1], color='lightsteelblue', alpha=0.6)

axes[0,1].set_title('Neutral Sentiment Score')

df.hist('sentiment_pos', bins=25, ax=axes[1,0], color='chartreuse', alpha=0.6)

axes[1,0].set_title('Positive Sentiment Score')

df.hist('sentiment_compound', bins=25, ax=axes[1,1], color='navajowhite', alpha=0.6)

axes[1,1].set_title('Compound')





fig.text(0.5, 0.04, 'Sentiment Scores',  fontweight='bold', ha='center')

fig.text(0.04, 0.5, 'Números de avaliações', fontweight='bold', va='center', rotation='vertical')





plt.suptitle('Analise de Sentimentos Airbnb das avaliações do Rio de Janeiro', fontsize=12, fontweight='bold');
percentiles = df.sentiment_compound.describe(percentiles=[.05, .1, .2, .3, .4, .5, .6, .7, .8, .9])

percentiles
neg = percentiles['5%']

mid = percentiles['20%']

pos = percentiles['max']

names = ['Comentários Negativos', 'Comentários Ok','Comentários Positivos']

size = [neg, mid, pos]





plt.pie(size, labels=names, colors=['lightcoral', 'lightsteelblue', 'chartreuse'], 

        autopct='%.2f%%', pctdistance=0.8,

        wedgeprops={'linewidth':7, 'edgecolor':'white' })



my_circle = plt.Circle((0,0), 0.6, color='white')





fig = plt.gcf()

fig.set_size_inches(7,7)

fig.gca().add_artist(my_circle)

plt.show()
df_pos = df.loc[df.sentiment_compound >= 0.95]





pos_comments = df_pos['comments'].tolist()
df_neg = df.loc[df.sentiment_compound < 0.0]





neg_comments = df_neg['comments'].tolist()
df_pos['text_length'] = df_pos['comments'].apply(len)

df_neg['text_length'] = df_neg['comments'].apply(len)
sns.set_style("whitegrid")

plt.figure(figsize=(8,5))



sns.distplot(df_pos['text_length'], kde=True, bins=50, color='chartreuse')

sns.distplot(df_neg['text_length'], kde=True, bins=50, color='lightcoral')



plt.title('\nGráfico de distribuição por tamanho de comentários\n')

plt.legend(['Comentários Positivos', 'Comentários Negativos'])

plt.xlabel('\nTamanho do Texto')

plt.ylabel('Percentual de comentários\n');
pos_comments[10:15]
neg_comments[10:15]
cmap = sns.cubehelix_palette(rot=-.4, as_cmap=True)

fig, ax = plt.subplots(figsize=(11,7))



ax = sns.scatterplot(x="longitude", y="latitude", size='number_of_reviews', sizes=(5, 200),

                     hue='sentiment_compound', palette=cmap,  data=df)

ax.legend(bbox_to_anchor=(1.3, 1), borderaxespad=0.)

plt.title('\nAcomodações no Rio de Janeiro por Números de avaliações\n', fontsize=12, fontweight='bold')



sns.despine(ax=ax, top=True, right=True, left=True, bottom=True);
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


wordcloud = WordCloud(max_font_size=200, max_words=200, background_color='white',

                      width= 3000, height = 2000,

                      stopwords = stopwords.words('english')).generate(str(df_pos.comments.values))



plot_wordcloud(wordcloud, '\nPalavras usadas nas avaliações positivas')
from sklearn.feature_extraction.text import CountVectorizer

from yellowbrick.text.freqdist import FreqDistVisualizer

from yellowbrick.style import set_palette
vectorizer = CountVectorizer(stop_words='english')

docs = vectorizer.fit_transform(pos_comments)

features = vectorizer.get_feature_names()





set_palette('pastel')

plt.figure(figsize=(18,8))

plt.title('Top 30 palavras mais usadas positivamente\n', fontweight='bold')





visualizer = FreqDistVisualizer(features=features, n=30)

visualizer.fit(docs)

visualizer.poof;
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



import pickle 
import gensim
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=3, id2word=dictionary, passes=15)
topics = ldamodel.print_topics(num_words=4)

for topic in topics:

    print(topic)
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=5, id2word=dictionary, passes=15)

topics = ldamodel.print_topics(num_words=4)

for topic in topics:

    print(topic)
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=10, id2word=dictionary, passes=15)
topics = ldamodel.print_topics(num_words=4)

for topic in topics:

    print(topic)
wordcloud = WordCloud(max_font_size=200, max_words=200, background_color="white",

                      width=3000, height=2000,

                      stopwords=stopwords.words('english')).generate(str(df_neg.comments.values))



plot_wordcloud(wordcloud, '\nPalavras usadas negativamente')
vectorizer = CountVectorizer(stop_words='english')

docs = vectorizer.fit_transform(neg_comments)

features = vectorizer.get_feature_names()





set_palette('flatui')

plt.figure(figsize=(18,8))

plt.title('AS 30 palavras mais usadas em comentários negativos\n', fontweight='bold')



visualizer = FreqDistVisualizer(features=features, n=30)

visualizer.fit(docs)

visualizer.poof;
doc_clean = [clean(comment).split() for comment in neg_comments]
dictionary = corpora.Dictionary(doc_clean)

corpus = [dictionary.doc2bow(text) for text in doc_clean]

ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=3, id2word=dictionary, passes=15)
topics = ldamodel.print_topics(num_words=4)

for topic in topics:

    print(topic)
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=5, id2word=dictionary, passes=15)
topics = ldamodel.print_topics(num_words=4)

for topic in topics:

    print(topic)
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=10, id2word=dictionary, passes=15)
topics = ldamodel.print_topics(num_words=4)

for topic in topics:

    print(topic)