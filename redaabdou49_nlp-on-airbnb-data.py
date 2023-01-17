import pandas as pd

import numpy as np



import matplotlib.pyplot as plt

plt.style.use('seaborn')

import seaborn as sns



import warnings

warnings.filterwarnings("ignore")



import time

import datetime
df_1 = pd.read_csv('../input/berlin-airbnb-data/reviews_summary.csv')



print("The dataset has {} rows and {} columns.".format(*df_1.shape))



print("It contains {} duplicates.".format(df_1.duplicated().sum()))
df_1.head()
df_2 = pd.read_csv('../input/berlin-airbnb-data/listings_summary.csv')

df_2.head()

df = pd.merge(df_1, df_2[['neighbourhood_group_cleansed', 'host_id', 'latitude',

                          'longitude', 'number_of_reviews', 'id', 'property_type']], 

              left_on='listing_id', right_on='id', how='left')



df.rename(columns = {'id_x':'id', 'neighbourhood_group_cleansed':'neighbourhood_group'}, inplace=True)

df.drop(['id_y'], axis=1, inplace=True)
df.head(3)
print("The dataset has {} rows and {} columns.".format(*df.shape))
df.isna().sum()
df.dropna(inplace=True)

df.isna().sum()
df.shape
# we use Python's langdetect 

from langdetect import detect
# write the function that detects the language

def language_detection(text):

    try:

        return detect(text)

    except:

        return None
%%time

df['language'] = df['comments'].apply(language_detection)
df.language.value_counts().head(10)
# visualizing the comments' languages a) quick and dirty

ax = df.language.value_counts(normalize=True).head(6).sort_values().plot(kind='barh', figsize=(9,5));
# splitting the dataframes in language related sub-dataframes

df_eng = df[(df['language']=='en')]

df_de  = df[(df['language']=='de')]

df_fr  = df[(df['language']=='fr')]
# import necessary libraries

from nltk.corpus import stopwords

from wordcloud import WordCloud

from collections import Counter

from PIL import Image



import re

import string
# wrap the plotting in a function for easier access

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
# load the SentimentIntensityAnalyser object in

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
# full dataframe with POSITIVE comments

df_pos = df_eng.loc[df_eng.sentiment_compound >= 0.7]



# only corpus of POSITIVE comments

pos_comments = df_pos['comments'].tolist()
# full dataframe with POSITIVE comments

df_neg = df_eng.loc[df_eng.sentiment_compound < 0.0]



# only corpus of POSITIVE comments

neg_comments = df_neg['comments'].tolist()
# importing libraries

from nltk.corpus import stopwords

from nltk.stem.wordnet import WordNetLemmatizer

import string
# prepare the preprocessing

stop = set(stopwords.words('english'))

exclude = set(string.punctuation)

lemma = WordNetLemmatizer()
# removing stopwords, punctuations and normalizing the corpus

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



# let LDA find 3 topics

ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=3, id2word=dictionary, passes=15)





topics = ldamodel.print_topics(num_words=4)

for topic in topics:

    print(topic)
# now let LDA find 5 topics

ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=5, id2word=dictionary, passes=15)



topics = ldamodel.print_topics(num_words=4)

for topic in topics:

    print(topic)
# and finally 10 topics

ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=10, id2word=dictionary, passes=15)



topics = ldamodel.print_topics(num_words=4)

for topic in topics:

    print(topic)
# calling the cleaning function we defined earlier

doc_clean = [clean(comment).split() for comment in neg_comments]
# create a dictionary from the normalized data, convert this to a bag-of-words corpus

dictionary = corpora.Dictionary(doc_clean)

corpus = [dictionary.doc2bow(text) for text in doc_clean]

# let LDA find 3 topics

ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=3, id2word=dictionary, passes=15)



topics = ldamodel.print_topics(num_words=4)

for topic in topics:

    print(topic)
# now let LDA find 5 topics

ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=5, id2word=dictionary, passes=15)



topics = ldamodel.print_topics(num_words=4)

for topic in topics:

    print(topic)
# and finally 10 topics

ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=10, id2word=dictionary, passes=15)



topics = ldamodel.print_topics(num_words=4)

for topic in topics:

    print(topic)