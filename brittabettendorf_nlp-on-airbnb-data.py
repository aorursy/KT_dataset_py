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

# checking shape ...

print("The dataset has {} rows and {} columns.".format(*df_1.shape))



# ... and duplicates

print("It contains {} duplicates.".format(df_1.duplicated().sum()))
df_1.head()
df_2 = pd.read_csv('../input/berlin-airbnb-data/listings_summary.csv')

df_2.head()

# merging full df_1 + add only specific columns from df_2

df = pd.merge(df_1, df_2[['neighbourhood_group_cleansed', 'host_id', 'latitude',

                          'longitude', 'number_of_reviews', 'id', 'property_type']], 

              left_on='listing_id', right_on='id', how='left')



df.rename(columns = {'id_x':'id', 'neighbourhood_group_cleansed':'neighbourhood_group'}, inplace=True)

df.drop(['id_y'], axis=1, inplace=True)
df.head(3)
# checking shape

print("The dataset has {} rows and {} columns.".format(*df.shape))
# group by hosts and count the number of unique listings --> cast it to a dataframe

properties_per_host = pd.DataFrame(df.groupby('host_id')['listing_id'].nunique())



# sort unique values descending and show the Top20

properties_per_host.sort_values(by=['listing_id'], ascending=False, inplace=True)

properties_per_host.head(20)
top1_host = df.host_id == 1625771

df[top1_host].neighbourhood_group.value_counts()



pd.DataFrame(df[top1_host].groupby('neighbourhood_group')['listing_id'].nunique())
pd.DataFrame(df[top1_host].groupby('property_type')['listing_id'].nunique())
top2_host = df.host_id == 8250486

df[top2_host].neighbourhood_group.value_counts()



pd.DataFrame(df[top2_host].groupby('neighbourhood_group')['listing_id'].nunique())
pd.DataFrame(df[top2_host].groupby('property_type')['listing_id'].nunique())
top3_host = df.host_id == 2293972

df[top3_host].neighbourhood_group.value_counts()



pd.DataFrame(df[top3_host].groupby('neighbourhood_group')['listing_id'].nunique())
pd.DataFrame(df[top3_host].groupby('property_type')['listing_id'].nunique())
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
# write the dataframe to a csv file in order to avoid the long runtime

#df.to_csv('../input/processed_df', index=False)

#df = pd.read_csv('../input/processed_df')
df.language.value_counts().head(10)
# visualizing the comments' languages a) quick and dirty

ax = df.language.value_counts(normalize=True).head(6).sort_values().plot(kind='barh', figsize=(9,5));
# visualizing the comments' languages b) neat and clean

ax = df.language.value_counts().head(6).plot(kind='barh', figsize=(9,5), color="lightcoral", 

                                             fontsize=12);



ax.set_title("\nWhat are the most frequent languages comments are written in?\n", 

             fontsize=12, fontweight='bold')

ax.set_xlabel(" Total Number of Comments", fontsize=10)

ax.set_yticklabels(['English', 'German', 'French', 'Spanish', 'Italian', 'Dutch'])



# create a list to collect the plt.patches data

totals = []

# find the ind. values and append to list

for i in ax.patches:

    totals.append(i.get_width())

# get total

total = sum(totals)



# set individual bar labels using above list

for i in ax.patches:

    ax.text(x=i.get_width(), y=i.get_y()+.35, 

            s=str(round((i.get_width()/total)*100, 2))+'%', 

            fontsize=10, color='black')



# invert for largest on top 

ax.invert_yaxis()
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
# use the polarity_scores() method to get the sentiment metrics

def print_sentiment_scores(sentence):

    snt = analyzer.polarity_scores(sentence)

    print("{:-<40} {}".format(sentence, str(snt)))
print_sentiment_scores("This raspberry cake is good.")
print_sentiment_scores("This raspberry cake is good.")

print_sentiment_scores("This raspberry cake is GOOD!")

print_sentiment_scores("This raspberry cake is VERY GOOD!!")

print_sentiment_scores("This raspberry cake is really GOOD! But the coffee is dreadful.")
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
negative_score("The food is really GOOD! But the service is dreadful.")
neutral_score("The food is really GOOD! But the service is dreadful.")
positive_score("The food is really GOOD! But the service is dreadful.")
compound_score("The food is really GOOD! But the service is dreadful.")
%%time



df_eng['sentiment_neg'] = df_eng['comments'].apply(negative_score)

df_eng['sentiment_neu'] = df_eng['comments'].apply(neutral_score)

df_eng['sentiment_pos'] = df_eng['comments'].apply(positive_score)

df_eng['sentiment_compound'] = df_eng['comments'].apply(compound_score)
# write the dataframe to a csv file in order to avoid the long runtime

#df_eng.to_csv('data/sentimentData/sentiment_df_eng', index=False)

#df = pd.read_csv('data/sentimentData/sentiment_df_eng')

df = df_eng
df.head(2)
# all scores in 4 histograms

fig, axes = plt.subplots(2, 2, figsize=(10,8))



# plot all 4 histograms

df.hist('sentiment_neg', bins=25, ax=axes[0,0], color='lightcoral', alpha=0.6)

axes[0,0].set_title('Negative Sentiment Score')

df.hist('sentiment_neu', bins=25, ax=axes[0,1], color='lightsteelblue', alpha=0.6)

axes[0,1].set_title('Neutral Sentiment Score')

df.hist('sentiment_pos', bins=25, ax=axes[1,0], color='chartreuse', alpha=0.6)

axes[1,0].set_title('Positive Sentiment Score')

df.hist('sentiment_compound', bins=25, ax=axes[1,1], color='navajowhite', alpha=0.6)

axes[1,1].set_title('Compound')



# plot common x- and y-label

fig.text(0.5, 0.04, 'Sentiment Scores',  fontweight='bold', ha='center')

fig.text(0.04, 0.5, 'Number of Reviews', fontweight='bold', va='center', rotation='vertical')



# plot title

plt.suptitle('Sentiment Analysis of Airbnb Reviews for Berlin\n\n', fontsize=12, fontweight='bold');
percentiles = df.sentiment_compound.describe(percentiles=[.05, .1, .2, .3, .4, .5, .6, .7, .8, .9])

percentiles
# assign the data

neg = percentiles['5%']

mid = percentiles['20%']

pos = percentiles['max']

names = ['Negative Comments', 'Okayish Comments','Positive Comments']

size = [neg, mid, pos]



# call a pie chart

plt.pie(size, labels=names, colors=['lightcoral', 'lightsteelblue', 'chartreuse'], 

        autopct='%.2f%%', pctdistance=0.8,

        wedgeprops={'linewidth':7, 'edgecolor':'white' })



# create circle for the center of the plot to make the pie look like a donut

my_circle = plt.Circle((0,0), 0.6, color='white')



# plot the donut chart

fig = plt.gcf()

fig.set_size_inches(7,7)

fig.gca().add_artist(my_circle)

plt.show()
# full dataframe with POSITIVE comments

df_pos = df.loc[df.sentiment_compound >= 0.95]



# only corpus of POSITIVE comments

pos_comments = df_pos['comments'].tolist()
# full dataframe with NEGATIVE comments

df_neg = df.loc[df.sentiment_compound < 0.0]



# only corpus of NEGATIVE comments

neg_comments = df_neg['comments'].tolist()
df_pos['text_length'] = df_pos['comments'].apply(len)

df_neg['text_length'] = df_neg['comments'].apply(len)
sns.set_style("whitegrid")

plt.figure(figsize=(8,5))



sns.distplot(df_pos['text_length'], kde=True, bins=50, color='chartreuse')

sns.distplot(df_neg['text_length'], kde=True, bins=50, color='lightcoral')



plt.title('\nDistribution Plot for Length of Comments\n')

plt.legend(['Positive Comments', 'Negative Comments'])

plt.xlabel('\nText Length')

plt.ylabel('Percentage of Comments\n');
# read some positive comments

pos_comments[10:15]
# read some negative comments

neg_comments[10:15]
sns.set_style("white")

cmap = sns.cubehelix_palette(rot=-.4, as_cmap=True)

fig, ax = plt.subplots(figsize=(11,7))



ax = sns.scatterplot(x="longitude", y="latitude", size='number_of_reviews', sizes=(5, 200),

                     hue='sentiment_compound', palette=cmap,  data=df)

ax.legend(bbox_to_anchor=(1.3, 1), borderaxespad=0.)

plt.title('\nAccommodations in Berlin by Number of Reviwws & Sentiment\n', fontsize=12, fontweight='bold')



sns.despine(ax=ax, top=True, right=True, left=True, bottom=True);
wordcloud = WordCloud(max_font_size=200, max_words=200, background_color="palegreen",

                      width= 3000, height = 2000,

                      stopwords = stopwords.words('english')).generate(str(df_pos.comments.values))



plot_wordcloud(wordcloud, '\nPositively Tuned')
# importing libraries

from sklearn.feature_extraction.text import CountVectorizer

from yellowbrick.text.freqdist import FreqDistVisualizer

from yellowbrick.style import set_palette
# vectorizing text

vectorizer = CountVectorizer(stop_words='english')

docs = vectorizer.fit_transform(pos_comments)

features = vectorizer.get_feature_names()



# preparing the plot

set_palette('pastel')

plt.figure(figsize=(18,8))

plt.title('The Top 30 most frequent words used in POSITIVE comments\n', fontweight='bold')



# instantiating and fitting the FreqDistVisualizer, plotting the top 30 most frequent terms

visualizer = FreqDistVisualizer(features=features, n=30)

visualizer.fit(docs)

visualizer.poof;
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



import pickle 

# uncomment the code if working locally

#pickle.dump(corpus, open('data/sentimentData/corpus.pkl', 'wb'))

#dictionary.save('data/sentimentData/dictionary.gensim')
import gensim



# let LDA find 3 topics

ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=3, id2word=dictionary, passes=15)



# uncomment the code if working locally

#ldamodel.save('../input/sentimentData/model3.gensim')



topics = ldamodel.print_topics(num_words=4)

for topic in topics:

    print(topic)
# now let LDA find 5 topics

ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=5, id2word=dictionary, passes=15)



# uncomment the code if working locally

#ldamodel.save('../input/sentimentData/model5.gensim')



topics = ldamodel.print_topics(num_words=4)

for topic in topics:

    print(topic)
# and finally 10 topics

ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=10, id2word=dictionary, passes=15)



# uncomment the code if working locally

#ldamodel.save('../input/sentimentData/model10.gensim')



topics = ldamodel.print_topics(num_words=4)

for topic in topics:

    print(topic)
#dictionary = gensim.corpora.Dictionary.load('../input/sentimentData/dictionary.gensim')

#corpus = pickle.load(open('../input/sentimentData/corpus.pkl', 'rb'))



#import pyLDAvis.gensim
# visualizing 5 topics

#lda = gensim.models.ldamodel.LdaModel.load('../input/sentimentData/model5.gensim')

#lda_display = pyLDAvis.gensim.prepare(lda, corpus, dictionary, sort_topics=False)

#pyLDAvis.display(lda_display)
# visualizing 3 topics

#lda = gensim.models.ldamodel.LdaModel.load('../input/sentimentData/model3.gensim')

#lda_display = pyLDAvis.gensim.prepare(lda, corpus, dictionary, sort_topics=False)

#pyLDAvis.display(lda_display)
wordcloud = WordCloud(max_font_size=200, max_words=200, background_color="mistyrose",

                      width=3000, height=2000,

                      stopwords=stopwords.words('english')).generate(str(df_neg.comments.values))



plot_wordcloud(wordcloud, '\nNegatively Tuned')
# vectorizing text

vectorizer = CountVectorizer(stop_words='english')

docs = vectorizer.fit_transform(neg_comments)

features = vectorizer.get_feature_names()



# preparing the plot

set_palette('flatui')

plt.figure(figsize=(18,8))

plt.title('The Top 30 most frequent words used in NEGATIVE comments\n', fontweight='bold')



# instantiating and fitting the FreqDistVisualizer, plotting the top 30 most frequent terms

visualizer = FreqDistVisualizer(features=features, n=30)

visualizer.fit(docs)

visualizer.poof;
# calling the cleaning function we defined earlier

doc_clean = [clean(comment).split() for comment in neg_comments]
# create a dictionary from the normalized data, convert this to a bag-of-words corpus

dictionary = corpora.Dictionary(doc_clean)

corpus = [dictionary.doc2bow(text) for text in doc_clean]



# save for later use

# uncomment the code if working locally

#pickle.dump(corpus, open('data/sentimentData/corpus_neg.pkl', 'wb'))

#dictionary.save('data/sentimentData/dictionary_neg.gensim')
# let LDA find 3 topics

ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=3, id2word=dictionary, passes=15)



# uncomment the code if working locally

#ldamodel.save('../input/sentimentData/model3_neg.gensim')



topics = ldamodel.print_topics(num_words=4)

for topic in topics:

    print(topic)
# now let LDA find 5 topics

ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=5, id2word=dictionary, passes=15)



# uncomment the code if working locally

#ldamodel.save('../input/sentimentData/model5_neg.gensim')



topics = ldamodel.print_topics(num_words=4)

for topic in topics:

    print(topic)
# and finally 10 topics

ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=10, id2word=dictionary, passes=15)



# uncomment the code if working locally

#ldamodel.save('../input/sentimentData/model10_neg.gensim')



topics = ldamodel.print_topics(num_words=4)

for topic in topics:

    print(topic)
#dictionary = gensim.corpora.Dictionary.load('./input/sentimentData/dictionary_neg.gensim')

#corpus = pickle.load(open('./input/sentimentData/corpus_neg.pkl', 'rb'))
# visualizing 5 topics

#lda = gensim.models.ldamodel.LdaModel.load('../input/sentimentData/model5_neg.gensim')

#lda_display = pyLDAvis.gensim.prepare(lda, corpus, dictionary, sort_topics=False)

#pyLDAvis.display(lda_display)
# visualizing 3 topics

#lda = gensim.models.ldamodel.LdaModel.load('../input/sentimentDatamodel3_neg.gensim')

#lda_display = pyLDAvis.gensim.prepare(lda, corpus, dictionary, sort_topics=False)

#pyLDAvis.display(lda_display)