# Link workspace to Comet experiment

# !pip install comet_ml

# from comet_ml import Experiment

# experiment = Experiment(api_key="zNkJjcVKOMD5gKd05z6CwT4OD", project_name="team-rm5-sigmoidfreuds", workspace="lizette95")
# Ignore warnings

import warnings

warnings.simplefilter(action='ignore')



# Install Prerequisites

# import sys

# import nltk

# !{sys.executable} -m pip install bs4 lxml wordcloud scikit-learn scikit-plot

# nltk.download('vader_lexicon')



# Exploratory Data Analysis

import re

import ast

import time

import nltk

import pickle

import numpy as np

import pandas as pd

import seaborn as sns

from textblob import TextBlob

import matplotlib.pyplot as plt

from wordcloud import WordCloud

from nltk.sentiment import SentimentIntensityAnalyzer



# Data Preprocessing

import string

from bs4 import BeautifulSoup

from collections import Counter

from nltk.corpus import wordnet

from nltk.corpus import stopwords

from sklearn.utils import resample

from nltk.stem import WordNetLemmatizer

from nltk.tokenize import TweetTokenizer 

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfVectorizer



# Classification Models

from sklearn.svm import LinearSVC, SVC

from sklearn.naive_bayes import MultinomialNB

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import AdaBoostClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression



# Performance Evaluation

from sklearn.metrics import roc_curve, auc

from sklearn.preprocessing import label_binarize

from sklearn.model_selection import GridSearchCV

from scikitplot.metrics import plot_roc, plot_confusion_matrix

from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier

from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, classification_report, confusion_matrix



# Display

%matplotlib inline

sns.set(font_scale=1)

sns.set_style("white")
train_data = pd.read_csv('../input/climate-change-belief-analysis/train.csv')

test_data = pd.read_csv('../input/climate-change-belief-analysis/test.csv')

df_train = train_data.copy() #For EDA on raw data

df_test = test_data.copy()
train_data.head()
test_data.head()
## Initial Cleaning

# contractions = {"ain't": "am not", "aren't": "are not", "can't": "cannot", "can't've": "cannot have", "'cause": "because", "could've": "could have", "couldn't": "could not", "couldn't've": "could not have", "didn't": "did not", "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hadn't've": "had not have", "hasn't": "has not", "haven't": "have not", "he'd": "he would", "he'd've": "he would have", "he'll": "he will", "he'll've": "he will have", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is", "i'd": "I would", "i'd've": "I would have", "i'll": "I will", "i'll've": "I will have", "i'm": "I am", "i've": "I have", "isn't": "is not", "it'd": "it had", "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have", "it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have", "mightn't": "might not", "mightn't've": "might not have", "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have", "o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is", "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have", "so's": "so is", "that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there had", "there'd've": "there would have", "there's": "there is", "they'd": "they would", "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we had", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are", "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is", "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have", "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all", "y'alls": "you alls", "y'all'd": "you all would", "y'all'd've": "you all would have", "y'all're": "you all are", "y'all've": "you all have", "you'd": "you had", "you'd've": "you would have", "you'll": "you you will", "you'll've": "you you will have", "you're": "you are", "you've": "you have"}

# contractions_re = re.compile('(%s)' % '|'.join(contractions.keys()))

# def expand_contractions(text):

#     """

#     Expand contractions using dictionary of contractions.

#     """

#     def replace(match):

#         return contractions[match.group(0)]

#     return contractions_re.sub(replace, text)



# punctuation = string.punctuation + '0123456789'

# stop_words = stopwords.words('english')

# stop_words.extend(['via','rt'])



# def clean(df):

#     """

#     Apply data cleaning steps to raw data.

#     """

#     df['lower'] = df['message'].str.lower()

#     df['html'] = df['lower'].apply(lambda x : BeautifulSoup(x, "lxml").text)

#     df['url'] = df['html'].apply(lambda x : re.sub(r'http\S+', 'weblink', x))

#     df['newlines'] = df['url'].apply(lambda x : x.replace('\n',' '))

#     df['mentions'] = df['newlines'].apply(lambda x : re.sub('@\w*', 'twittermention', x))

#     df['hashtags'] = df['mentions'].apply(lambda x : re.sub('#\w*', 'hashtag', x))

#     df['expand'] = df['hashtags'].apply(expand_contractions)

#     df['token'] = df['expand'].apply(TweetTokenizer().tokenize)

#     df['stopwords'] = df['token'].apply(lambda x : [word for word in x if word not in stop_words])

#     df['punc'] = df['stopwords'].apply(lambda x : [i for i in x if i not in punctuation])

#     df['dig'] = df['punc'].apply(lambda x: [i for i in x if i not in list(string.digits)])

#     df['final'] = df['dig'].apply(lambda x: [i for i in x if len(i) > 1])

#     return df['final']



# train_data['final'] = clean(train_data)

# test_data['final'] = clean(test_data)
# Final Cleaning

def sentiment_changer(df):

    """

    Change key words to reflect the general sentiment associated with it.

    """

    df['message'] = df['message'].apply(lambda x: x.replace('global', 'negative'))

    df['message'] = df['message'].apply(lambda x: x.replace('climate', 'positive'))

    df['message'] = df['message'].apply(lambda x: x.replace('MAGA', 'negative'))

    return df['message']



train_data['message'] = sentiment_changer(train_data)

test_data['message'] = sentiment_changer(test_data)



def clean(df):

    """

    Apply data cleaning steps to raw data.

    """

    df['token'] = df['message'].apply(TweetTokenizer().tokenize) ## first we tokenize

    df['punc'] = df['token'].apply(lambda x : [i for i in x if i not in string.punctuation])## remove punctuations

    df['dig'] = df['punc'].apply(lambda x: [i for i in x if i not in list(string.digits)]) ## remove digits

    df['final'] = df['dig'].apply(lambda x: [i for i in x if len(i) > 1]) ## remove all words with only 1 character

    return df['final']



train_data['final'] = clean(train_data)

test_data['final'] = clean(test_data)
train_data.info()
test_data.info()
# Number of Tweets Per Sentiment Class

fig, axis = plt.subplots(ncols=2, figsize=(15, 5))



ax = sns.countplot(x='sentiment', data=train_data, palette='winter', ax=axis[0])

axis[0].set_title('Number of Tweets Per Sentiment Class',fontsize=14)

axis[0].set_xlabel('Sentiment Class')

axis[0].set_ylabel('Tweets')

for p in ax.patches:

    ax.text(p.get_x() + p.get_width()/2., p.get_height(), '%d' % int(p.get_height()), fontsize=11, ha='center', va='bottom')



train_data['sentiment'].value_counts().plot.pie(autopct='%1.1f%%', colormap='winter_r', ax=axis[1])

axis[1].set_title('Proportion of Tweets Per Sentiment Class',fontsize=14)

axis[1].set_ylabel('Sentiment Class')

    

plt.show()
# Remove noise

stop_words = stopwords.words('english')

stop_words.extend(['via','rt'])



def remove_noise(tweet):

    """

    Remove noise from text data, such as newlines, punctuation, URLs and numbers.

    """

    new_tweet = BeautifulSoup(tweet, "lxml").text #HTML Decoding

    new_tweet = re.sub(r'http\S+', '', new_tweet) #Remove URLs

    new_tweet = new_tweet.lower() #Remove Capital Letters

    new_tweet = new_tweet.replace('\n',' ') #Remove Newlines

    new_tweet = re.sub('#(RT|rt)*', '', new_tweet) #Remove RT #s

    new_tweet = re.sub('@\w*', '', new_tweet) #Remove Mentions

    new_tweet = re.sub('\w*\d\w*','', new_tweet) #Remove Numbers/Words with Numbers

    new_tweet = re.sub('[^a-zA-z\s]', '', new_tweet) #Remove Punctuation

    new_tweet = ' '.join(word for word in new_tweet.split() if word not in stop_words) #Remove Stopwords

    return new_tweet



df_train['noise'] = df_train['message'].apply(remove_noise)

df_test['noise'] = df_test['message'].apply(remove_noise)
# Make Wordclouds

def wc(df):

    """

    Join words to create wordclouds.

    """

    words = ''

    for i in df:  

        words += i+" "

    return words

# Training Set

train_words = wc(df_train['noise'])

train_wordcloud = WordCloud(width=1500, height=700, background_color='white', colormap='winter', min_font_size=10).generate(train_words)

# Testing Set

test_words = wc(df_test['noise'])

test_wordcloud = WordCloud(width=1500, height=700, background_color='white', colormap='winter_r', min_font_size=10).generate(test_words)
# Wordcloud Per Sentiment Class

fig, axis = plt.subplots(nrows=2, ncols=2, figsize=(18, 12))



news = wc(df_train['noise'][df_train['sentiment']==2])

news_wordcloud = WordCloud(width=900, height=600, background_color='white', colormap='winter').generate(news)

axis[0, 0].imshow(news_wordcloud)

axis[0, 0].set_title('News (2)',fontsize=14)

axis[0, 0].axis("off") 



neutral = wc(df_train['noise'][df_train['sentiment']==0])

neutral_wordcloud = WordCloud(width=900, height=600, background_color='white', colormap='winter', min_font_size=10).generate(neutral)

axis[1, 0].imshow(neutral_wordcloud)

axis[1, 0].set_title('Neutral (0)',fontsize=14)

axis[1, 0].axis("off") 



pro = wc(df_train['noise'][df_train['sentiment']==1])

pro_wordcloud = WordCloud(width=900, height=600, background_color='white', colormap='winter', min_font_size=10).generate(pro)

axis[0, 1].imshow(pro_wordcloud)

axis[0, 1].set_title('Pro (1)',fontsize=14)

axis[0, 1].axis("off") 



anti = wc(df_train['noise'][df_train['sentiment']==-1])

anti_wordcloud = WordCloud(width=900, height=600, background_color='white', colormap='winter', min_font_size=10).generate(anti)

axis[1, 1].imshow(anti_wordcloud)

axis[1, 1].set_title('Anti (-1)',fontsize=14)

axis[1, 1].axis("off") 



plt.show()
# Wordcloud for Training Data

plt.figure(figsize = (15, 7), facecolor = None) 

plt.title("Training Set",fontsize=20)

plt.imshow(train_wordcloud) 

plt.axis("off") 

plt.tight_layout(pad = 0)



plt.show() 
# Wordcloud for Testing Data

plt.figure(figsize = (15, 7), facecolor = None) 

plt.title("Testing Set",fontsize=20)

plt.imshow(test_wordcloud) 

plt.axis("off") 

plt.tight_layout(pad = 0)



plt.show() 
df1_train = pd.read_csv('../input/climate-change-belief-analysis/train.csv')



drop1 = df1_train.drop_duplicates(['message', 'sentiment'])['tweetid'].values

drop2 = df1_train.drop_duplicates(['message'])['tweetid'].values

print(f"Data is duplicated {len(df1_train)-len(drop2)} times")



# Check for tweets that are the same but have different sentiments

ids = [tweet_id for tweet_id in drop1 if tweet_id not in drop2]

repeated_tweets = []



for tweet_id in ids:

    tweet = df_train[df1_train['tweetid']==tweet_id]['message']

    repeated_tweets.append(tweet.values[0])



print('\n\n Tweets that are duplicate but have different sentiments:')

for repeated_tweet in repeated_tweets:

    print(df1_train[df_train['message']==repeated_tweet])



# Get dropped rows 

# https://stackoverflow.com/questions/28901683/pandas-get-rows-which-are-not-in-other-dataframe

dropped_rows = df1_train[~df1_train['tweetid'].isin(drop2)].dropna()



# Drop all duplicates from data set

df1_train.drop_duplicates(['message', 'sentiment'], inplace=True)



# Drop duplicates with inconsistent classes

inds_to_drop = []

for tweet_id in ids:

    inds_to_drop = df1_train[df_train['tweetid']==tweet_id].index[0] 



df1_train.drop(index=inds_to_drop, inplace = True)



df1_train.set_index('tweetid', inplace=True)
# Vader sentiment

sid = SentimentIntensityAnalyzer() ## instantiate Sentiment analyzer



def sent_decider(compound):

    """

    Function to determine if sentiment is positive, nuetral or negative.

    """

    neutral_point = 0.00

    if compound > neutral_point:

        return 'positive'#1

    elif compound < -neutral_point:

        return 'negative' #-1

    else: 

        return 'neutral'#0



# Get sentiment

df1_train['scores'] = df1_train['message'].apply(lambda review: sid.polarity_scores(review))

df1_train['compound']  = df1_train['scores'].apply(lambda score_dict: score_dict['compound'])

df1_train['comp_score'] = df1_train['compound'].apply(sent_decider)



def put_numbers_on_bars(axis_object):

    """

    Function to plot labels above countplot bars.

    """

    for p in axis_object.patches:

        axis_object.text(p.get_x() + p.get_width()/2., p.get_height(),'%d' % round(p.get_height()), fontsize=11,ha='center', va='bottom')



fig, axis = plt.subplots(ncols=2, figsize=(15, 5))



ax = sns.countplot(x='comp_score', data=df1_train, palette='winter', hue='sentiment', ax=axis[0])

axis[0].set_title('Number of Tweets Per Vader Sentiment',fontsize=14)

axis[0].set_xlabel('')

axis[0].set_ylabel('Tweets')

put_numbers_on_bars(ax)



ax = sns.countplot(x='sentiment', data=df1_train, palette='winter', hue='comp_score', ax=axis[1])

axis[1].set_title('Number of Tweets Per Sentiment Class',fontsize=14)

axis[1].set_xlabel('')

axis[1].set_ylabel('')

put_numbers_on_bars(ax)



plt.show()



# Plots as percentages

fig, axis = plt.subplots(ncols=2, figsize=(15, 5))

counts = (df1_train.groupby(['comp_score'])['sentiment'].value_counts(normalize=True).rename('percentage_tweets').mul(100).reset_index())



ax = sns.barplot(x="comp_score", y="percentage_tweets", palette='winter', hue="sentiment", data=counts, ax=axis[0])

put_numbers_on_bars(ax)

ax.set_xlabel('Vader Sentiment')

ax.set_ylabel('Tweets (%)')



count = (df1_train.groupby(['sentiment'])['comp_score'].value_counts(normalize=True).rename('percentage_tweets').mul(100).reset_index())



ax = sns.barplot(x="sentiment", y="percentage_tweets", palette='winter', hue="comp_score", data=counts, ax=axis[1])

put_numbers_on_bars(ax)

plt.xlabel('Sentiment Class')

plt.ylabel('')



# Plot pie charts

fig, axis = plt.subplots(ncols=4, figsize=(20, 5))

group = ["(Anti)", "(Neutral)", "(Pro)", "(News)"]

for i in range(4):

    df1_train[df1_train['sentiment']==i-1]['comp_score'].value_counts().plot.pie(autopct='%1.1f%%',colormap='winter_r',ax=axis[i])

    axis[i].set_title('Proportion of Tweets Per Vader Sentiment '+ group[i],fontsize=12)

    axis[i].set_ylabel('')

plt.show()
def subjectivity(text):

    """

    Calculate subjectivity per tweet.

    """

    return TextBlob(text).sentiment.subjectivity

df_train['Subjectivity'] = df_train['message'].apply(subjectivity)
# Sentiment Count from Subjectivity Threshold = 0.5

df_train['Subj'] = ["> 0.5" if i >=0.5 else "< 0.5" for i in df_train['Subjectivity']]

plt.figure(figsize = (8,5))

sns.countplot(x='sentiment', data=df_train, palette='winter', hue='Subj')

plt.xlabel('sentiment')

plt.ylabel('Counts')

plt.legend(title = 'Subjectivity')

plt.title('Sentiment Count from Subjectivity Threshold = 0.5',fontsize=14)

plt.show()
# Tweet Character lengths

df_train['number_of_characters'] = df_train['message'].apply(lambda x: len(x))



# Distribution of Character Count Per Sentiment Class

fig, axis = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))

fig.suptitle('Distribution of Character Count Per Sentiment Class', fontsize=14)



bins = range(0, 180, 10)



train_news = df_train[df_train['sentiment']==2]

axis[0, 0].hist(train_news['number_of_characters'], bins=bins, weights=np.zeros(len(train_news)) + 1. / len(train_news), color='#0017F3')

axis[0, 0].set_title('News (2)')

axis[0, 0].set_ylabel('Frequency')



train_neutral = df_train[df_train['sentiment']==0]

axis[1, 0].hist(train_neutral['number_of_characters'], bins=bins, weights=np.zeros(len(train_neutral)) + 1. / len(train_neutral), color='#008BB9')

axis[1, 0].set_title('Neutral (0)')

axis[1, 0].set_ylabel('Frequency')

axis[1, 0].set_xlabel('Character Count')



train_pro = df_train[df_train['sentiment']==1]

axis[0, 1].hist(train_pro['number_of_characters'], bins=bins, weights=np.zeros(len(train_pro)) + 1. / len(train_pro), color='#00BAA2')

axis[0, 1].set_title('Pro (1)')



train_anti = df_train[df_train['sentiment']==-1]

axis[1, 1].hist(train_anti['number_of_characters'], bins=bins, weights=np.zeros(len(train_anti)) + 1. / len(train_anti), color='#00E88B')

axis[1, 1].set_title('Anti (-1)')

axis[1, 1].set_xlabel('Character Count')



plt.show()
# Tweet word lengths

df_train['number_of_words'] = df_train['message'].apply(lambda x: len(x.split(' ')))



# Distribution of Word Count Per Sentiment Class

fig, axis = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))

fig.suptitle('Distribution of Word Count Per Sentiment Class', fontsize=14)



bins = range(0, 30, 2)



train_news = df_train[df_train['sentiment']==2]

axis[0, 0].hist(train_news['number_of_words'], bins=bins, weights=np.zeros(len(train_news)) + 1. / len(train_news), color='#0017F3')

axis[0, 0].set_title('News (2)')

axis[0, 0].set_ylabel('Frequency')



train_neutral = df_train[df_train['sentiment']==0]

axis[1, 0].hist(train_neutral['number_of_words'], bins=bins, weights=np.zeros(len(train_neutral)) + 1. / len(train_neutral), color='#008BB9')

axis[1, 0].set_title('Neutral (0)')

axis[1, 0].set_ylabel('Frequency')

axis[1, 0].set_xlabel('Word Count')



train_pro = df_train[df_train['sentiment']==1]

axis[0, 1].hist(train_pro['number_of_words'], bins=bins, weights=np.zeros(len(train_pro)) + 1. / len(train_pro), color='#00BAA2')

axis[0, 1].set_title('Pro (1)')



train_anti = df_train[df_train['sentiment']==-1]

axis[1, 1].hist(train_anti['number_of_words'], bins=bins, weights=np.zeros(len(train_anti)) + 1. / len(train_anti), color='#00E88B')

axis[1, 1].set_title('Anti (-1)')

axis[1, 1].set_xlabel('Word Count')



plt.show()
# Check if URLs have anything to do with distribution of sentiments.

# Count URLs

url_text = r"https?://t\.co/\S+"



df_train['url_list'] = df_train['message'].apply(lambda tweet: re.findall(url_text, tweet))



df_train['url_count'] = df_train['url_list'].apply(len)



# Plot sentiment

plt.figure(figsize = (8,5))

sns.countplot(x='sentiment', data=df_train, palette='winter', hue='url_count')

plt.title('Number of URLs Per Sentiment Class',fontsize=14)

plt.xlabel('Sentiment Class')

plt.ylabel('Tweet Frequency')

plt.legend(title='URL Count')

plt.show()
def grab_hash_tags_or_mention(tweet, grab='#'):

    """

    Extract hashtags and mentions based on prefix symbols.

    """

    return [tag.lower() for tag in tweet.split() if tag.startswith(grab)]



puntuations=string.punctuation+'…'+'Ã¢â‚¬Â¦'



def remove_punct_end_of_tag(tag):

    """

    Remove punctuation at end of hashtag.

    """

    ortag = tag

    not_alpha=True

    tag_lenth = len(tag)

    if tag_lenth <= 2:

        return None

    elif (tag[-8:]=='#Ã¢â‚¬Â¦') or (tag[-8:]=='@Ã¢â‚¬Â¦'):

        return None

    for i in range(tag_lenth):

        if tag:

            if (tag[-1] in puntuations):

                tag = tag[:-1]

                not_alpha = tag[-1] in puntuations

            else:

                break         

    return tag
# Grab hashtags and mentions

df_train['hash_tags'] = df_train['message'].apply(grab_hash_tags_or_mention)

df_train['hash_tags'] = df_train['hash_tags'].apply(lambda tags: [remove_punct_end_of_tag(tag) for tag in tags if tag])



df_train['mentions'] = df_train['message'].apply(lambda tweet: grab_hash_tags_or_mention(tweet, grab='@'))

df_train['mentions'] = df_train['mentions'].apply(lambda tags: [remove_punct_end_of_tag(tag) for tag in tags if tag])
# Split hashtags based on sentiment class

HT_neg = df_train['hash_tags'][df_train['sentiment'] == -1]

HT_neutral = df_train['hash_tags'][df_train['sentiment'] == 0]

HT_pos = df_train['hash_tags'][df_train['sentiment'] == 1]

HT_news = df_train['hash_tags'][df_train['sentiment'] == 2]



HT_neg = sum(HT_neg, [])

HT_neutral = sum(HT_neutral, [])

HT_pos = sum(HT_pos, [])

HT_news = sum(HT_news, [])
# HT_neg

p = nltk.FreqDist(HT_neg)

d = pd.DataFrame({'Hashtag': list(p.keys()), 'Count': list(p.values())})

d = d.nlargest(columns = 'Count', n = 15)

plt.figure(figsize=(12, 6))

ax = sns.barplot(data=d, y ='Hashtag', x ='Count', palette='winter', orient='h')

plt.title('Negative Hashtags',fontsize=14)

plt.show()
# HT_neutral

p = nltk.FreqDist(HT_neutral)

d = pd.DataFrame({'Hashtag': list(p.keys()), 'Count': list(p.values())})

d = d.nlargest(columns = 'Count', n = 15)

plt.figure(figsize=(12, 6))

ax = sns.barplot(data=d, y ='Hashtag', x ='Count', palette='winter_r', orient='h')

plt.title('Neutral Hashtags',fontsize=14)

plt.show()
# HT_pos

p = nltk.FreqDist(HT_pos)

d = pd.DataFrame({'Hashtag': list(p.keys()), 'Count': list(p.values())})

d = d.nlargest(columns = 'Count', n = 15)

plt.figure(figsize=(12, 6))

ax = sns.barplot(data=d, y ='Hashtag', x ='Count', palette='winter', orient='h')

plt.title('Positive Hashtags',fontsize=14)

plt.show()
# HT_news

p = nltk.FreqDist(HT_news)

d = pd.DataFrame({'Hashtag': list(p.keys()), 'Count': list(p.values())})

d = d.nlargest(columns = 'Count', n = 15)

plt.figure(figsize=(12, 6))

ax = sns.barplot(data=d, y ='Hashtag', x ='Count', palette='winter_r', orient='h')

plt.title('News Related Hashtags',fontsize=14)

plt.show()
def mentions_or_hash_count_df(df, col = 'mentions'): 

    """

    Make count dataframes.

    """

    tags = []

    for i, row in df.iterrows():

        for tag in row[col]:

            if tag:

                tags.append(tag)

    counts = dict(Counter(tags)).items()

    return pd.DataFrame(counts, columns=[col, col+'_count']).sort_values(by=col+'_count', ascending=False)



mentions_counts = mentions_or_hash_count_df(df_train, col = 'mentions')

hash_tags = mentions_or_hash_count_df(df_train, col = 'hash_tags')
# Create tables for sentiment for Hashtag- and Mentions counts

df_sent_selctors = [df_train['sentiment']==sent for sent in [-1,0,1,2]]



# Create empty dataframe that will store the counts with sentiments

mentions_sents = pd.DataFrame(columns=['mentions', 'mentions_count', 'sentiment'])

hash_sents = pd.DataFrame(columns=['hash_tags', 'hash_tags_count', 'sentiment'])



# Add values to the empty dataframe

for i, selector in enumerate(df_sent_selctors): 

  # mentions

    mentions_counts = mentions_or_hash_count_df(df_train[selector], col = 'mentions')

    mentions_counts['sentiment'] = i-1

    mentions_sents=pd.concat([mentions_sents, mentions_counts], ignore_index=True, copy=False)

  

  # hashtags

    hash_counts = mentions_or_hash_count_df(df_train[selector], col = 'hash_tags')

    hash_counts['sentiment'] = i-1

    hash_sents = pd.concat([hash_sents, hash_counts], ignore_index=True, copy=False)
# Check Hashtag counts with Sentiments

b = hash_sents.drop(columns=['sentiment']).groupby('hash_tags').sum()

(b['hash_tags_count']).sort_values()

b = b[b['hash_tags_count']>20].sort_values(by= 'hash_tags_count', ascending=False)

print(b)

top_hash_tags = list(b.index)
# Check Mentions counts with sentiments

a = mentions_sents.drop(columns=['sentiment']).groupby('mentions').sum()

a = a[a['mentions_count']>40].sort_values(by= 'mentions_count', ascending=False)

print(a)

top_mentions = list(a.index)
# Top Mentions and Hashtags

conditions = [(mentions_sents['mentions']==mention) for mention in top_mentions]

selector = mentions_sents['mentions'] == ',,,,,,'    

for condition in conditions:

    selector = selector|condition



fig,axis = plt.subplots(figsize=(15,5))

sns.barplot(x='mentions', y='mentions_count', data=mentions_sents[selector], palette='winter', hue='sentiment')

plt.xticks(rotation=90)

plt.title('Number of Sentiment Classes Per Mention',fontsize=14)

plt.legend(title='Class',loc='upper right')

plt.ylabel('Count')

plt.xlabel('Mention')

plt.show()



conditions = [(hash_sents['hash_tags']==hashtag) for hashtag in top_hash_tags]

selector = hash_sents['hash_tags'] == ',,,,,,' 

for condition in conditions:

    selector = selector|condition

    

fig,axis = plt.subplots(figsize=(15,5))

sns.barplot(x='hash_tags', y ='hash_tags_count', data=hash_sents[selector], palette='winter', hue='sentiment')

plt.xticks(rotation=90)

plt.title('Number of Sentiment Classes Per Hashtag',fontsize=14)

plt.legend(title='Class',loc='upper right')

plt.ylabel('Count')

plt.xlabel('Hashtag')

plt.show()
# Class distribution for set of retweeted-tweets and set without retweets

plt.figure(figsize = (8,5))

df_train['is_retweet'] = df_train['message'].apply(lambda tweet: 1 if tweet.startswith('RT @') else 0)

sns.countplot(x='is_retweet', data=df_train, palette='winter', hue='sentiment')

plt.title('Number of Retweets Per Sentiment Class',fontsize=14)

plt.xlabel('Is Retweet')

plt.ylabel('Count')

plt.legend(title='Class')

plt.show()
# Top retweets per Sentiment

df_train1 = pd.read_csv('../input/climate-change-belief-analysis/train.csv')

def count_retweets(df):

    retweets = []

    for i, row in df.iterrows():

        if row['message'].startswith('RT @'):

            retweets.append(row['message'])

    counts = dict(Counter(retweets)).items()

    return pd.DataFrame(counts, columns=['retweeted', 'retweet_count']).sort_values(by='retweet_count', ascending=False)



df_sent_selctors = [df_train1['sentiment']==sent for sent in [-1,0,1,2]]

retweet_sents = pd.DataFrame(columns=['retweeted', 'retweet_count', 'sentiment'])



for i, selector in enumerate(df_sent_selctors):

    retweet_counts = count_retweets(df_train1[selector])

    retweet_counts['sentiment'] = i-1

    retweet_sents=pd.concat([retweet_sents, retweet_counts], ignore_index=True,copy=False)  

    

c = retweet_sents.drop(columns=['sentiment']).groupby('retweeted').sum()

c = c[c['retweet_count']>20].sort_values(by= 'retweet_count').index

top_retweets = list(c)



# Plot for all sents

conditions = [(retweet_sents['retweeted']==retweet) for retweet in top_retweets]

selector = retweet_sents['retweeted'] == ',,,,,,' 



for condition in conditions:

    selector = selector|condition



shortened_retweets = retweet_sents.copy()

shortened_retweets['retweeted'] = shortened_retweets['retweeted'].apply(lambda rt: rt[:20])

fig,axis = plt.subplots(figsize=(10,5))

sns.barplot(y='retweeted', x='retweet_count', data=shortened_retweets[selector], palette='winter', hue='sentiment', orient='h')

plt.title('Sentiment Class Per Retweet for Top Retweets',fontsize=14)

plt.xlabel('Count')

plt.ylabel('Retweeted')

plt.show()
# Retweet Proportions

df1_train['is_retweet'] = df1_train['message'].apply(lambda tweet: 1 if tweet.startswith('RT @') else 0)

fig, axis = plt.subplots(ncols=4, figsize=(25, 5))

fig.suptitle('Proportion of Tweets to Retweets Per Sentiment Class', fontsize=20)

group = ["(Anti)", "(Neutral)", "(Pro)", "(News)"]

for i in range(4):

    df1_train[(df1_train['sentiment']==i-1) & (df1_train['comp_score']=='negative')]['is_retweet'].value_counts().plot.pie(autopct='%1.1f%%',colormap='winter_r',ax=axis[i])

    axis[i].set_title('Negative Vader Sentiment '+ group[i])

    axis[i].set_ylabel('')

plt.show()



fig, axis = plt.subplots(ncols=4, figsize=(25, 5))

for i in range(4):

    df1_train[(df1_train['sentiment']==i-1) & (df1_train['comp_score']=='neutral')]['is_retweet'].value_counts().plot.pie(autopct='%1.1f%%',colormap='winter_r',ax=axis[i])

    axis[i].set_title('Neutral Vader Sentiment '+ group[i])

    axis[i].set_ylabel('')

plt.show()



fig, axis = plt.subplots(ncols=4, figsize=(25, 5))

for i in range(4):

    df1_train[(df1_train['sentiment']==i-1) & (df1_train['comp_score']=='positive')]['is_retweet'].value_counts().plot.pie(autopct='%1.1f%%',colormap='winter_r',ax=axis[i])

    axis[i].set_title('Positve Vader Sentiment '+ group[i])

    axis[i].set_ylabel('')

plt.show()
sid = SentimentIntensityAnalyzer() ## instantiate Sentiment analyzer



def sent_decider(compound):

    """

    Function to determine if sentiment is positive, nuetral or negative.

    """

    neutral_point = 0.00

    if compound > neutral_point:

        return 'positive'#1

    elif compound < -neutral_point:

        return 'negative' #-1

    else: 

        return 'neutral'#0



# Get sentiment

dropped_rows['scores'] = dropped_rows['message'].apply(lambda review: sid.polarity_scores(review))

dropped_rows['compound']  = dropped_rows['scores'].apply(lambda score_dict: score_dict['compound'])

dropped_rows['comp_score'] = dropped_rows['compound'].apply(sent_decider)



# Plots



fig, axis = plt.subplots(ncols=2, figsize=(20, 5))

counts = (dropped_rows.groupby(['comp_score'])['sentiment'].value_counts(normalize=True).rename('percentage_tweets').mul(100).reset_index())



ax = sns.barplot(x="comp_score", y="percentage_tweets", palette='winter', hue="sentiment", data=counts, ax=axis[0])

put_numbers_on_bars(ax)

axis[0].set_xlabel('Vader Sentiment')

axis[0].set_ylabel('Tweets (%)')

axis[0].set_title('Dropped Retweets Per Vader Sentiment',fontsize=14)



counts = (dropped_rows.groupby(['sentiment'])['comp_score'].value_counts(normalize=True).rename('percentage_tweets').mul(100).reset_index())



ax = sns.barplot(x="sentiment", y="percentage_tweets", palette='winter', hue="comp_score", data=counts, ax=axis[1])

put_numbers_on_bars(ax)

axis[1].set_xlabel('Sentiment Class')

axis[1].set_ylabel('Tweets (%)')

axis[1].set_title('Dropped Retweets Per Sentiment Class',fontsize=14)



fig, axis = plt.subplots(ncols=2, figsize=(20, 5))



ax = sns.countplot(x='comp_score', data=dropped_rows, palette='winter', hue='sentiment', ax=axis[0])

axis[0].set_title('Dropped Retweets Per Vader Sentiment',fontsize=14)

axis[0].set_xlabel('Vader Sentiment')

axis[0].set_ylabel('Tweets')

put_numbers_on_bars(ax)



ax = sns.countplot(x='sentiment', data=dropped_rows, palette='winter', hue='comp_score', ax=axis[1])

axis[1].set_title('Dropped Retweets Per Sentiment Class',fontsize=14)

axis[1].set_xlabel('Sentiment Class')

axis[1].set_ylabel('Tweets')

put_numbers_on_bars(ax)



plt.show()



# Plot Pie chart for duplicate retweets's Vader sentiment share and Class share

fig, axis = plt.subplots(ncols=3, figsize=(20, 5))

axis = axis.flat



(dropped_rows['comp_score'].value_counts().plot.pie(autopct='%1.1f%%', colormap='winter_r', ax=axis[0]))

axis[0].set_title('Proportion of Dropped Retweets Per Vader Sentiment',fontsize=14)

axis[0].set_ylabel('')



(dropped_rows['sentiment'].value_counts().plot.pie(autopct='%1.1f%%', colormap='winter_r', ax=axis[1]))

axis[1].set_title('Proportion of Dropped Retweets Per Sentiment Class',fontsize=14)

axis[1].set_ylabel('')



(dropped_rows['comp_score'].value_counts().plot.pie(autopct='%1.1f%%', colormap='winter_r', ax=axis[2]))

axis[2].set_title('Proportion of Pro-Climate Change \n Dropped Retweets Per Vader Sentiment',fontsize=14)

axis[2].set_ylabel('')

plt.show()
# Analyse dropped rows (retweets)

dropped_rows['is_retweet'] = dropped_rows['message'].apply(lambda tweet: 1 if tweet.startswith('RT @') else 0)

dropped_rows = dropped_rows[dropped_rows['is_retweet']==1]



fig, axis = plt.subplots(ncols=2, figsize=(15, 5))



ax = sns.countplot(x="sentiment", palette='winter', data=dropped_rows, ax=axis[0])

put_numbers_on_bars(ax)

axis[0].set_xlabel('Sentiment Class')

axis[0].set_ylabel('Tweets')

axis[0].set_title('Dropped Retweets Per Sentiment Class',fontsize=14)





ax = sns.countplot(x="sentiment", palette='winter', data=df1_train[df1_train['is_retweet']==1], ax=axis[1])

put_numbers_on_bars(ax)

axis[1].set_xlabel('Sentiment Class')

axis[1].set_ylabel('Tweets')

axis[1].set_title('Retweets Per Sentiment Class',fontsize=14)

plt.show()



# As percentage

fig, axis = plt.subplots(ncols=2, figsize=(15, 5))



# Count retweets in duplicate retweets and complete orignal data

counts_original = (df1_train['sentiment'].value_counts())

counts_retweets = (dropped_rows['sentiment'].value_counts())



counts_retweets_fraction = ((counts_retweets/counts_original*100).rename_axis('sentiment').reset_index(name='Number of Tweets'))



ax = sns.barplot(x='sentiment', y='Number of Tweets', data=counts_retweets_fraction, palette='winter', ax=axis[0])

ax.set_title('Dropped Retweets as Fraction of Total All Tweets',fontsize=14)

ax.set_xlabel('Sentiment Class')

ax.set_ylabel("Retweets (%)")

ax.set_ylim(ymax=33)

put_numbers_on_bars(ax)



# Count retweets in orignal data

counts_original = (df1_train[df1_train['is_retweet']==1]['sentiment'].value_counts())

counts_retweets = (dropped_rows['sentiment'].value_counts())



counts_retweets_fraction = ((counts_retweets/counts_original*100).rename_axis('sentiment').reset_index(name='Number of Tweets'))



ax = sns.barplot(x='sentiment', y='Number of Tweets', data=counts_retweets_fraction, palette='winter', ax=axis[1])

ax.set_title('Dropped Retweets as Fraction of Retweets',fontsize=14)

ax.set_xlabel('Sentiment Class')

ax.set_ylabel("Retweets (%)")

ax.set_ylim(ymax=33)

put_numbers_on_bars(ax)

plt.show()
# Remove words used to scrape tweets

# all_classes = ['globalwarming','global','warming','climatechange', 'climate', 'change']

# train_data['final'] = train_data['final'].apply(lambda x : [i for i in x if i not in all_classes])

# test_data['final'] = test_data['final'].apply(lambda x : [i for i in x if i not in all_classes])
# train_data['final'] = train_data['final'].replace(to_replace = r"[\d-]", value = '', regex = True)

# train_data['final'] = train_data['final'].replace(to_replace = "...", value = '')

# train_data['final'] = train_data['final'].replace(to_replace = "..", value = '')
# df_pro = train_data[train_data['sentiment']==1]

# df_news =  train_data[train_data['sentiment']==2]

# df_neutral = train_data[train_data['sentiment']==0]

# df_anti = train_data[train_data['sentiment']==-1]

# train_data['sentiment'].value_counts()
# class_size = 3640 # Class size = can be roughly half the size of the largest size

# def resampling(df):

#     if len(df['sentiment']) > class_size:

#         df = resample(df, replace=False, n_samples=class_size, random_state=42)

#     else:

#         df = resample(df, replace=True, n_samples=class_size, random_state=42)

#     return df

# df_pro_resampled = resampling(df_pro)

# df_news_resampled =  resampling(df_news)

# df_neutral_resampled = resampling(df_neutral)

# df_anti_resampled = resampling(df_anti)

# train_data = pd.concat([df_pro_resampled,df_news_resampled,df_neutral_resampled,df_anti_resampled])

# train_data['sentiment'].value_counts()
def get_part_of_speech(word):

    """

    Find part of speech of word if part of speech is either noun, verb, adjective etc and add it to a list.

    """

    probable_part_of_speech = wordnet.synsets(word) ## finding word that is most similar (synonyms) for semantic reasoning

    pos_counts = Counter() ## instantiating our counter class

    pos_counts["n"] = len([i for i in probable_part_of_speech if i.pos()=="n"])

    pos_counts["v"] = len([i for i in probable_part_of_speech if i.pos()=="v"])

    pos_counts["a"] = len([i for i in probable_part_of_speech if i.pos()=="a"])

    pos_counts["r"] = len([i for i in probable_part_of_speech if i.pos()=="r"])

    most_likely_part_of_speech = pos_counts.most_common(1)[0][0] ## will extract the most likely part of speech from the list

    return most_likely_part_of_speech



normalizer = WordNetLemmatizer()



train_data['final'] = train_data['final'].apply(lambda x: [normalizer.lemmatize(token, get_part_of_speech(token)) for token in x])

test_data['final'] = test_data['final'].apply(lambda x: [normalizer.lemmatize(token, get_part_of_speech(token)) for token in x])
X = train_data['final']

y = train_data['sentiment']

X_test = test_data['final']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state = 42)
X_train = list(X_train.apply(' '.join))

X_val = list(X_val.apply(' '.join))



vectorizer = TfidfVectorizer(sublinear_tf=True, smooth_idf = True, max_df = 0.3, min_df = 5, ngram_range = (1, 2))

vectorizer.fit(X_train)



# vect_save_path = "TfidfVectorizer.pkl"

# with open(vect_save_path,'wb') as file:

#     pickle.dump(vectorizer,file)



X_train = vectorizer.transform(X_train)

X_val = vectorizer.transform(X_val)
modelstart = time.time()

logreg = LogisticRegression(C=1000, multi_class='ovr', solver='saga', random_state=42, max_iter=10)

logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_val)

logreg_f1 = round(f1_score(y_val, y_pred, average='weighted'),2)

print('Accuracy %s' % accuracy_score(y_pred, y_val))

print("Model Runtime: %0.2f seconds"%((time.time() - modelstart)))

report = classification_report(y_val, y_pred, output_dict=True)

results = pd.DataFrame(report).transpose()

# results.to_csv("logreg_report.csv")

results

# model_save_path = "logreg_model.pkl"

# with open(model_save_path,'wb') as file:

#     pickle.dump(logreg,file)
plot_confusion_matrix(y_val, y_pred, normalize=True,figsize=(8,8),cmap='winter_r')

plt.show()
modelstart= time.time()

multinb = MultinomialNB()

multinb.fit(X_train, y_train)

y_pred = multinb.predict(X_val)

multinb_f1 = round(f1_score(y_val, y_pred, average='weighted'),2)

print('Accuracy %s' % accuracy_score(y_pred, y_val))

print("Model Runtime: %0.2f seconds"%((time.time() - modelstart)))

report = classification_report(y_val, y_pred, output_dict=True)

results = pd.DataFrame(report).transpose()

# results.to_csv("multinb_report.csv")

results

# model_save_path = "multinb_model.pkl"

# with open(model_save_path,'wb') as file:

#     pickle.dump(multinb,file)
plot_confusion_matrix(y_val, y_pred, normalize=True,figsize=(8,8),cmap='winter_r')

plt.show()
modelstart = time.time()

rf = RandomForestClassifier(max_features=4, random_state=42)

rf.fit(X_train, y_train)

y_pred = rf.predict(X_val)

rf_f1 = round(f1_score(y_val, y_pred, average='weighted'),2)

print('Accuracy %s' % accuracy_score(y_pred, y_val))

print("Model Runtime: %0.2f seconds"%((time.time() - modelstart)))

report = classification_report(y_val, y_pred, output_dict=True)

pd.DataFrame(report).transpose()
plot_confusion_matrix(y_val, y_pred, normalize=True,figsize=(8,8),cmap='winter_r')

plt.show()
modelstart = time.time()

svc = SVC(gamma = 0.8, C = 10, random_state=42)

svc.fit(X_train, y_train)

y_pred = svc.predict(X_val)

svc_f1 = round(f1_score(y_val, y_pred, average='weighted'),2)

print('Accuracy %s' % accuracy_score(y_pred, y_val))

print("Model Runtime: %0.2f seconds"%((time.time() - modelstart)))

report = classification_report(y_val, y_pred, output_dict=True)

results = pd.DataFrame(report).transpose()

# results.to_csv("svc_report.csv")

results

# model_save_path = "svc_model.pkl"

# with open(model_save_path,'wb') as file:

#     pickle.dump(svc,file)
plot_confusion_matrix(y_val, y_pred, normalize=True,figsize=(8,8),cmap='winter_r')

plt.show()
modelstart = time.time() 

linsvc = LinearSVC()

linsvc.fit(X_train, y_train)

y_pred = linsvc.predict(X_val)

linsvc_f1 = round(f1_score(y_val, y_pred, average='weighted'),2)

print('Accuracy %s' % accuracy_score(y_pred, y_val))

print("Model Runtime: %0.2f seconds"%((time.time() - modelstart)))

report = classification_report(y_val, y_pred, output_dict=True)

results = pd.DataFrame(report).transpose()

# results.to_csv("linsvc_report.csv")

results

# model_save_path = "linsvc_model.pkl"

# with open(model_save_path,'wb') as file:

#     pickle.dump(linsvc,file)
plot_confusion_matrix(y_val, y_pred, normalize=True,figsize=(8,8),cmap='winter_r')

plt.show()
modelstart = time.time()

kn = KNeighborsClassifier(n_neighbors=1)

kn.fit(X_train, y_train)

y_pred = kn.predict(X_val)

kn_f1 = round(f1_score(y_val, y_pred, average='weighted'),2)

print('accuracy %s' % accuracy_score(y_pred, y_val))

print("Model Runtime: %0.2f seconds"%((time.time() - modelstart)))

report = classification_report(y_val, y_pred, output_dict=True)

results = pd.DataFrame(report).transpose()

# results.to_csv("kn_report.csv")

results

# model_save_path = "kn_model.pkl"

# with open(model_save_path,'wb') as file:

#     pickle.dump(kn,file)
plot_confusion_matrix(y_val, y_pred, normalize=True,figsize=(8,8),cmap='winter_r')

plt.show()
modelstart = time.time()

dt = DecisionTreeClassifier(random_state=42)    

dt.fit(X_train, y_train)

y_pred = dt.predict(X_val)

dt_f1 = round(f1_score(y_val, y_pred, average='weighted'),2)

print('accuracy %s' % accuracy_score(y_pred, y_val))

print("Model Runtime: %0.2f seconds"%((time.time() - modelstart)))

report = classification_report(y_val, y_pred, output_dict=True)

pd.DataFrame(report).transpose()
plot_confusion_matrix(y_val, y_pred, normalize=True,figsize=(8,8),cmap='winter_r')

plt.show()
modelstart = time.time()

ad = AdaBoostClassifier(random_state=42)

ad.fit(X_train, y_train)

y_pred = ad.predict(X_val)

ad_f1 = round(f1_score(y_val, y_pred, average='weighted'),2)

print('accuracy %s' % accuracy_score(y_pred, y_val))

print("Model Runtime: %0.2f seconds"%((time.time() - modelstart)))

report = classification_report(y_val, y_pred, output_dict=True)

pd.DataFrame(report).transpose()
plot_confusion_matrix(y_val, y_pred, normalize=True,figsize=(8,8),cmap='winter_r')

plt.show()
# Compare Weighted F1-Scores Between Models

fig,axis = plt.subplots(figsize=(10, 5))

rmse_x = ['Multinomial Naive Bayes','Logistic Regression','Random Forest Classifier','Support Vector Classifier','Linear SVC','K Neighbours Classifier','Decision Tree Classifier','AdaBoost Classifier']

rmse_y = [multinb_f1,logreg_f1,rf_f1,svc_f1,linsvc_f1,kn_f1,dt_f1,ad_f1]

ax = sns.barplot(x=rmse_x, y=rmse_y,palette='winter')

plt.title('Weighted F1-Score Per Classification Model',fontsize=14)

plt.xticks(rotation=90)

plt.ylabel('Weighted F1-Score')

for p in ax.patches:

    ax.text(p.get_x() + p.get_width()/2, p.get_y() + p.get_height(), round(p.get_height(),2), fontsize=12, ha="center", va='bottom')

    

plt.show()
LogisticRegression().get_params()
param_grid = {'C': [1000], #[100,1000]

              'max_iter': [10], #[10,100]

              'multi_class': ['ovr'], #['ovr', 'multinomial']

              'random_state': [42],

              'solver': ['saga']} #['saga','lbfgs']

grid_LR = GridSearchCV(LogisticRegression(), param_grid, scoring='f1_weighted', cv=5, n_jobs=-1)

grid_LR.fit(X_train, y_train)

y_pred = grid_LR.predict(X_val)

print("Best parameters:")

print(grid_LR.best_params_)

print('accuracy %s' % accuracy_score(y_pred, y_val))

print(classification_report(y_val, y_pred))
LinearSVC().get_params()
param_grid = {'C': [100],#[0.1,1,10,100,1000]

              'max_iter': [10], #[10,100]

              'multi_class' : ['ovr'], #['crammer_singer', 'ovr']

              'random_state': [42]} 

grid_LSVC = GridSearchCV(LinearSVC(), param_grid, scoring='f1_weighted', cv=5, n_jobs=-1)

grid_LSVC.fit(X_train, y_train)

y_pred = grid_LSVC.predict(X_val)

print(grid_LSVC.best_params_)

print('accuracy %s' % accuracy_score(y_pred, y_val))

print(classification_report(y_val, y_pred))
SVC().get_params()
param_grid = {'C': [10],#[0.1,1,10,100,1000]

              'gamma': [0.8], #[0.8,1]

              'kernel': ['rbf'], #['linear','rbf']

              'random_state': [42]} 

grid_SVC = GridSearchCV(SVC(), param_grid, scoring='f1_weighted', cv=5, n_jobs=-1)

grid_SVC.fit(X_train, y_train)

y_pred = grid_SVC.predict(X_val)

print(grid_SVC.best_params_)

print('accuracy %s' % accuracy_score(y_pred, y_val))

print(classification_report(y_val, y_pred))
y_pred = svc.predict(X_val)

print('accuracy %s' % accuracy_score(y_pred, y_val))

print(classification_report(y_val, y_pred))
con_mat = plot_confusion_matrix(y_val, y_pred, normalize=True,figsize=(8,8),cmap='winter_r')

plt.show()
# If model doesn't take "probability=True" as an argument (e.g. LinearSVC)

ovr =  OneVsRestClassifier(SVC(random_state=42,class_weight='balanced'))



y_train_binarized = label_binarize(y_train, classes=[-1, 0, 1, 2])

y_val_binarized = label_binarize(y_val, classes=[-1, 0, 1, 2])

n_classes = 4



ovr.fit(X_train, y_train_binarized)



# decision_function predicts a “soft” score for each sample in relation to each class, 

# rather than the “hard” categorical prediction produced by predict. Its input is 

# usually only some observed data, X.

y_probas = ovr.decision_function(X_val)



plot_roc(y_val, y_probas,figsize=(8,8),cmap='winter_r')

plt.show()
# Make prediction on test data

X = train_data['final']

y = train_data['sentiment']

X_test = test_data['final']



X = list(X.apply(' '.join))

X_test = list(X_test.apply(' '.join))



vectorizer = TfidfVectorizer(sublinear_tf=True, smooth_idf = True, max_df = 0.3, min_df = 5, ngram_range = (1, 2))

vectorizer.fit(X)



X = vectorizer.transform(X)

X_test = vectorizer.transform(X_test)



svc = SVC(gamma=0.8, C=10, random_state=42)

svc.fit(X, y)

y_test = svc.predict(X_test)
# Number of Tweets Per Sentiment Class

fig, axis = plt.subplots(ncols=2, figsize=(15, 5))



ax = sns.countplot(y_test,palette='winter',ax=axis[0])

axis[0].set_title('Number of Tweets Per Sentiment Class',fontsize=14)

axis[0].set_xlabel('Sentiment Class')

axis[0].set_ylabel('Tweets')

for p in ax.patches:

    ax.text(p.get_x() + p.get_width()/2., p.get_height(), '%d' % int(p.get_height()), fontsize=11, ha='center', va='bottom')



results = pd.DataFrame({"tweetid":test_data['tweetid'],"sentiment": y_test})

results['sentiment'].value_counts().plot.pie(autopct='%1.1f%%',colormap='winter_r',ax=axis[1])

axis[1].set_title('Proportion of Tweets Per Sentiment Class',fontsize=14)

axis[1].set_ylabel('Sentiment Class')

    

plt.show()
# Create Kaggle Submission File

results = pd.DataFrame({"tweetid":test_data['tweetid'],"sentiment": y_test})

results.to_csv("Team_RM5_final_submission.csv", index=False)
## Create Comet logs

# print("\nResults\nConfusion matrix \n {}".format(confusion_matrix(y_val, y_pred)))

# f1 = f1_score(y_val, y_pred,average="macro")

# precision = precision_score(y_val, y_pred,average="macro")

# recall = recall_score(y_val, y_pred,average="macro")

# params = {"random_state": 42,

#           "model_type": "svc",

#           "param_grid": "str(param_grid)",

#           "stratify": True,

#           }

# metrics = {"f1": f1,

#            "recall": recall,

#            "precision": precision

#            }



# experiment.log_parameters(params)

# experiment.log_metrics(metrics)

# experiment.end()
## Display Comet experiment

# experiment.display()