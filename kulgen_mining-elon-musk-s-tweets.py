# Import Modules



import pandas as pd

import re, string

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from nltk.stem import PorterStemmer

from nltk.tokenize import TweetTokenizer

from nltk.corpus import stopwords

import networkx as nx

import re, string

import nltk

from collections import Counter

from wordcloud import WordCloud, STOPWORDS

from scipy.misc import imread

from subprocess import check_output

import warnings

warnings.filterwarnings('ignore')

# Inputfile



def load_tweets(tweet_file):



    """ Load and process a Twitter analytics data file """

    # Read tweet data (obtained from Twitter Analytics)

    tweet_df = pd.read_csv(tweet_file,sep = ',', low_memory = False,encoding="L5")

    return tweet_df



tweet_df = load_tweets('../input/dataelonmuskstweet/data_elonmusk.csv')

#-----------------------------------------------------------------#



df = pd.DataFrame(tweet_df)

df['Time'] = pd.to_datetime(df['Time'])

df['Time'] = pd.to_datetime(df['Time'], format='%y-%m-%d %H:%M:%S')

df['Time'].hist(label="Frequency",alpha=0.7)

plt.legend()

plt.title("Tweet Activty Over The Years")

plt.show()

#-----------------------------------------------------------------#



df['Retweet from'].value_counts(dropna=True)[:5].plot(kind='bar')

plt.title('Retweet from')

plt.tight_layout()

plt.show()

# Total tweets

print('Total tweets this period:', len(tweet_df.index), '\n')

df = df.drop('row ID',1)

df = df.drop('User',1)

tweet = df['Tweet'].tolist()



clean = [i.replace('+', ' ').replace('.', ' ').replace(',', ' ').replace(':', ' ').replace('(',' ').replace(')',' ').replace('\n',' ').replace('http',' ') for i in tweet]

clean = re.sub("(^|\W)\d+($|\W)", " ", str(clean)).lower()

clean = re.sub(r'https?:\/\/.*\/\w*',' ',clean) # Remove hyperlinks

clean = re.sub(r'['+string.punctuation+']+', ' ',clean) # Remove puncutations like 's





stop_words = set(stopwords.words('english'))

text_=[]

for w in clean.split():

    if w not in stop_words and len(w) > 2:

        ps = PorterStemmer()

        text_.append(ps.stem(w)) # stemming words

#print(text_)

def word_list(text_):

    wordfreq = {}

    for word in text_:

        if word in wordfreq.keys():

            wordfreq[word] += 1

        else:

            wordfreq[word] = 1

    return wordfreq



wordfreq = word_list(text_)



rslt = pd.DataFrame(Counter(wordfreq).most_common(15),columns=['Word', 'Frequency']).set_index('Word')

print('All frequencies, including STOPWORDS:')

print('=' * 60)

print(rslt)

print('=' * 60)



ax1 = rslt.plot.bar(rot=0,  width=0.8)

plt.xticks(fontsize=9,rotation=25)

plt.title('All frequencies, including stopwords')

plt.tight_layout()

plt.show()

# Create bigrams

bgs2 = nltk.bigrams(text_)



# compute frequency distribution for all the bigrams 

fdist2 = nltk.FreqDist(bgs2)



# for k,v in sorted(fdist2.items(),key = operator.itemgetter(1)):

#    print(k,v)



rslt2 = pd.DataFrame(Counter(fdist2).most_common(10),columns=['Word', 'Frequency']).set_index('Word')

print('All frequencies, including STOPWORDS:')

print('=' * 60)

print(rslt2)

print('=' * 60)



ax2 = rslt2.plot.bar(rot=0,  width=0.8)

plt.xticks(fontsize=9, rotation=35)

plt.tight_layout()

plt.show()
#----------------------------------------------------------------#



logo_mask = imread('../input/tweet-mask/tweet_mask.png', flatten=True)



# Generate a word cloud image

wordcloud = WordCloud().generate(clean)



# adding movie script specific stopwords

stopwords = set(STOPWORDS)

stopwords.add("co")

stopwords.add("rt")

stopwords.add("dr")

stopwords.add("et")

stopwords.add("will")

stopwords.add("re")



# lower max_font_size

wordcloud = WordCloud(max_font_size=40, stopwords=stopwords, background_color='black',

                      width=1800,

                      height=1400,

                      mask=logo_mask, normalize_plurals=bool).generate(clean)



plt.figure()

plt.imshow(wordcloud,cmap=plt.cm.gray)

plt.axis("off")

plt.show()