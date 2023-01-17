import re

import pickle

import numpy as np

import pandas as pd

from collections import Counter



import matplotlib.pyplot as plt

from matplotlib import ticker

import seaborn as sns

from wordcloud import WordCloud



import nltk 

nltk.download('stopwords')

from nltk.corpus import stopwords

from nltk.stem import SnowballStemmer
data = pd.read_csv('../input/tweets-on-sushant-singh-rajput/SSR_tweets.csv')

data.head()
from nltk.stem import WordNetLemmatizer

def preprocess(textdata):

    processedText = []

    

    # Create Lemmatizer and Stemmer.

    wordLemm = WordNetLemmatizer()

    

    # Defining regex patterns.

    userPattern       = '@[^\s]+'

    alphaPattern      = "[^a-zA-Z0-9]"

    sequencePattern   = r"(.)\1\1+"

    seqReplacePattern = r"\1\1"

    

    for tweet in textdata:

        tweet = tweet.lower()

        

     

        # Replace @USERNAME to ' '.

        tweet = re.sub(userPattern,' ', tweet)        

        # Replace all non alphabets.

        tweet = re.sub(alphaPattern, " ", tweet)

        # Replace 3 or more consecutive letters by 2 letter.

        tweet = re.sub(sequencePattern, seqReplacePattern, tweet)



        tweetwords = ''

        for word in tweet.split():

            # Checking if the word is a stopword.

            #if word not in stopwordlist:

            if len(word)>1:

                # Lemmatizing the word.

                word = wordLemm.lemmatize(word)

                tweetwords += (word+' ')

            

        processedText.append(tweetwords)

        

    return processedText
text = data['text']
import time

t = time.time()

corpus = preprocess(text)

print(f'Text Preprocessing complete.')

print(f'Time Taken: {round(time.time()-t)} seconds')
df = pd.Series(corpus)

df.head()
word_list = [word for line in df for word in line.split()]



sns.set(style="darkgrid")

counts = Counter(word_list).most_common(50)

counts_df = pd.DataFrame(counts)

counts_df

counts_df.columns = ['word', 'frequency']



fig, ax = plt.subplots(figsize = (12, 12))

ax = sns.barplot(y="word", x='frequency', ax = ax, data=counts_df)

plt.savefig('wordcount_bar.png')
wordcloud = WordCloud(

    background_color='black',

    max_words=50,

    max_font_size=40, 

    scale=5,

    random_state=1,

    collocations=False,

    normalize_plurals=False

).generate(' '.join(word_list))





plt.figure(figsize = (12, 10), facecolor = None)

plt.imshow(wordcloud)

plt.axis("off")

plt.tight_layout(pad = 0)