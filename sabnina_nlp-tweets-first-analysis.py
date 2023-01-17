import pandas as pd

import numpy as np
tweet_disaster=pd.read_csv("../input/nlp-getting-started/train.csv")

test = pd.read_csv("../input/nlp-getting-started/test.csv")
tweet_disaster.head()
tweet_disaster.info()
#tweet Sentiment Analysis

text=tweet_disaster['text']
text1=text.values.tolist()

#for tw in text1:

    #print (tw.split(' '))
#https://www.earthdatascience.org/courses/earth-analytics-python/using-apis-natural-language-processing-twitter/analyze-tweet-sentiments-in-python/

from textblob import TextBlob

# Create textblob objects of the tweets

sentiment_objects = [TextBlob(tw) for tw in text1]

# Create list of polarity valuesx and tweet text

sentiment_values = [[tweet.sentiment.polarity] for tweet in sentiment_objects]



#polarity values that range from 1 to -1.

#Values closer to 1 indicate more positivity, while values closer to -1 indicate more negativity.

sentiment_df = pd.DataFrame(sentiment_values, columns=["polarity"])

sentiment_df.head(10)
def add_target1(_df):

    _df = pd.concat([_df, sentiment_df], axis=1)

    return _df
tweet_disaster=add_target1(tweet_disaster)
tweet_disaster.head(100)
#del tweet_disaster['polarity']

#Analyze Sentiments Using Twitter Data

# Remove polarity values equal to zero

import matplotlib.pyplot as plt

sentiment_df = sentiment_df[sentiment_df.polarity != 0]

fig, ax = plt.subplots(figsize=(8, 6))

# Plot histogram with break at zero

sentiment_df.hist(bins=[-1, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1],

             ax=ax,

             color="purple")



plt.title("Real or Not? Disaster Tweets/Polarity ")

plt.show()
import nltk

nltk.download('stopwords')

nltk.download('punkt')

  
from nltk.corpus import stopwords

from nltk.tokenize import word_tokenize 

from nltk.stem import PorterStemmer

from textblob import Word 
#NLP tweets: Cleaning & Preprocessing tweets Data

stop_words = set(stopwords.words('english'))

tweet_1=[]

for tw in text1:

    word_tokens = word_tokenize(tw) 

    #Delete ponctuation

    word_tokens=[word.lower() for word in word_tokens if word.isalpha()]

    #Delete stop words

    filtered_sentence = [w for w in word_tokens if not w in stop_words] 

    filtered_sentence = [] 

  

    for w in word_tokens: 

        if w not in stop_words : 

            if  w!='http':

                filtered_sentence.append(w) 

               

    #print(word_tokens) 

    #print(filtered_sentence) 

    

    Stem_words = []

    ps =PorterStemmer()

    for w in filtered_sentence:

        rootWord=ps.stem(w)

        Stem_words.append(rootWord)

    #print(filtered_sentence)

    #print(Stem_words)

    #https://www.analyticsvidhya.com/blog/2019/08/how-to-remove-stopwords-text-normalization-nltk-spacy-gensim-python/

    lem=[]

    for w in filtered_sentence:

        word1 = Word(w).lemmatize("n")

        word2 = Word(word1).lemmatize("v")

        word3 = Word(word2).lemmatize("a")

        lem.append(Word(word3).lemmatize())

    tweet_1.append(lem)

sentiment_objects = [TextBlob(str(t)) for t in tweet_1]

sentiment_values = [[tweet_1.sentiment.polarity, str(tweet_1)] for tweet_1 in sentiment_objects]

sentiment_values[0]

sentiment_df1 = pd.DataFrame(sentiment_values, columns=["polarity_nlp", "tweet_nlp"])

sentiment_df1.head(10)

def add_target1(_df):

    _df = pd.concat([_df, sentiment_df1], axis=1)

    return _df
tweet_disaster=add_target1(tweet_disaster)
tweet_disaster.head(100)
#Analyze Sentiments Using NLP Twitter Data

sentiment_df1 = sentiment_df1[sentiment_df1.polarity_nlp != 0]

fig, ax = plt.subplots(figsize=(8, 6))

sentiment_df1.hist(bins=[-1, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1],

             ax=ax,

             color="purple")



plt.title("Real or Not? Disaster NLP Tweets/Polarity  ")

plt.show()
