import nltk                             

from nltk.corpus import twitter_samples   

import matplotlib.pyplot as plt           

import random                              
nltk.download('twitter_samples')
all_positive_tweets = twitter_samples.strings('positive_tweets.json')

all_negative_tweets = twitter_samples.strings('negative_tweets.json')



print('Number of positive tweets: ', len(all_positive_tweets))

print('Number of negative tweets: ', len(all_negative_tweets))



print('\nThe type of all_positive_tweets is: ', type(all_positive_tweets))

print('The type of a tweet entry is: ', type(all_negative_tweets[0]))
all_positive_tweets[:20]
all_negative_tweets[:20]
total_positive_words = []

for sentence in all_positive_tweets:

    total_positive_words.append(sentence.count(' '))

    

total_negative_words = []

for sentence in all_negative_tweets:

    total_negative_words.append(sentence.count(' '))

    

import plotly.graph_objects as go

import numpy as np



x0 = np.array(total_positive_words)

x1 = np.array(total_negative_words)



fig = go.Figure()

fig.add_trace(go.Histogram(x=x1, name = 'Negative'))

fig.add_trace(go.Histogram(x=x0, name = 'Positive'))



# Overlay both histograms

fig.update_layout(barmode='overlay')

# Reduce opacity to see both histograms

fig.update_traces(opacity=0.75)

fig.show()
tweet = all_positive_tweets[1455]

print(tweet)
nltk.download('stopwords')



import re                                  

import string                             

from nltk.corpus import stopwords 

from nltk.stem import PorterStemmer

from nltk.tokenize import TweetTokenizer  
print('Original Tweet: ')

print(tweet)



# it will remove the old style retweet text "RT"

tweet2 = re.sub(r'^RT[\s]+', '', tweet)



# it will remove hyperlinks

tweet2 = re.sub(r'https?:\/\/.*[\r\n]*', '', tweet2)



# it will remove hashtags. We have to be careful here not to remove 

# the whole hashtag because text of hashtags contains huge information. 

# only removing the hash # sign from the word

tweet2 = re.sub(r'#', '', tweet2)



# it will remove single numeric terms in the tweet. 

tweet2 = re.sub(r'[0-9]', '', tweet2)

print('\nAfter removing old style tweet, hyperlinks and # sign')

print(tweet2)
print('Before Tokenizing: ')

print(tweet2)



# instantiate the tokenizer class

tokenizer = TweetTokenizer(preserve_case=False, 

                           strip_handles=True,

                           reduce_len=True)



# tokenize the tweets

tweet_tokens = tokenizer.tokenize(tweet2)



print('\nTokenized string:')

print(tweet_tokens)
#Import the english stop words list from NLTK

stopwords_english = stopwords.words('english') 



print('Stop words\n')

print(stopwords_english)



print('\nPunctuation\n')

print(string.punctuation)
print('Before tokenization')

print(tweet_tokens)





tweets_clean = []



for word in tweet_tokens: # Go through every word in your tokens list

    if (word not in stopwords_english and  # remove stopwords

        word not in string.punctuation):  # remove punctuation

        tweets_clean.append(word)



print('\n\nAfter removing stop words and punctuation:')

print(tweets_clean)
# Instantiate stemming class

stemmer = PorterStemmer() 



# Create an empty list to store the stems

tweets_stem = [] 



for word in tweets_clean:

    stem_word = stemmer.stem(word)  # stemming word

    tweets_stem.append(stem_word)  # append to the list



print('Words after stemming: ')

print(tweets_stem)