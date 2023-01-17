!pip install tweepy

import tweepy 
import numpy as np
import pandas as pd

# Twitter Developer tokens and keys here
# It is CENSORED
consumer_key = 'Rxk3CrnZEBNp9rnKVrLEtgb0n'
consumer_secret = 'fS246Gu7fd0A6ngeo1ZqCNKH3dH3Fg3vM32HVMAYr9cIi4BP00'
access_token = '1283244120127889408-k33Iv4YI5d53NrA7qTAKtGW1iWtKk4'
access_token_secret = '9vELsHFineQO3llVaq89oFI2gSf5xiyJwXrjMXjWL8r0i'
# The following codes were from the following website: https://www.kaggle.com/amar09/sentiment-analysis-on-scrapped-tweets#3.-Text-Pre-processing

class TwitterClient(object): 
    def __init__(self): 
        # Initialization method. 
        try: 
            # Create OAuthHandler object 
            auth = tweepy.OAuthHandler(consumer_key, consumer_secret) 
            # Set access token and secret 
            auth.set_access_token(access_token, access_token_secret) 
            # Create tweepy API object to fetch tweets 
            # Add hyper parameter 'proxy' if executing from behind proxy "proxy='http://172.22.218.218:8085'"
            self.api = tweepy.API(auth, wait_on_rate_limit = True, wait_on_rate_limit_notify = True)
            
        except tweepy.TweepError as e:
            print(f"Error: Tweeter Authentication Failed - \n{str(e)}")

    def get_tweets(self, query, maxTweets = 1000):
        # Function to fetch tweets. 
        # Empty list to store parsed tweets 
        tweets = [] 
        sinceId = None
        max_id = -1
        tweetCount = 0
        tweetsPerQry = 100

        while tweetCount < maxTweets:
            try:
                if (max_id <= 0):
                    if (not sinceId):
                        new_tweets = self.api.search(q = query, count = tweetsPerQry)
                    else:
                        new_tweets = self.api.search(q = query, count = tweetsPerQry,
                                                since_id = sinceId)
                else:
                    if (not sinceId):
                        new_tweets = self.api.search(q = query, count = tweetsPerQry,
                                                max_id = str(max_id - 1))
                    else:
                        new_tweets = self.api.search(q = query, count = tweetsPerQry,
                                                max_id = str(max_id - 1),
                                                since_id = sinceId)
                if not new_tweets:
                    print("No more tweets found")
                    break

                for tweet in new_tweets:
                    parsed_tweet = {} 
                    parsed_tweet['tweets'] = tweet.text 

                    # Appending parsed tweet to tweets list 
                    if tweet.retweet_count > 0: 
                        # If tweet has retweets, ensure that it is appended only once 
                        if parsed_tweet not in tweets: 
                            tweets.append(parsed_tweet) 
                    else: 
                        tweets.append(parsed_tweet) 
                        
                tweetCount += len(new_tweets)
                print("Downloaded {0} tweets".format(tweetCount))
                max_id = new_tweets[-1].id

            except tweepy.TweepError as e:
                # Just exit if any error
                print("Tweepy error : " + str(e))
                break
        
        return pd.DataFrame(tweets)
# The following codes were from the following website: https://www.kaggle.com/amar09/sentiment-analysis-on-scrapped-tweets#3.-Text-Pre-processing

twitter_client = TwitterClient()

# Calling function to get tweets
tweets_df = twitter_client.get_tweets('covid', maxTweets = 15000)
print(f'tweets_df Shape - {tweets_df.shape}')
tweets_df.tail(10)
import langid
ids_langid = tweets_df['tweets'].apply(langid.classify)
langs = ids_langid.apply(lambda tuple: tuple[0])
sum(langs == "en")
result = pd.concat([tweets_df, langs], axis = 1, sort = False, ignore_index = True)
result.columns = ['tweets','language']
result.head()
# Filtering out non-English tweets
filtered_tweets = result.loc[result['language'] == "en"]
filtered_tweets.head()
filtered_tweets.to_csv('filtered_tweets.csv')
df = pd.read_csv('filtered_tweets.csv', usecols = ['tweets','language'])
df.head(20)
import re, string

def remove_pattern(text, pattern_regex):
    r = re.findall(pattern_regex, text)
    for i in r:
        text = re.sub(i, '', text)
    return text
# Removing mentions and Twitter handles
df['remove_names'] = np.vectorize(remove_pattern)(df['tweets'], "@[\w]*: | *RT* |@[\w]*")
df.head()
# Removing punctuation, numbers and special characters
df['remove_special_characters'] = df['remove_names'].str.replace("[^a-zA-Z# ]", "")
df.head()
# Removing links
df['remove_links'] = df['remove_special_characters'].replace(to_replace = r'^https?:\/\/.*[\r\n]*', value = '', regex = True)
df.head()
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
tweet_tokenizer = TweetTokenizer()

stopwords = set(stopwords.words("english"))
text_tokenized = []

for texts in df['remove_links']: #for each texts in the column
    text_tokenized.append(tweet_tokenizer.tokenize(texts)) #tokenize and append the lists without punctuation

# The list contains lists of titles

# Put all words in lists into one single list
all_words = []
for lists in text_tokenized:
    for i in range(len(lists)):
        if lists[i].lower() not in stopwords and 'http' not in lists[i].lower():
            all_words.append(lists[i].lower())

# Lemmatize
lemmatizer = WordNetLemmatizer()
words = [(lemmatizer.lemmatize(w)) for w in all_words]

print(words)
from nltk.probability import FreqDist
words_freq = FreqDist(words)
print(words_freq.most_common(100))
from wordcloud import WordCloud
import matplotlib.pyplot as plt
# Frequency of words
fdist = FreqDist(words)
# WordCloud
wc = WordCloud(width = 800, height = 400, max_words = 150, background_color = 'white').generate_from_frequencies(fdist)
plt.figure(figsize = (12,10))
plt.imshow(wc, interpolation = "bilinear")
plt.axis("off")
plt.show()