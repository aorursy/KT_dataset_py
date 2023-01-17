import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



from wordcloud import WordCloud,STOPWORDS

stopwords = set(STOPWORDS)



from textblob import TextBlob



import warnings

warnings.filterwarnings("ignore")



import re

from collections import Counter

import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         if (filename.endswith('Tweets.CSV')) :

#             print(os.path.join(dirname, filename))
# Reading data

# df=pd.read_csv('/kaggle/input/coronavirus-covid19-tweets-early-april/2020-03-29 Coronavirus Tweets.CSV', skiprows=lambda i: i!=0 and (i) % 1000 != 0)



# Read all files and down-sample

df2 = []

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        if (filename.endswith('Tweets.CSV')):

            df2.append(pd.read_csv(os.path.join(dirname, filename), header=0, skiprows=lambda i: i!=0 and (i) % 50 != 0))

df = pd.concat(df2, axis=0, ignore_index=True)



df.head()

df.shape
# display columns

print ("original columns: ")

df.columns



# dropping columns

tweet = df.copy()

tweet.drop(['status_id','user_id','screen_name','source','reply_to_status_id','reply_to_user_id','is_retweet','place_full_name','place_type','reply_to_screen_name','is_quote','followers_count','friends_count','account_lang','account_created_at','verified'],axis=1, inplace = True)

tweet.head()
# filtering data with 'country_code = US' and 'language = en'

# (tweet.country_code == "US") & 

tweet =tweet[(tweet.lang == "en")].reset_index(drop = True)

tweet.drop(['country_code','lang'],axis=1,inplace=True)



# check missing values

# tweet.isna().sum()



tweet.head()
# shape

tweet.shape



# # Top 5 most favourited tweets:

# fav = tweet[['favourites_count','text']].sort_values('favourites_count',ascending = False)[:5].reset_index()

# for i in range(5):

#     print(i,']', fav['text'][i],'\n')

    

# #Top 5 most retweeted tweets:

# retweet = tweet[['retweet_count','text']].sort_values('retweet_count',ascending = False)[:5].reset_index()

# for i in range(5):

#     print(i,']', retweet['text'][i],'\n')
def show_wordcloud(data , title = None):

    wordcloud = WordCloud(background_color='black',stopwords=stopwords,max_words=200,max_font_size=40).generate(str(data))

  

    fig = plt.figure(1, figsize=(15, 15))

    plt.axis('off')

    plt.title(title, size = 25)

    plt.imshow(wordcloud, interpolation='bilinear')

    plt.show()



show_wordcloud(tweet['text'])
# Extracting hashtags and accounts

stoptags = ['#covid19', '#covid_19', '#covid-19', '#covid', '#coronavirus', '#outgreak', '#virus', '#pandemic']



tweet['tags'] = tweet['text'].str.findall(r'(?:(?<=\s)|(?<=^))#.*?(?=\s|$|\.,)')

tweet['tags'] = tweet['tags'].apply(lambda word_list:list(map(lambda w: w.lower(), word_list))).apply(lambda word_list:list(filter(lambda w: w not in stoptags, word_list)))



tweet['accts'] = tweet['text'].str.findall(r'(?:(?<=\s)|(?<=^))@.*?(?=\s|$)')

tweet['entity_text'] = tweet['tags'].apply(' '.join) + ' ' + tweet['accts'].apply(' '.join)

tweet.head()
# Tokenizing and Removing special charactors

for i in range(tweet.shape[0]) :

    tweet['text'][i] = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|(#[A-Za-z0-9]+)", " ", tweet['text'][i]).split()).lower()

tweet['text'].head()
#Removing Stop Words

stopwords



tweet['text'] = tweet['text'].apply(lambda tweets: ' '.join([word for word in tweets.split() if word not in stopwords]))

tweet['text'].head() 
%time

tweet['sentiment'] = ' '

tweet['polarity'] = None

for i,tweets in enumerate(tweet.text) :

    blob = TextBlob(tweets)

    tweet['polarity'][i] = blob.sentiment.polarity

    if blob.sentiment.polarity > 0 :

        tweet['sentiment'][i] = 'positive'

    elif blob.sentiment.polarity < 0 :

        tweet['sentiment'][i] = 'negative'

    else :

        tweet['sentiment'][i] = 'neutral'

pd.set_option('display.max_colwidth', 400)

tweet.head()
print(tweet.sentiment.value_counts())

sns.countplot(x='sentiment', data = tweet);
count = pd.DataFrame(tweet.groupby('sentiment')['favourites_count'].sum())

count.head()
count = pd.DataFrame(tweet.groupby('sentiment')['retweet_count'].sum())

count.head()
plt.figure(figsize=(10,6))

sns.distplot(tweet['polarity'], bins=30)

plt.title('Sentiment Distribution',size = 15)

plt.xlabel('Polarity',size = 15)

plt.ylabel('Frequency',size = 15)

plt.show();
# format timestamp

tweet['created_at'] = pd.to_datetime(tweet['created_at'])

tweet['created_at'] = pd.IntervalIndex(pd.cut(tweet['created_at'], pd.date_range('2020-03-29', '2020-05-01', freq='2880T'))).left



# count sentiment

tweet_count1 = tweet.groupby(['created_at','sentiment'])['text'].count().reset_index().rename(columns={'text':'count'})

tweet_count1.head()



# check missing values

# tweet_count1.isna().sum()
#format sentiment table

times = tweet_count1.loc[tweet_count1['sentiment'] == 'negative']['created_at'].reset_index(drop = True)

pos = tweet_count1.loc[tweet_count1['sentiment'] == 'positive']['count'].reset_index(drop = True)

neutral = tweet_count1.loc[tweet_count1['sentiment'] == 'neutral']['count'].reset_index(drop = True)

neg = tweet_count1.loc[tweet_count1['sentiment'] == 'negative']['count'].reset_index(drop = True)



plt.figure(figsize=(10,6))

plt.xticks(rotation='45')

plt.title("Sentiment count vs. Time")



lin1=plt.plot(times, pos, 'ro-', label='positive')

lin2=plt.plot(times, neutral, 'g^-', label='neutral')

lin3=plt.plot(times, neg, 'b--', label='negative')

plt.legend()

plt.show
# "score" is defined as percent of positive tweets minus percent of negative tweets

score = (pos - neg) / (pos + neutral + neg)



plt.figure(figsize=(10,6))

plt.xticks(rotation='45')

plt.title("positive scroe (positive percent - negative percent) vs. Time")



lin1=plt.plot(times, score, 'ro-', label='positive score')

plt.legend()

plt.show
all_words = []

all_words = [word for i in tweet.entity_text for word in i.split()]

pos_words = tweet['entity_text'][tweet['sentiment'] == 'positive']

neg_words = tweet['entity_text'][tweet['sentiment'] == 'negative']

neutral_words = tweet['entity_text'][tweet['sentiment'] == 'neutral']

# show_wordcloud(pos_words , 'POSITIVE')

# show_wordcloud(neg_words , 'NEGATIVE')

# show_wordcloud(neutral_words , 'NEUTRAL')



def get_freq(word_list):

    freq = Counter(word_list).most_common(100)

    freq = pd.DataFrame(freq)

    freq.columns = ['word', 'frequency']

    return freq



all_freq = get_freq(all_words)

pos_freq = get_freq([word for i in pos_words for word in i.split()])

neg_freq = get_freq([word for i in neg_words for word in i.split()])



freq = pd.merge(all_freq,pos_freq,on='word',how='left').rename(columns={'frequency_x':'total','frequency_y':'pos'})

freq = pd.merge(freq,neg_freq,on='word',how='left').rename(columns={'frequency':'neg'}).fillna(0)

freq['score'] = (freq['pos'] - freq['neg'] ) / freq['total']



neg_freq_filtered = freq[(freq['score'] < 0.2) & (freq['neg'] > 0)].head(40).sort_values('score',ascending = True)



neg_freq_filtered.head(40)
#Positive 

freq[(freq['score'] >0.4) & (freq['pos'] !=0)].head(40).sort_values('score',ascending = False)
plt.figure(figsize = (20, 20))

sns.barplot(y="word", x="score",data=freq);
# tweet.to_csv('tweet.csv',index=False)