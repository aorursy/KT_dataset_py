import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



from wordcloud import WordCloud,STOPWORDS

stopwords = set(STOPWORDS)



from textblob import TextBlob



import re



from collections import Counter



import warnings

warnings.filterwarnings("ignore")



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# Reading data

df=pd.read_csv('/kaggle/input/coronavirus-covid19-tweets-late-april/2020-04-16 Coronavirus Tweets.CSV')

df.head()
# display columns

df.columns
# dropping columns

tweet = df.copy()

tweet.drop(['status_id','user_id','screen_name','source','reply_to_status_id','reply_to_user_id','is_retweet','place_full_name','place_type','reply_to_screen_name','is_quote','followers_count','friends_count','account_lang','account_created_at','verified'],axis=1, inplace = True)

tweet.head()
# filtering data with 'country_code = IN' and 'language = en'

tweet =tweet[(tweet.country_code == "IN") & (tweet.lang == "en")].reset_index(drop = True)

tweet.drop(['country_code','lang'],axis=1,inplace=True)

tweet.head()
# created_at column

tweet["created_at"] = tweet["created_at"].apply(lambda i:(int(i.split("T")[1].split(":")[0])+int(i.split("T")[1].split(":")[1])/60))
# shape

tweet.shape
# check missing values

tweet.isna().sum()
# data preprocessing

for i in range(tweet.shape[0]) :

    tweet['text'][i] = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|(#[A-Za-z0-9]+)", " ", tweet['text'][i]).split()).lower()

tweet['text'].head()
fav = tweet[['favourites_count','text']].sort_values('favourites_count',ascending = False)[:5].reset_index()

for i in range(5):

    print(i,']', fav['text'][i],'\n')
retweet = tweet[['retweet_count','text']].sort_values('retweet_count',ascending = False)[:5].reset_index()

for i in range(5):

    print(i,']', retweet['text'][i],'\n')
plt.figure(1, figsize=(10,6))

plt.hist(tweet["created_at"],bins = 24);

plt.xlabel('Hours',size = 15)

plt.ylabel('No. of Tweets',size = 15)

plt.title('No. of Tweets per Hour',size = 15)
def show_wordcloud(data , title = None):

    wordcloud = WordCloud(background_color='black',stopwords=stopwords,max_words=200,max_font_size=40).generate(str(data))

  

    fig = plt.figure(1, figsize=(15, 15))

    plt.axis('off')

    plt.title(title, size = 25)

    plt.imshow(wordcloud, interpolation='bilinear')

    plt.show()



show_wordcloud(tweet['text'])
stopwords
#Removing Stop Words

tweet['text'] = tweet['text'].apply(lambda tweets: ' '.join([word for word in tweets.split() if word not in stopwords]))

tweet['text'].head() 
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

tweet.head()
print(tweet.sentiment.value_counts())

sns.countplot(x='sentiment', data = tweet);
plt.figure(figsize=(10,6))

sns.distplot(tweet['polarity'], bins=30)

plt.title('Sentiment Distribution',size = 15)

plt.xlabel('Polarity',size = 15)

plt.ylabel('Frequency',size = 15)

plt.show();
pos = tweet['text'][tweet['sentiment'] == 'positive']

show_wordcloud(pos , 'POSITIVE')



neg = tweet['text'][tweet['sentiment'] == 'negative']

show_wordcloud(neg , 'NEGATIVE')



neutral = tweet['text'][tweet['sentiment'] == 'neutral']

show_wordcloud(neutral , 'NEUTRAL')
count = pd.DataFrame(tweet.groupby('sentiment')['favourites_count'].sum())

count.head()
words = []

words = [word for i in tweet.text for word in i.split()]
freq = Counter(words).most_common(30)

freq = pd.DataFrame(freq)

freq.columns = ['word', 'frequency']

freq.head()
plt.figure(figsize = (10, 10))

sns.barplot(y="word", x="frequency",data=freq);
tweet.to_csv('tweet.csv',index=False)
#Big data project start

#Analysis of United Sates US 

#Analysis of Canada CN
#to view country codes needed first start by making a copy of the datset then dropping columns

country_view = df.copy()

country_view.drop(['status_id','user_id','screen_name','source','reply_to_status_id','reply_to_user_id','is_retweet','place_full_name','place_type','reply_to_screen_name','is_quote','followers_count','friends_count','account_lang','account_created_at','verified'],axis=1, inplace = True)

country_view.head()

#to view country codes needed

country_view = country_view.dropna()

country_view.head()
#update stop words for better word cloud data 

stopwords.update(["https", "name", "dtype", "text", "she", "whether", "ft", "in"])

stopwords
#create a new dataset that houses all data for US 

us_dataset = pd.DataFrame(df[(df.country_code == "US") & (df.lang == "en")])

us_dataset.to_csv('us_data.csv')
#Create a new dataset that houses all data for CN 

cn_dataset = pd.DataFrame(df[(df.country_code == "CN") & (df.lang == "en")])

cn_dataset.to_csv('cn_data.csv')
#view contents of us_data file 

us_dataset = pd.read_csv('./us_data.csv')

us_dataset
us_dataset.shape
# Making a copy of the dataset and dropping columns from us_dataset 

us_tweet = us_dataset.copy()

us_tweet.drop(['status_id','user_id','screen_name','source','reply_to_status_id','reply_to_user_id','is_retweet','place_full_name','place_type','reply_to_screen_name','is_quote','followers_count','friends_count','account_lang','account_created_at','verified'],axis=1, inplace = True)

us_tweet.head()
us_tweet.shape
# data preprocessing to make text uniform 

for i in range(us_tweet.shape[0]) :

    us_tweet['text'][i] = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|(#[A-Za-z0-9]+)", " ", us_tweet['text'][i]).split()).lower()

us_tweet['text'].head()
#Removing Stop Words

us_tweet['text'] = us_tweet['text'].apply(lambda tweets: ' '.join([word for word in tweets.split() if word not in stopwords]))

us_tweet['text'].head() 
#first word cloud showing data without sentiment 

def show_wordcloud(data , title = None):

    

    wordcloud = WordCloud(background_color='black',stopwords=stopwords,max_words=200,max_font_size=40).generate(str(data))

  

    fig = plt.figure(1, figsize=(15, 15))

    plt.axis('off')

    plt.title(title, size = 25)

    plt.imshow(wordcloud, interpolation='bilinear')

    plt.show()



show_wordcloud(us_tweet['text'])

#sentiment analysis of positive negative and neutral on us_tweet dataset

us_tweet['sentiment'] = ' '

us_tweet['polarity'] = None

for i,tweets in enumerate(us_tweet.text) :

    blob = TextBlob(tweets)

    us_tweet['polarity'][i] = blob.sentiment.polarity

    if blob.sentiment.polarity > 0 :

        us_tweet['sentiment'][i] = 'positive'

    elif blob.sentiment.polarity < 0 :

        us_tweet['sentiment'][i] = 'negative'

    else :

        us_tweet['sentiment'][i] = 'neutral'

us_tweet.head()
#chart representation of sentiment for US

print(us_tweet.sentiment.value_counts())

sns.countplot(x='sentiment', data = us_tweet);
# word cloud representation of sentiment analysis for US 

pos = us_tweet['text'][us_tweet['sentiment'] == 'positive']

show_wordcloud(pos , 'POSITIVE')



neg = us_tweet['text'][us_tweet['sentiment'] == 'negative']

show_wordcloud(neg , 'NEGATIVE')



neutral = us_tweet['text'][us_tweet['sentiment'] == 'neutral']

show_wordcloud(neutral , 'NEUTRAL')