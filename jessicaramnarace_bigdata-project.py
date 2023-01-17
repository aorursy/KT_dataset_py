#Big Data Project 

#STEP 1 : IMPORT NECESSARY LIBRARIES 
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
#STEP 2 : READ DESIRED CSV FILE TO INVESTIGATE  DATA
# Reading data

df=pd.read_csv('/kaggle/input/coronavirus-covid19-tweets-late-april/2020-04-16 Coronavirus Tweets.CSV')

df.head()
#STEP 3 : INVESITGATE DATA SOURCE FOR COUNTIRES WITH SIMILAR ATTRIBUTES
# display columns

df.columns
#to view country codes needed first start by making a copy of the datset then dropping columns

tweet = df.copy()

tweet.drop(['status_id','user_id','screen_name','source','reply_to_status_id','reply_to_user_id','is_retweet','place_full_name','place_type','reply_to_screen_name','is_quote','followers_count','friends_count','account_lang','account_created_at','verified'],axis=1, inplace = True)

tweet = tweet.dropna()

tweet.head()
tweet.shape
#STEP 4: EXTRACT DATA TO BE USED IN PROJECT
#Create a new subset dataset that houses all data for both US & CN to be used for comparison

us_cn_dataset = pd.DataFrame(df[(df.country_code == "CN") | (df.country_code == "US") & (df.lang == "en")])

us_cn_dataset.drop(['status_id','user_id','screen_name','source','reply_to_status_id','reply_to_user_id','is_retweet','place_full_name','place_type','reply_to_screen_name','is_quote','followers_count','friends_count','account_lang','account_created_at','verified'],axis=1, inplace = True)

us_cn_dataset.to_csv('us_ca_data.csv')

us_cn_dataset.head()
#STEP 5: READ AND CLEAN DATA
us_cn_dataset = pd.read_csv("./us_ca_data.csv")
# data preprocessing



for i in range(us_cn_dataset.shape[0]):

    us_cn_dataset["text"][i] = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|(#[A-Za-z0-9]+)", " ", us_cn_dataset["text"][i]).split()).lower()

us_cn_dataset["text"].head()



  
#STEP 6: PREPARE SENTIMENT ANALYSIS 
#update stop words for better word cloud data 

stopwords.update(["https", "name", "dtype", "text", "she", "whether", "ft", "in"])

#Removing Stop Words

us_cn_dataset['text'] = us_cn_dataset['text'].apply(lambda tweets: ' '.join([word for word in tweets.split() if word not in stopwords]))

us_cn_dataset['text'].head() 

#sentiment analysis of positive negative and neutral on us_cn dataset

us_cn_dataset['sentiment'] = ' '

us_cn_dataset['polarity'] = None

for i,tweets in enumerate(us_cn_dataset.text) :

    blob = TextBlob(tweets)

    us_cn_dataset['polarity'][i] = blob.sentiment.polarity

    if blob.sentiment.polarity > 0 :

        us_cn_dataset['sentiment'][i] = 'positive'

    elif blob.sentiment.polarity < 0 :

        us_cn_dataset['sentiment'][i] = 'negative'

    else :

        us_cn_dataset['sentiment'][i] = 'neutral'

#us_cn_dataset.head()

print(us_cn_dataset.sentiment.value_counts())
#STEP 7 : DISPLAY SENTIMENT IN WORD CLOUD 
#word cloud function

def show_wordcloud(data , title = None):

    

    wordcloud = WordCloud(background_color='black',stopwords=stopwords,max_words=200,max_font_size=40).generate(str(data))

  

    fig = plt.figure(1, figsize=(15, 15))

    plt.axis('off')

    plt.title(title, size = 25)

    plt.imshow(wordcloud, interpolation='bilinear')

    plt.show()
# word cloud representation of sentiment analysis for both us&cn 

pos = us_cn_dataset['text'][us_cn_dataset['sentiment'] == 'positive']

show_wordcloud(pos , 'POSITIVE')



neg = us_cn_dataset['text'][us_cn_dataset['sentiment'] == 'negative']

show_wordcloud(neg , 'NEGATIVE')



neutral = us_cn_dataset['text'][us_cn_dataset['sentiment'] == 'neutral']

show_wordcloud(neutral , 'NEUTRAL')
#STEP 8: COMPARE SENTIMENT SEPARATELY 
#Create a new subset dataset for US to be used for comparison

us_dataset = pd.DataFrame(us_cn_dataset[(us_cn_dataset.country_code == "US") & (us_cn_dataset.lang == "en")])

print(us_dataset.sentiment.value_counts())
# word cloud representation of sentiment analysis for US 

pos = us_dataset['text'][us_dataset['sentiment'] == 'positive']

show_wordcloud(pos , 'POSITIVE')



neg = us_dataset['text'][us_dataset['sentiment'] == 'negative']

show_wordcloud(neg , 'NEGATIVE')



neutral = us_dataset['text'][us_dataset['sentiment'] == 'neutral']

show_wordcloud(neutral , 'NEUTRAL')
#Create a new subset dataset for CN to be used for comparison

cn_dataset = pd.DataFrame(us_cn_dataset[(us_cn_dataset.country_code == "CN") & (us_cn_dataset.lang == "en")])

print(cn_dataset.sentiment.value_counts())
# word cloud representation of sentiment analysis for CN 

pos = cn_dataset['text'][cn_dataset['sentiment'] == 'positive']

show_wordcloud(pos , 'POSITIVE')



neg = cn_dataset['text'][cn_dataset['sentiment'] == 'negative']

show_wordcloud(neg , 'NEGATIVE')



neutral = cn_dataset['text'][cn_dataset['sentiment'] == 'neutral']

show_wordcloud(neutral , 'NEUTRAL')
#END OF PROJECT 