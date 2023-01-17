!pip install tweepy #install tweepy package
#import libraries
import os
import tweepy as tw
import pandas as pd
from textblob import TextBlob
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
from emoji import emojize
import nltk # Natural Language ToolKit
from nltk.corpus import stopwords
%matplotlib inline
#Enter Credentials
auth = tw.OAuthHandler("CUSTOMER_KEY", "CUSTOMER_SECRET")
auth.set_access_token("ACCESS_TOKEN", "ACCESS_TOKEN_SECRET")
api = tw.API(auth, wait_on_rate_limit=True)
#Enter keyword and number of tweets to be extracted from twitter
search_word = "#BBNaijaReunion"
max_count = 1000

#Extract tweets from twitter
tweets = tw.Cursor(api.search , q = search_word + " -filter:retweets", result_type = "recent", tweet_mode = "extended").items(max_count)

#Empty list
tweet_list = []
#loop through each extracted tweet and append/load the created_at,tweet_id etc into the empty list created above 
for tweet in tweets:
    tweet_list.append((tweet.created_at, tweet.id, tweet.full_text, tweet.user.screen_name, tweet.user.location))

#convert tweet list to a DataFrame
tweet_df = pd.DataFrame(tweet_list)
#Update the column names of the data frame
tweet_df.columns = ["created_on","tweet_id","tweet_text","user","location"]
#show first five records in the data frame
tweet_df.shape

#use TextBlob sentiment polarity and emojize to get the sentiment and assign appropriate emojis
tw_list = []
for i in tweet_df.tweet_text:
    a= TextBlob(i)
    b= a.sentiment.polarity
    if b < 0:
        tw_list.append((b, emojize(":disappointed_face:")))
    elif b == 0:
        tw_list.append((b, emojize(":zipper-mouth_face:")))
    elif b > 0 and b <= 1:
        tw_list.append((b, emojize(":grinning_face_with_big_eyes:")))
    else:
        tw_list.append((b, ""))
        
em_df = pd.DataFrame(tw_list)      
em_df.columns = ["Sentiment", "Emoji"]
em_df
#Join sentiment and emoji with the original dataframe created earlier        
new_df = tweet_df.join(em_df)    


new_df.head() #review top 5 records in the data frame
#plot the first tweet in a wordcloud for review. 
text = new_df.tweet_text[0]
# Create and generate a word cloud image:
wordcloud = WordCloud(max_words=100, background_color="white").generate(text)

# Display the generated image:
plt.imshow(wordcloud,interpolation="bilinear")
plt.axis("off")
plt.show()

#wordcloud.to_file("img/first_review.png") saves the image if you want
#Put all the texts in the tweets together for creating the wordcloud
text = ""
for i in new_df.tweet_text:
    text = text + i

# Create and generate a word cloud image (using default stopwords):
wordcloud = WordCloud(background_color="white").generate(text)

plt.subplots(figsize = (8,8))
plt.imshow(wordcloud,interpolation="bilinear")
plt.axis("off")
plt.show()
# Create and generate a word cloud image (this time with additional stopwords excluded):

stopwords = set(STOPWORDS)
sw_list = ['BBNREUNION', 'BBNaijaReunion' , 'https' , 'people' , 'show' , 'BBNaijaPepperDemReunion' , 'Know' , 'go' , 'show' , 'will' , 'dey' , 'BBNaijaReunion2020', '#BBNaijaReunion2020', '#BBNaijaReunion', '#bbnreunion','bbnreunion', '#BBNReunion', 'co',
          '#BBNReunion ', 'BBNReunion ', ' BBNReunion ', 'BBN', 'Reunion', 'BBNreunionMorning','#bbnaija2020', '#bbnreunionYou', '#BBNreunion#LANBENA', 'https://t.co/aHjU3qXYMa@Symply_Tacha', 'https://t.co/myNcecKZ18Happy',
'#BBNreunion',  'https://t.co/T7HjgjiUpZ','https://t.co/jCXmAybvunI', '#bbnreunionWe', 'reunion', '#BBNaijaReunion2020I',
'BBNReunion,','https://t.co/m40wFmVVePBut','#bbnreunionWho','#bbnreunionCooking', '#bbnreunionTodays', '#BBNaijaPepperDemReunion',
'#BBNAIJA','https://t.co/ZTvmOsdJeUThe','#BBNaija','#BBNPepperDemReunion','https://t.co/Thp1hOWpZRIf','https://t.co/q2Lbmh8c7O@Ebuka',
'🤣#BbnreunionHappy','https://t.co/Md3XTknIW7@Cynthia4reel1', 'https://t.co/59mgdQglViThis', 'https://t.co/MAwuijbQSqMercyeke',
'#bbnreunionAbsolutely','https://t.co/UjzTHcOHnyThe','https://t.co/tU32gqW0dVOmashola', '#BBNaijaPepperDemReunion','https://t.co/hcKhmixa4aAlready',
'#BBNaijaPepperDemReunion', 'https://t.co/uXzeSCUrrAAfter','https://t.co/5HqgICR2hWThe','https://t.co/qOhAjh3OKmLambo',"#bbnreunionI'm",
'#BBNaijaPepperDemReunionI''#BBNaijaPepperDemReunion','#bbnreunion@Ebuka', '#BBNreunion','#BBNPepperDemReunion','#bbnreunion“Na']
for i in sw_list:
    stopwords.add(i)
    
wordcloud = WordCloud(stopwords=stopwords, background_color="white").generate(text)


plt.subplots(figsize = (10,10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

#extract negative tweets into a different data frame
filter1 = new_df.Sentiment < 0
neg_df = new_df[filter1]
neg_df.head()
#put all the negative texts together
text = ""
for i in neg_df.tweet_text:
    text = text + i
#create wordcloud for negative tweets
wordcloud = WordCloud(stopwords=stopwords, background_color="white").generate(text)

plt.subplots(figsize = (10,10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
#Extract positive tweets
filter2 = new_df.Sentiment > 0 
poss_df = new_df[filter2]
poss_df.head()
#Put all positive tweet texts together 
text = ""
for i in poss_df.tweet_text:
    text = text + i
#create wordcloud for positive tweets
wordcloud = WordCloud(stopwords=stopwords, background_color="white").generate(text)

plt.subplots(figsize = (10,10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
#what location are most of the tweets coming from
location = new_df.groupby("location")["tweet_id"].count()
location.sort_values(ascending = False)
#you can also check which users are tweeting about #BBNaijaReunion the most
user = new_df.groupby("user")["tweet_id"].count()
user.sort_values(ascending = False)
#Export tweets to csv for further analysis
new_df.to_csv("bbnaijareuniontweets.csv")