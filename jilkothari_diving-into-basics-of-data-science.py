!pip install tweepy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import requests
import tweepy 
import json
import tweepy
from tweepy import OAuthHandler
from timeit import default_timer as timer

a = pd.read_csv("../input/weratedogs-udacity/twitter-archive-enhanced.csv")
a.shape
a.columns
a.head(3)
url = "https://d17h27t6h515a5.cloudfront.net/topher/2017/August/599fd2ad_image-predictions/image-predictions.tsv"
response = requests.get(url)

with open('image-predictions.tsv', mode ='wb') as file:
    file.write(response.content)

b = pd.read_csv("image-predictions.tsv", sep='\t')
b.shape
b.columns
b.head(3)
import tweepy
from tweepy import OAuthHandler
import json
from timeit import default_timer as timer

# Query Twitter API for each tweet in the Twitter archive and save JSON in a text file
# These are hidden to comply with Twitter's API terms and conditions
consumer_key = 'HIDDEN'
consumer_secret = 'HIDDEN'
access_token = 'HIDDEN'
access_secret = 'HIDDEN'

auth = OAuthHandler(consumer_key , consumer_secret )
auth.set_access_token(access_token, access_secret)

api = tweepy.API(auth, wait_on_rate_limit=True)


# df_1 is a DataFrame with the twitter_archive_enhanced.csv file.
# Tweet IDs for which to gather additional data via Twitter's API
d_1 = pd.read_csv("../input/weratedogs-udacity/twitter-archive-enhanced.csv")
tweet_ids = d_1.tweet_id.values
len(tweet_ids)
# Query Twitter's API for JSON data for each tweet ID in the Twitter archive
count = 0
fails_dict = {}
start = timer()
# Save each tweet's returned JSON as a new line in a .txt file
with open('tweet_json.txt', 'w') as outfile:
    # This loop will likely take 20-30 minutes to run because of Twitter's rate limit
    for tweet_id in tweet_ids:
        count += 1
        print(str(count) + ": " + str(tweet_id))
        try:
            tweet = api.get_status(tweet_id, tweet_mode='extended')
            print("Success")
            json.dump(tweet._json, outfile)
            outfile.write('\n')
        except tweepy.TweepError as e:
            print("Fail")
            fails_dict[tweet_id] = e
            pass
end = timer()
print(end - start)
print(fails_dict)
d = pd.read_json("../input/weratedogs-udacity/tweet-json.txt", orient = 'records', lines = True)
d.columns
c= d[["id", "retweet_count", "favorite_count"]].copy()
c= c.rename(columns={"id": "tweet_id"})
c.shape
c.head(3)
a_clean = a.copy()
b_clean = b.copy()
c_clean = c.copy()
df = b.merge(a, on='tweet_id', how='inner').merge(c, on='tweet_id', how='inner')
df.head()
df_clean = df.copy()
sum(df.duplicated())
df.tweet_id.unique()
df.jpg_url.unique()
df.img_num.value_counts()
df.p1.unique()
df.p1_conf.unique()
df.p1_dog.unique
df.columns
df.in_reply_to_status_id.unique()
df.in_reply_to_user_id.unique()
df.timestamp.unique()
df.source.unique()
df.text.unique()
df.retweeted_status_id.unique()
df.retweeted_status_user_id.unique()
df.retweeted_status_timestamp.unique()
df.expanded_urls.unique()
df.rating_numerator.unique()
df.rating_denominator.value_counts()
df.name.unique()
df.doggo.unique()
df.floofer.unique()
df.pupper.unique()
df.puppo.unique()
df.retweet_count.unique
df.favorite_count.unique
df.columns
df = df.rename(columns={"tweet_id":"Tweet_Id", "jpg_url":"Jpg_Url" ,"img_num":"Image_Number", "p1":"First_Prediction", "p1_conf":"Confidence_of_First_Prediction", "p1_dog":"Result_of_First_Prediction","p2":"Second_Prediction", "p2_conf":"Confidence_of_Second_Prediction", "p2_dog":"Result_of_Second_Prediction","p3":"Third_Prediction", "p3_conf":"Confidence_of_Third_Prediction", "p3_dog":"Result_of_Third_Prediction", "in_reply_to_status_id":"Status_Id_of_Reply", "in_reply_to_user_id":"User_Id_of_Reply", "timestamp":"Timestamp", "source":"Source", "text":"Text", "expanded_urls":"Urls", "rating_numerator":"Rating", "rating_denominator":"Rating_Denominator", "name":"Name","retweet_count":"Number_of_Retweets", "favorite_count":"Number_of_favourites" })
df.columns
breed= pd.read_csv("../input/weratedogs-udacity/breeds.csv")
breed.head()
arr=list(breed['breed'])
arr
df['First_Prediction'] = df['First_Prediction'].str.lower()
df = df[df['First_Prediction'].isin(arr)]
df.First_Prediction.unique()
df['Second_Prediction'] = df['Second_Prediction'].str.lower()
df = df[df['Second_Prediction'].isin(arr)]
df['Third_Prediction'] = df['Third_Prediction'].str.lower()
df = df[df['Third_Prediction'].isin(arr)]
df=df[df['Status_Id_of_Reply'].isnull()]

df=df[df['User_Id_of_Reply'].isnull()]
df = df.drop(['Status_Id_of_Reply', 'User_Id_of_Reply'], axis=1)
df.columns
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df.dtypes
df = df[df['Rating'] >= 10]  
df.Rating.unique()
df = df[df['Rating_Denominator'] == 10]  
df.Rating_Denominator.unique()
df = df.drop(["Rating_Denominator"], axis=1)
df=df[df['retweeted_status_id'].isnull()]

df.retweeted_status_user_id.value_counts()
df.retweeted_status_timestamp.value_counts()
df = df.drop(['retweeted_status_id', 'retweeted_status_user_id','retweeted_status_timestamp'], axis=1)
df.columns
df['New_name'] = df.Name.str.capitalize()
df.shape
df[['Name','New_name']].head()

df=df[df['Name']==df['New_name']]
df.drop(['New_name'], axis=1)
df.shape
df.Name.unique()
import numpy as np
test = df[['doggo','floofer','pupper', 'puppo'  ]].copy()
test['doggo'].replace('None', np.nan, inplace=True)
test['floofer'].replace('None', np.nan, inplace=True)
test['pupper'].replace('None', np.nan, inplace=True)
test['puppo'].replace('None', np.nan, inplace=True)

y=test.fillna('').sum(1).replace('', np.nan)
df['Life_Cycle']= y
df= df.drop(['doggo','floofer', 'pupper','puppo'], axis=1)

df.to_csv("twitter_archive_master.csv")

graph=df.select_dtypes(["int64","float64"])
graph.hist(figsize=(16,14))
import seaborn as sns
corr_new=df[['Number_of_favourites','Number_of_Retweets']].corr()
sns.heatmap(corr_new,annot=True)

new_plot  = df[df.Life_Cycle.str.len()<8]
new_plot.Life_Cycle.value_counts().plot(kind='bar')
df['First_Prediction'].value_counts().sort_values(ascending=False).head(10).plot(kind='bar')
df['Second_Prediction'].value_counts().sort_values(ascending=False).head(10).plot(kind='bar')
df['Third_Prediction'].value_counts().sort_values(ascending=False).head(10).plot(kind='bar')