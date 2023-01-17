#Importing required libraries
import pandas as pd
import numpy as np
from timeit import default_timer as timer
import matplotlib.pyplot as plt
import requests
import os
# import tweepy
import re
import seaborn as sns
%matplotlib inline
#Reading the csv file of tweets
archive_df = pd.read_csv('../input/wrangle/twitter_archive_enhanced.csv')
archive_df.head()
# Getting image prediction file from the udacity url 
# url = 'https://d17h27t6h515a5.cloudfront.net/topher/2017/August/599fd2ad_image-predictions/image-predictions.tsv'<br>
# response = requests.get(url)<br>

# with open(os.path.join('image_predictions.tsv'),mode='wb') as file:
#     file.write(response.content)
#Reading the csv file downloaded
img_prediction_df = pd.read_csv('../input/wrangle/image_predictions.tsv',sep='\t')
img_prediction_df.head()
# Creating an API object that will be used to gather Twitter data, ids removed for privacy reasons.
# CONSUMER_KEY = ""
# CONSUMER_SECRET = ""
# OAUTH_TOKEN = ""
# OAUTH_TOKEN_SECRET = ""



# auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
# auth.set_access_token(OAUTH_TOKEN, OAUTH_TOKEN_SECRET)
# api = tweepy.API(auth,parser=tweepy.parsers.JSONParser())

# #List of all the tweet ids in the archive dataframe<br>
# tweet_ids = archive_df['tweet_id'].values
# # Accessing the data from the api ids.

# tweet_data=[]  #Store data
# success_ids = [] # Store ID of successful tweets
# failure_ids = [] # Store ID of failed tweets

# #Time required to collect all the data
# start = timer()

# # Getting tweet JSON data via tweet ID from the API created using Tweepy 
# for tweet in tweet_ids:
#     try:
#         data = api.get_status(tweet,tweet_mode='extended',wait_on_rate_limit = True,wait_on_rate_limit_notify = True)
#         tweet_data.append(data)
#         success_ids.append(tweet)</br>
        
#         except:
#         print(tweet)
#         failure_ids.append(tweet)
        

# end = timer()
# time_taken = end-start

# print("Time Taken : {}s".format(time_taken))
# #Storing the json data in txt file
# import json
# with open('tweet_json.txt', mode = 'w') as file:
#     json.dump(tweet_data, file)
# #Reading the json file in a dataframe
# counts_df = pd.read_json('tweet_json.txt')
# counts_df['tweet_id'] = success_ids
# #Checking for the required columns
# counts_df.columns
# #Updating the dataframe with required columns
# counts_df = counts_df[['tweet_id','retweet_count','favorite_count']]
# counts_df
# #Save the dataframe for future reference 
# counts_df.to_csv('retweet_count.csv',index=False)
counts_df = pd.read_csv('../input/wrangle/retweet_count.csv')
archive_df
archive_df.tweet_id.duplicated().sum()
archive_df.info()
archive_df.in_reply_to_status_id.value_counts()
archive_df.in_reply_to_user_id.value_counts()
archive_df.retweeted_status_id.value_counts()
archive_df.retweeted_status_user_id.value_counts()
archive_df.retweeted_status_timestamp.value_counts()
archive_df.rating_numerator.value_counts()
archive_df.rating_denominator.value_counts()
archive_df.source.value_counts()
archive_df.name.value_counts()
img_prediction_df.sample(5)
img_prediction_df.tweet_id.duplicated().sum()
img_prediction_df.shape
img_prediction_df.info()
img_prediction_df.jpg_url.duplicated().sum()
img_prediction_df.p1.value_counts()
counts_df.sample(5)
counts_df.info()
counts_df.tweet_id.duplicated().sum()
counts_df.describe()
#Creating a new dataframe
merged_df = archive_df.merge(img_prediction_df,left_on='tweet_id',right_on='tweet_id',how='inner')
merged_df = merged_df.merge(counts_df,left_on='tweet_id',right_on='tweet_id',how='inner')
merged_df.info()
#Creating a single coumn for dog_stage
merged_df['dog_stage'] = merged_df['text'].str.extract('(doggo|floofer|pupper|puppo)')


merged_df.drop(['doggo','floofer','pupper','puppo'],axis=1,inplace=True)
merged_df.info()
#Remove the retweets and keeping only the original tweets
merged_df = merged_df[np.isnan(merged_df.retweeted_status_id)]
merged_df.sample(5)
#Empty columns
empty_cols = [col for col in merged_df.columns if merged_df[col].isnull().all()]
empty_cols
#Dropping those empty columns
merged_df.drop(['retweeted_status_id','retweeted_status_user_id','retweeted_status_timestamp'],axis=1,inplace=True)

#Dropping unwanted column
merged_df.drop(['in_reply_to_status_id','in_reply_to_user_id','img_num'],axis=1,inplace=True)
list(merged_df)
#Slicing the timezone from timestamp and then converting it to datetime
merged_df['timestamp'] = merged_df.timestamp.str.slice(start=0, stop=-6)
merged_df['timestamp'] = pd.to_datetime(merged_df.timestamp, format= "%Y-%m-%d %H:%M:%S")
merged_df.info()
merged_df.source.value_counts()
#Split the column and then reassign
no_need , merged_df['source'] = merged_df.source.str.split('">',1).str

#Slice the tag at the end
merged_df['source'] = merged_df['source'].str.slice(0,-4)
merged_df.source.value_counts()
#tweet_id to string
merged_df['tweet_id'] = merged_df['tweet_id'].astype(str)
merged_df.info()
# Find all names that start with a lowercase letter
lower_name = []
for data in merged_df['name']:
    if data[0].islower() and data not in lower_name:
        lower_name.append(data)
        
lower_name
# Replace names starting with a lowercase letter with a NaN
merged_df['name'].replace(lower_name, 
                        np.nan,
                       inplace = True)

# Replace None with a NaN
merged_df['name'].replace('None', 
                        np.nan,
                       inplace = True)
merged_df.name.value_counts()
merged_df.rating_numerator.value_counts()
# Find text, index, and rating for tweets that contain a decimal in the numerator 
ratings_of_decimals_text = []
ratings_of_decimals_index = []
ratings_of_decimals = []

for i, text in merged_df['text'].iteritems():
    if bool(re.search('\d+\.\d+\/\d+', text)):
        ratings_of_decimals_text.append(text)
        ratings_of_decimals_index.append(i)
        ratings_of_decimals.append(re.search('\d+\.\d+', text).group())

        
ratings_of_decimals_text
#Put those ratings in that index
merged_df.loc[ratings_of_decimals_index[0],'rating_numerator'] = float(ratings_of_decimals[0])
merged_df.loc[ratings_of_decimals_index[1],'rating_numerator'] = float(ratings_of_decimals[1])
merged_df.loc[ratings_of_decimals_index[2],'rating_numerator'] = float(ratings_of_decimals[2])
merged_df.loc[ratings_of_decimals_index[3],'rating_numerator'] = float(ratings_of_decimals[3])
# Convert all the values to float
merged_df.rating_numerator = merged_df.rating_numerator.astype(float)
merged_df.rating_denominator = merged_df.rating_denominator.astype(float)
merged_df.info()
#Creating ratio to understand and analyze better
merged_df['ratio'] = merged_df['rating_numerator'] / merged_df['rating_denominator']
merged_df['ratio']
merged_df.jpg_url.duplicated().sum()
pd.set_option('display.max_columns', None)
merged_df.sample(5)
merged_df.to_csv('twitter_archive_master.csv',encoding = 'utf-8',index=False)
sns.set()
sns.lmplot(x='favorite_count',y='retweet_count',data=merged_df)
plt.title('Favorite Count vs Retweet Count')
plt.xlabel('Retweet Count')
plt.ylabel('Favorite Count')
plt.savefig('retweet_vs_favorite.png');
top_10_names = merged_df.name.value_counts()[0:10].sort_values(ascending=True)
top_10_names.plot.barh()
plt.savefig('top_names.png');
plt.figure(figsize=(12,6))
merged_df.plot(x='retweet_count', y='ratio', kind='scatter')
plt.ylim(-0.1,2)
plt.xlabel('Retweet Counts')
plt.ylabel('Ratings')
plt.title('Retweet Counts by Ratings Scatter Plot')
plt.savefig('ratio_retweet.png')
merged_df.groupby('timestamp')['ratio'].mean().plot(kind='line')
plt.ylim(0, 2)
plt.title('Ratings over Time')
plt.xlabel('Time')
plt.ylabel('Rating Ratio')
plt.savefig('ratio_with_time');
merged_df.sample(5)