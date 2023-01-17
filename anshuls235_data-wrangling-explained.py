!pip install tweepy
#Libraries Used

import pandas as pd

import requests

import tweepy

import os

import json

import numpy as np

import re

%matplotlib inline
for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
url = 'https://d17h27t6h515a5.cloudfront.net/topher/2017/August/599fd2ad_image-predictions/image-predictions.tsv'

response = requests.get(url)

with open(os.path.join(os.getcwd(), url.split('/')[-1]), mode='wb') as file:

    file.write(response.content)
for dirname, _, filenames in os.walk('/kaggle/working'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
consumer_key = 'U4hUt6MkwsBunrLeP7gBrfs9q'

consumer_secret = '9RBX4KeUOxVhBRab0TBHjDcJp9hcSyvHieyA50as2auN5PxzWJ'

access_token = '1929558530-UuS1sgoWlZtz5xHhJVbWpq0pWoCdR9X7H8Cq89P'

access_secret = '36sqFNfP4b8QIY374adQUgUBrk0Ui5UEB4i3Z5e2qS5Qm'
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)

auth.set_access_token(access_token, access_secret)



api = tweepy.API(auth, parser=tweepy.parsers.JSONParser(), wait_on_rate_limit=True)
df_twitter_archive = pd.read_csv('../input/twitter-archive-enhanced.csv')
for tweet_id in df_twitter_archive.tweet_id:

    try:

        tweet_json = api.get_status(tweet_id, tweet_mode = 'extented')

        with open('/kaggle/working/tweet_json.txt', mode='a') as file:

            json.dump(tweet_json, file)

            file.write('\n')

    except Exception as e:

        print(str(tweet_id) + ': ' + str(e))
lists = [] #an empyty list to store a dictionaries

with open('/kaggle/working/tweet_json.txt') as file:

    lines = file.read().splitlines()

    for line in lines:

        data = json.loads(line)

        row = {

            'tweet_id'      : data['id'],

            'retweet_count' : data['retweet_count'],

            'favorite_count': data['favorite_count']

        }

        lists.append(row)

df_tweet_data = pd.DataFrame(lists,columns=['tweet_id','retweet_count','favorite_count'])
df_img_predictions = pd.read_csv('/kaggle/working/image-predictions.tsv', sep='\t')
df_twitter_archive
df_img_predictions
df_tweet_data
df_twitter_archive.info()
df_img_predictions.info()
df_tweet_data.info()
all_columns = pd.Series(list(df_twitter_archive) + list(df_img_predictions) + list(df_tweet_data))

all_columns[all_columns.duplicated()]
df_twitter_archive.tweet_id.nunique()
df_img_predictions.tweet_id.nunique()
df_tweet_data.tweet_id.nunique()
df_twitter_archive[df_twitter_archive.text.duplicated()]
df_twitter_archive.source.value_counts()
df_twitter_archive.sample(25)
df_twitter_archive.name.isnull().sum()
df_twitter_archive.loc[np.random.randint(0,df_twitter_archive.shape[0],40), ['text','name']]
df_twitter_archive.name.value_counts()
df_twitter_archive.describe()
df_twitter_archive.rating_numerator.value_counts()
df_twitter_archive.rating_denominator.value_counts()
df_img_predictions.describe()
df_tweet_data.describe()
#Create a copy of all the gathered dataframes

df_twitter_archive_copy = df_twitter_archive.copy()

df_img_predictions_copy = df_img_predictions.copy()

df_tweet_data_copy = df_tweet_data.copy()
#In the text the name always starts with a capital letter.

def extract_name_from_text(row):

    try:

        if 'This is' in row['text']:

            name = re.search('This is ([A-Z]\w+)',row['text']).group(1)

        elif 'Meet' in row.text:

            name = re.search('Meet ([A-Z]\w+)', row['text']).group(1)

        elif 'Say hello to' in row.text:

            name = re.search('Say hello to ([A-Z]\w+)', row['text']).group(1)

        elif 'named' in row.text:

            name = re.search('named ([A-Z]\w+)', row['text']).group(1)

        else:

            name = ''

    except AttributeError:

        name = ''

    return name
df_twitter_archive_copy['name'] = df_twitter_archive_copy.apply(extract_name_from_text, axis=1)
df_twitter_archive_copy.name.value_counts()
def extract_source(row):

    try:

        source = re.search('>(.+)</a>', row['source']).group(1)

    except AttributeError:

        source = ''

    return source
df_twitter_archive_copy['source'] = df_twitter_archive_copy.apply(extract_source, axis=1)

df_twitter_archive_copy['source'] = df_twitter_archive_copy.source.astype('category')
df_twitter_archive_copy.source.dtype
df_twitter_archive_copy.source.value_counts()
def extract_gender(row):

    if 'He' in row['text']:

        gender = 'M'

    elif 'She' in row['text']:

        gender = 'F'

    else:

        gender = ''

    return gender
df_twitter_archive_copy['gender'] = df_twitter_archive_copy.apply(extract_gender, axis=1)

df_twitter_archive_copy['gender'] = df_twitter_archive_copy.gender.astype('category')
df_twitter_archive_copy.gender.dtype
df_twitter_archive_copy.gender.value_counts()
def extract_hashtag(row):

    try:

        if '#' in row['text']:

            hashtag = re.search('#(\w+)[\s\.]', row['text']).group(1)

        else:

            hashtag = float('NaN')

    except AttributeError:

        hashtag = ''

    return hashtag



df_twitter_archive_copy['hashtag'] = df_twitter_archive_copy.apply(extract_hashtag, axis=1)
df_twitter_archive_copy.hashtag.value_counts()
def get_dog_stage(row):

    if 'doggo' in row['text'].lower():

        stage = 'doggo'

    elif 'floof' in row['text'].lower():

        stage = 'floofer'

    elif 'pupper' in row['text'].lower():

        stage = 'pupper'

    elif 'puppo' in row['text'].lower():

        stage = 'puppo'

    else:

        stage = ''

    return stage
df_twitter_archive_copy['stage'] = df_twitter_archive_copy.apply(get_dog_stage, axis=1)

df_twitter_archive_copy['stage'] = df_twitter_archive_copy.stage.astype('category')
df_twitter_archive_copy.drop(['doggo','pupper','floofer','puppo'], axis=1, inplace=True)
df_twitter_archive_copy.stage.value_counts()
df_twitter_archive_copy.stage.dtype
list(df_twitter_archive_copy)
breed = []

confidence = []



def get_breed_and_confidence(row):

    if row['p1_dog'] == True:

        breed.append(row['p1'])

        confidence.append(row['p1_conf'])

    elif row['p2_dog'] == True:

        breed.append(row['p2'])

        confidence.append(row['p2_conf'])

    elif row['p3_dog'] == True:

        breed.append(row['p3'])

        confidence.append(row['p3_conf'])

    else:

        breed.append('Not identified')

        confidence.append(np.nan)

        

df_img_predictions_copy.apply(get_breed_and_confidence, axis=1)

df_img_predictions_copy['breed'] = pd.Series(breed)

df_img_predictions_copy['confidence'] = pd.Series(confidence)

df_img_predictions_copy.drop(['p1','p1_conf','p1_dog','p2','p2_conf','p2_dog','p3','p3_conf','p3_dog'], axis=1, inplace=True)
df_img_predictions_copy.head()
df_img_predictions_copy.info()
df = pd.merge(df_twitter_archive_copy, df_img_predictions_copy, on='tweet_id')

df = df.merge(df_tweet_data_copy, on='tweet_id')
list(df)
df.head()
df.info()
df = df.query('breed != "Not identified"')
df.query('breed == "Not identified"').shape[0]
df.info()
df.drop(['in_reply_to_status_id', 'in_reply_to_user_id', 'retweeted_status_user_id', 'retweeted_status_id', 'retweeted_status_timestamp'], axis=1, inplace=True)
df.columns
df.tweet_id = df.tweet_id.to_string()

df.timestamp = pd.to_datetime(df.timestamp, yearfirst=True)
df.info()
df_twitter_archive_copy.rating_numerator.describe()
df['rating'] = df.rating_numerator/df.rating_denominator



#Use ratings to divide into categories

df['rating_category'] = pd.cut(df.rating, bins = [0.0, np.percentile(df.rating,25), np.percentile(df.rating,50), np.percentile(df.rating,75), np.max(df.rating)],labels=['Low','Below_average','Above_average','High'])



#Drop the unwanted columns

df.drop(['rating_numerator','rating_denominator'], axis=1, inplace=True)
df.rating_category.value_counts()
df.columns
df.loc[df['name'] == '', 'name'] = None

df.loc[df['gender'] == '', 'gender'] = None

df.loc[df['stage'] == '', 'stage'] = None

df.loc[df['breed'] == '', 'breed'] = None

df.loc[df['rating'] == 0.0, 'rating'] = np.nan

df.loc[df['rating'] == 0.0, 'rating_category'] = None
df.info()
#Store the final cleaned dataframe

df.to_csv('twitter_archive_master.csv', index=False)
df.gender.value_counts().plot(kind='bar');
df.source.value_counts().plot(kind='bar');
df.name.value_counts()[0:19].plot(kind='bar');
df.breed.value_counts()[0:19].plot(kind='bar');
#group by breed and store the means of retweet_count and favorite_count.

df_group = df.groupby(['breed'])['retweet_count', 'favorite_count'].mean()

#order by retweet_count and favorite_count.

df_group = df_group.sort_values(['retweet_count', 'favorite_count'], ascending=False)

#plot the top 15 average counts.

df_group.iloc[0:14,].plot(kind='bar');