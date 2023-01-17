#importing requests to get url content from online storage

# import requests

#importing os to download the content on our physical storage

import os

#importing pandas to store data in dataframes

import pandas as pd

# import tweepy

import json

import numpy as np

import matplotlib.pyplot as plt



%matplotlib inline
# #storing source urls

# twitter_achived_basic_source_url = "https://d17h27t6h515a5.cloudfront.net/topher/2017/August/59a4e958_twitter-archive-enhanced/twitter-archive-enhanced.csv"

# twitter_dogs_bread_source_url = "https://d17h27t6h515a5.cloudfront.net/topher/2017/August/599fd2ad_image-predictions/image-predictions.tsv"

# source_urls = [{

#             "url" : twitter_achived_basic_source_url,

#             "file_name" : "twitter_archive_enhanced.csv"

#         },{

#             "url": twitter_dogs_bread_source_url,

#             "file_name":"image_predictions.tsv"

#         }]

# #loop and download files in root folder with the file_names

# for source in source_urls:

#     response = requests.get(source["url"])

#     with open(os.path.join("", source["file_name"]), mode='wb') as file:

#         file.write(response.content)

tweets_df = pd.read_csv('../input/tweet-csv/twitter_archive_enhanced.csv', dtype={"tweet_id": str})

tweets_df.head()
#importing tsv file as csv with tab delimeter

dog_bread_df = pd.read_csv('../input/tweet-csv/image_predictions.tsv', sep='\t', dtype={"tweet_id": str})

dog_bread_df.head()
# consumer_key = '*****'

# consumer_secret = '****'

# access_token = '****'

# access_secret = '****'



# auth = tweepy.OAuthHandler(consumer_key, consumer_secret)

# auth.set_access_token(access_token, access_secret)



# api = tweepy.API(auth)
# #Testing the api by using one tweet id.

# tweet = api.get_status('666447344410484738',tweet_mode='extended')

# tweet._json
# #just to test how to access the parameters i need

# tweet.retweet_count, tweet.favorite_count
unique_tweet_ids = np.union1d(tweets_df.tweet_id.unique(), dog_bread_df.tweet_id.unique())

unique_tweet_ids.size
# api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify =True)

# failed_process_tweets = {}

# with open('tweet_json.txt', 'w') as file:

#     file.write('[')

#     for tweet_id  in unique_tweet_ids:

#         try:

#             print(f"processing tweet_id: {tweet_id}")

#             tweet = api.get_status(tweet_id,tweet_mode='extended')

#             json.dump(tweet._json, file)

#             file.write(',')

#         except tweepy.TweepError as e:

#             print(f"failed to process tweet_id:{tweet_id}")

#             failed_process_tweets[tweet_id] = e

#             pass            

#     file.write('{}]')
tweet_extended_df = pd.read_json("../input/tweet-json/tweet_json.txt",dtype ={"id_str": str})[:-1]

tweet_extended_df.head()
all_df = tweets_df.set_index('tweet_id').join(tweet_extended_df.set_index('id_str'),lsuffix='_basic', rsuffix='_extended')

all_df = all_df.join(dog_bread_df.set_index('tweet_id'),lsuffix='', rsuffix='_breed')

all_df.info()
all_df.index.duplicated().any()
all_df.replace('None', np.nan, inplace=True)

all_df.info()
all_df[['timestamp', 'created_at']].sample(5)
all_df[['source_basic', 'source_extended']].sample(5)
all_df.source_basic.unique(), all_df.source_extended.unique()
all_df.id.map(lambda x: '{:.0f}'.format(x)).sample(5)
all_df.drop(columns=['contributors', 'coordinates', 'geo', 'place', 'in_reply_to_status_id_basic', 

                     'in_reply_to_user_id_basic', 'in_reply_to_screen_name', 'in_reply_to_status_id_extended', 

                     'in_reply_to_status_id_str', 'in_reply_to_user_id_extended', 'in_reply_to_user_id_str', 

                     'quoted_status', 'quoted_status_id','is_quote_status', 'quoted_status_id_str', 'quoted_status_permalink', 

                     'retweeted_status_id', 'retweeted_status_user_id', 'retweeted_status_id', 'retweeted_status',

                     'retweeted_status_timestamp'], inplace=True)
all_df.drop(columns=['created_at','source_extended','id','expanded_urls'], inplace=True)

all_df.sample(5)
all_df.info()
all_df[['doggo', 'floofer', 'pupper', 'puppo']].sample(10)
all_df.doggo.unique(), all_df.floofer.unique(), all_df.pupper.unique(),all_df.puppo.unique()
all_df.source_basic.replace('<a href="http://twitter.com/download/iphone" rel="nofollow">Twitter for iPhone</a>','iPhone', inplace=True)

all_df.source_basic.replace('<a href="http://twitter.com" rel="nofollow">Twitter Web Client</a>','Twitter', inplace=True)

all_df.source_basic.replace('<a href="http://vine.co" rel="nofollow">Vine - Make a Scene</a>','Vine', inplace=True)

all_df.source_basic.replace('<a href="https://about.twitter.com/products/tweetdeck" rel="nofollow">TweetDeck</a>','Tweetdeck', inplace=True)

all_df.rename(columns={"source_basic": "source"}, inplace=True)
all_df.timestamp = pd.to_datetime(all_df["timestamp"])

all_df.rename(columns={'timestamp':"tweet_date"}, inplace=True)
all_df.doggo.replace(np.nan, False,inplace=True)

all_df.floofer.replace(np.nan, False,inplace=True)

all_df.pupper.replace(np.nan, False,inplace=True)

all_df.puppo.replace(np.nan, False,inplace=True)



all_df.doggo.replace('doggo', True,inplace=True)

all_df.floofer.replace('floofer', True,inplace=True)

all_df.pupper.replace('pupper', True,inplace=True)

all_df.puppo.replace('puppo', True,inplace=True)
all_df[['doggo','floofer','pupper','puppo']].sum(axis=1).unique()
all_df['multiple_age_stage'] = all_df[['doggo','floofer','pupper','puppo']].sum(axis=1) == 2

multiple_age_stage = all_df.query('multiple_age_stage == True')[['text', 'doggo','floofer','pupper','puppo']]

multiple_age_stage.shape
multiple_age_stage.to_csv('multiple_age_stage.csv')
all_df.rating_numerator.unique(), all_df.rating_denominator.unique()
rating_denominator_no_10 = all_df.query('rating_denominator != 10')[['text','rating_numerator', 'rating_denominator']]

rating_denominator_no_10
rating_numerator_above_15 = all_df.query('rating_denominator == 10 and rating_numerator > 15')[['text','rating_numerator', 'rating_denominator']]

rating_numerator_above_15
visual_ratings_checkup_df = pd.concat([rating_denominator_no_10,rating_numerator_above_15])

visual_ratings_checkup_df.to_csv('visual_ratings_checkup.csv')
rating_fixed = pd.read_csv('../input/visual-fixing/visual_ratings_checkup_fix.csv', dtype={"tweet_id": str})

not_dogs = rating_fixed.query('number_of_dogs != number_of_dogs').tweet_id

all_df = all_df.query(f'tweet_id not in {not_dogs.tolist()}')
dogs = rating_fixed.query('number_of_dogs == number_of_dogs')

dogs = dogs.set_index('tweet_id')

all_df_fixed = all_df.join(dogs[['rating_numerator', 'rating_denominator', 'number_of_dogs']], rsuffix='_fixed')

all_df_fixed['rating_numerator'] = all_df_fixed["rating_numerator_fixed"].fillna(all_df_fixed["rating_numerator"]).astype(int)

all_df_fixed['rating_denominator'] = all_df_fixed["rating_denominator_fixed"].fillna(all_df_fixed["rating_denominator"]).astype(int)

all_df_fixed.rating_denominator.unique()
all_df_fixed.rating_numerator.unique()
all_df = all_df_fixed.drop(columns=['rating_numerator_fixed','rating_denominator_fixed'])
all_df['rating'] = (all_df.rating_numerator / all_df.rating_denominator).astype(float) 
all_df.rating.describe()
all_df.query('rating < 177').rating.describe()
all_df = all_df.query('rating < 177').drop(columns=['rating_numerator','rating_denominator'])
age_stage = pd.read_csv('../input/visual-fixing/multiple_age_stage_fix.csv',dtype={"tweet_id": str})

not_dogs = age_stage.query('is_dog==False').tweet_id

all_df = all_df.query(f'tweet_id not in {not_dogs.tolist()}')
dogs = age_stage.query('is_dog == True')

dogs = dogs.set_index('tweet_id')

all_df_fixed = all_df.join(dogs, rsuffix='_fixed')

all_df_fixed['doggo'] = all_df_fixed["doggo_fixed"].fillna(all_df_fixed["doggo"])

all_df_fixed['floofer'] = all_df_fixed["floofer_fixed"].fillna(all_df_fixed["floofer"])

all_df_fixed['pupper'] = all_df_fixed["pupper_fixed"].fillna(all_df_fixed["pupper"])

all_df_fixed['puppo'] = all_df_fixed["puppo_fixed"].fillna(all_df_fixed["puppo"])

all_df = all_df_fixed.drop(columns=['doggo_fixed', 'floofer_fixed','pupper_fixed','puppo_fixed','is_dog','text_fixed'])
all_df = all_df[all_df[['doggo','floofer','pupper','puppo']].sum(axis=1) != 2]
def age_stage_process(row):

    if row.doggo:

        return 'doggo'

    elif row.floofer:

        return 'floofer'

    elif row.pupper:

        return 'pupper'

    elif row.puppo:

        return 'puppo'

    else:

        return np.NaN

    

all_df['age_stage'] = all_df.apply(age_stage_process,axis=1)
all_df.drop(columns=['doggo','floofer','pupper','puppo'], inplace=True)
all_df['favorite_count'].unique(), all_df['retweet_count'].unique()
all_df.sample(1).entities.tolist()
all_df.sample(1).extended_entities.tolist()
all_df.favorited.unique(), all_df.retweeted.unique(), all_df.possibly_sensitive.unique(), all_df.possibly_sensitive_appealable.unique()
all_df.query('favorite_count != favorite_count')
all_df.query('favorite_count != favorite_count and p1 != p1')
all_df.drop(columns = ["multiple_age_stage", "number_of_dogs", "display_text_range", "entities", "extended_entities", 

                       "favorited", "full_text", "lang","retweeted", "truncated", "user", "jpg_url", "img_num", 

                       "possibly_sensitive", "possibly_sensitive_appealable"], inplace=True)
missing_values = all_df.query('favorite_count != favorite_count and p1 != p1').index.tolist()

missing_values
all_df = all_df.query(f'tweet_id not in {missing_values}')
missing_values = all_df.query('favorite_count != favorite_count').index.tolist()

missing_values
all_df = all_df.query(f'tweet_id not in {missing_values}')
all_df.info()
def bread_extraction(row):

    bread_conf = 0

    bread_name = ''

    is_bread = False

    if(row.p1_dog == True):

        if(row.p1_conf >= bread_conf):

            bread_conf = row.p1_conf

            bread_name = row.p1

            is_bread = True

    if(row.p2_dog == True):

        if(row.p2_conf >= bread_conf):

            bread_conf = row.p2_conf

            bread_name = row.p2

            is_bread = True

    if(row.p3_dog == True):

        if(row.p2_conf >= bread_conf):

            bread_conf = row.p3_conf

            bread_name = row.p3

            is_bread = True

    if is_bread == False:

        return np.nan

    else:

        return bread_name

        

        

    

all_df['bread'] = all_df.apply(bread_extraction,axis=1)
all_df.drop(columns=['p1','p2','p3','p1_dog','p2_dog','p3_dog','p1_conf', 'p2_conf', 'p3_conf'],inplace=True)
all_df.info()
all_df.bread.unique()
all_df[all_df.name.str.lower() == all_df.name].name.unique()
all_df[all_df.name.str.lower() == all_df.name].sample(10).text.tolist()
all_df['name'] = all_df['name'].apply(lambda x: x if str(x).lower() != x else np.nan)
all_df[all_df.name.str.lower() == all_df.name].name.unique()
all_df.name.unique()
all_df.info()
all_df.to_csv('twitter_archive_master.csv')
df = pd.read_csv('twitter_archive_master.csv')
named_dogs = df.query('name == name')

named_dogs_grouped = named_dogs.groupby('name').count()[['tweet_id']]

named_dogs_grouped.rename(columns={'tweet_id':'name_count'}, inplace=True)

named_dogs_grouped.query('name_count >= 8').sort_values(by=['name_count']).plot.bar()

plt.ylim(top=12)

plt.title("Most Popular Dog Names",{'fontsize': 20},pad=20)

plt.xlabel("Dogs Names")

plt.legend(["Dogs Names Frequencey"])
df[df.favorite_count == df.favorite_count.max()]
df[df.retweet_count == df.retweet_count.max()]
df.query('tweet_id == "744234799360020481"').text.tolist()
tweet_sources = df.groupby('source').count()[['tweet_id']]

tweet_sources.rename(columns={'tweet_id': 'source_count'}, inplace=True)

tweet_sources['source_percentage'] = tweet_sources.source_count / tweet_sources.source_count.sum() * 100

tweet_sources['source_percentage'].plot.pie(figsize=(10,8), autopct='%1.1f%%',

        explode=(0,0,0,0.1))

plt.title("Source of Tweets", {'fontsize': 20})

plt.legend(["Tweetdeck", "Twitter", "Vine", "iPhone"])

plt.ylabel("")
bread_ratings = df.query('bread == bread')[['rating', 'bread']].groupby('bread').mean() * 10

bread_ratings.hist()
bread_ratings.sort_values(by=['rating']).tail(5).plot.bar(figsize=(10,5))

plt.ylim(top=14)

plt.title("Top 5 Rated Breads",{'fontsize': 20},pad=20)

plt.xlabel("Breads")

plt.legend(["Average Ratings"])
bread_ratings.sort_values(by=['rating']).head(5).plot.bar(figsize=(10,5))

plt.ylim(top=14)

plt.title("Least 5 Rated Breads",{'fontsize': 20},pad=20)

plt.xlabel("Breads")

plt.legend(["Average Ratings"])
from subprocess import call

call(['python', '-m', 'nbconvert', 'wrangle_act.ipynb'])