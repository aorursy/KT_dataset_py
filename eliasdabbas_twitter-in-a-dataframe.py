import pandas as pd

pd.set_option('display.max_colwidth', 60)

pd.set_option('display.max.columns', None) # 70+ columns to explore! 
# get these from your dashboard on developer.twitter.com: 

auth_params = {

    'app_key': 'YOUR_APP_KEY',

    'app_secret': 'YOUR_APP_SECRET',

    'oauth_token': 'YOUR_OAUTH_TOKEN',

    'oauth_token_secret': 'YOUR_OAUTH_TOKEN_SECRET',

}



# twython:

from twython import Twython

twitter = Twython(**auth_params) 

# twitter.search(q='basketball') 

# or any other function and / or parameters 



# advertools: 

import advertools as adv

adv.twitter.set_auth_params(**auth_params)

# adv.twitter.get_user_timeline(screen_name='twitter') 

# or some other function
# python = adv.twitter.search(q='#python', count=1000, tweet_mode='extended', lang='en')

# python.to_csv('python_tweets.csv', index=False) 
python = pd.read_csv('../input/python_tweets.csv', 

                     parse_dates=['tweet_created_at', 'user_created_at'])

print(python.shape)

python.head(3)
print('Columns starting with "tweet_" :', python.columns.str.contains('tweet_').sum()) 

print('Columns starting with "user_" :', python.columns.str.contains('user_').sum()) 
python[['tweet_created_at', 'user_created_at']].dtypes
# we actually need to deduplicate users, but just for a quick demo:

python['user_created_at'].dt.year.lt(2010).sum()
python['tweet_created_at'][0]
(python

 .sort_values(['user_followers_count'], ascending=False)

 [['tweet_full_text', 'user_screen_name', 

   'user_followers_count', 'user_description']]

 .head(10))
# number of rows - number of duplicates:

python.shape[0] - python['user_id'].duplicated().sum()
python['user_screen_name'].value_counts().head(10)
python['user_screen_name'].value_counts().cumsum().head(10)
python['tweet_entities'][0]
python['tweet_entities'][1]
python.filter(regex='count', axis=1).head() # user ID and tweet ID are integers but not 'numeric' in this sense
print('Boolean columns: ')

python_bool = python.select_dtypes(bool)

python_bool.head()
(python_bool

 .mean()

 .to_frame().T

 .rename(axis=0, mapper={0: 'mean:'})

 .style.format("{:.2%}"))
python['tweet_id'].duplicated().sum()
(python['user_description']

 .drop_duplicates()

 .str.contains('python')

 .sum())
(python['user_description']

 .drop_duplicates()

 .str.contains('django|flask|web')

 .mean().round(3))
(python['user_description']

 .drop_duplicates()

 .str.contains('data|mining|machine learning|ai ')

 .mean().round(3))
(python['user_description']

 .drop_duplicates()

 .str.contains('developer|development|programming')

 .mean().round(3))
(python['user_description']

 .drop_duplicates()

 .str.contains('developer|development|programming|data|django|flask|web|python|machine learning|ai |mining')

 .mean().round(3))
python['user_description'].isna().sum()
(python

 .drop_duplicates(subset=['user_id'])['user_verified']

 .apply(['mean', 'count']))
python['tweet_source'].value_counts().head(10)
print('Number of unique apps:', python['tweet_source_url'].nunique())

python['tweet_source_url'].value_counts().head(10)
adv.word_frequency(text_list=python['tweet_full_text'], 

                   num_list=python['user_followers_count']).head(15)
(adv.word_frequency(text_list=python['tweet_full_text'], 

                    num_list=python['user_followers_count'])

 .sort_values(['abs_freq'], ascending=False)

 .head(15))
# rate_limit = adv.twitter.get_application_rate_limit_status(consumed_only=False)

# rate_limit.to_csv('data/app_rate_limit_status.csv', index=False)
rate_limit = pd.read_csv('../input/app_rate_limit_status.csv')

rate_limit.sample(10)
# adv.twitter.get_application_rate_limit_status()
# available_trends = adv.twitter.get_available_trends()

# available_trends.to_csv('data/available_trends.csv', index=False)
available_trends = pd.read_csv('../input/available_trends.csv')

print(available_trends.shape)

available_trends.sample(15)
(available_trends

 [available_trends['name'].duplicated(keep=False)]

 .sort_values(['name']))
spain_ids = available_trends.query('country == "Spain"')

spain_ids
# spain_trends = adv.twitter.get_place_trends(ids=spain_ids['woeid'])

# spain_trends.to_csv('data/spain_trends.csv', index=False)
spain_trends = pd.read_csv('../input/spain_trends.csv')

print(spain_trends.shape)

spain_trends.sample(15)
spain_trends[['name', 'tweet_volume', 'location', 'time']].sample(15)
(spain_trends[['name', 'tweet_volume', 'location', 'time']]

 .groupby(['location']).head(2)) # this works because data are sorted based on tweet_volume, 

                                 # otherwise you have to sort
spain_trends.query('location=="Madrid"')[['name','tweet_volume','location', 'time']]
# twtr_favs = adv.twitter.get_favorites(screen_name='twitter', 

#                                       count=3000, tweet_mode='extended')

# twtr_favs.to_csv('data/twitter_favorites.csv', index=False)
twtr_favs = pd.read_csv('../input/twitter_favorites.csv')

twtr_favs.head()
(twtr_favs

 .user_screen_name

 .value_counts(normalize=False)

 .head(10))
(twtr_favs

 .user_screen_name

 .value_counts(normalize=True)

 .cumsum()

 .head(30))
# kaggle_followers = adv.twitter.get_followers_ids(screen_name='kaggle', count=5000)



# import json

# with open('data/kaggle_follower_ids.json', 'wt') as file:

#     json.dump(kaggle_followers, file)
import json

with open('../input/kaggle_follower_ids.json', 'rt') as file:

    kaggle_follower_ids = json.load(file)
print(kaggle_follower_ids.keys(), '\n')

print('previous cursor: ', kaggle_follower_ids['previous_cursor'])

print('next cursor: ', kaggle_follower_ids['next_cursor'])

print('Follower IDs:', kaggle_follower_ids['ids'][:10])

print('List length:', len(kaggle_follower_ids['ids']))
# the_psf = adv.twitter.get_followers_list(screen_name='ThePSF', count=2500, 

                                        # include_user_entities=True)

# the_psf.to_csv('data/the_psf_followers.csv', index=False)
the_psf = pd.read_csv('../input/the_psf_followers.csv')

print(the_psf.shape)

the_psf.head()
# kaggle_list_memberships = adv.twitter.get_list_memberships(screen_name='kaggle', count=500)

# kaggle_list_memberships.to_csv('data/kaggle_list_memberships.csv', index=False)
kaggle_list_memberships = pd.read_csv('../input/kaggle_list_memberships.csv')

kaggle_list_memberships.shape
kaggle_list_memberships.head()
# languages = adv.twitter.get_supported_languages()

# languages.to_csv('data/languages.csv', index=False)
languages = pd.read_csv('../input/languages.csv')

print(languages.shape)

languages.head()
# search_operators = pd.read_html('https://developer.twitter.com/en/docs/tweets/search/guides/standard-operators', 

#                                header=0)[0]

# search_operators.to_csv('data/twitter_search_operators.csv', index=False)
search_operators = pd.read_csv('../input/twitter_search_operators.csv')

pd.set_option('display.max_colwidth', 160)

search_operators