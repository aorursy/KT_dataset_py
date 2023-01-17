import pandas as pd
pd.set_option('display.max_columns', None)
import advertools as adv
import os
os.listdir('../input')
# import advertools as adv
# adv.twitter.set_auth_params(app_id='YOUR_APP_ID', app_secret='YOUR_APP_SECRET', 
#                            oauth_token='YOUR_OAUTH_TOKEN', oauth_token_secret='YOUR_OAUTH_TOKEN_SECRET')
# ama = adv.twitter.search(q='#AMAs -filter:retweets', count=5000, tweet_mode='extended', include_entities=True)
amas = pd.read_csv('../input/amas_combined.csv', low_memory=False,
                   parse_dates=['tweet_created_at', 'user_created_at'])
print(amas.shape)
amas.head(3)
print('Columns containing "tweet_":', amas.columns.str.contains('tweet_').sum())
print('Columns containing "user_":', amas.columns.str.contains('user_').sum())
amas.filter(regex='_count', axis=1).head()
amas.select_dtypes(bool).head()
amas[['tweet_full_text', 'user_description']].sample(20)
(amas
 .sort_values(['tweet_retweet_count'], ascending=False)
 [['tweet_full_text', 'user_screen_name', 'user_followers_count', 'tweet_retweet_count']]
 .head(20))
word_freq = adv.word_frequency(amas['tweet_full_text'], 
                               amas['user_followers_count'], 
                               rm_words=adv.stopwords['english'])
word_freq.head(20)
word_freq.sort_values(['abs_freq'], ascending=False).head(20)
hashtags = adv.extract_hashtags(amas['tweet_full_text'])
hashtags['overview']
hashtags['top_hashtags'][:20]
mentions = adv.extract_mentions(amas['tweet_full_text'])

mentions['overview']
mentions['top_mentions'][:20]
emoji = adv.extract_emoji(amas['tweet_full_text'])
emoji['overview']
emoji['top_emoji'][:20]
emoji['top_emoji_text'][:20]
import unicodedata
unicodedata.name('❤'), unicodedata.name('🤔')
trends = pd.read_csv('../input/trends_combined.csv')
trends.sample(10)
trends[['location', 'name', 'tweet_volume', 'time']].sample(15)
print('Number of locations:', trends['location'].nunique())
trends['location'].unique()
winners = pd.read_csv('../input/categories_winners.csv')
winners
retweeters = pd.read_csv('../input/retweeters_ids.csv')
retweeters.head()