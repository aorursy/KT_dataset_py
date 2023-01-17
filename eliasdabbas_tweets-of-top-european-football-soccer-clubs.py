!pip install advertools==0.7.4 plotly==4.0.0
from IPython.display import clear_output

clear_output()

import advertools as adv

import pandas as pd

pd.options.display.max_columns = None

import plotly.graph_objects as go

import plotly



print('Package     Version')

print('=' * 20)

for pack in [pd, adv, plotly]:

    print(f'{pack.__name__:<10}', ':', pack.__version__)
# clubs = pd.read_html('https://en.wikipedia.org/wiki/List_of_UEFA_club_competition_winners')[1]

# clubs.to_csv('clubs.csv', index=False)
clubs = pd.read_csv('../input/clubs.csv')

clubs.head(7)
handles = [

    'realmadrid',

    'acmilan',

    'FCBarcelona',

    'LFC',

    'juventusfc',

    'FCBayern',

    'AFCAjax',

]
handles_clubs = dict(zip(handles, clubs['Club'][:7]))

handles_clubs
auth_params = {

    'app_key': 'YOUR_APP_KEY',

    'app_secret': 'YOUR_APP_SECRET',

    'oauth_token': 'YOUR_OAUTH_TOKEN',

    'oauth_token_secret': 'YOUR_OAUTH_TOKEN_SECRET',

}



adv.twitter.set_auth_params(**auth_params)
# user_dfs = []



# for club in handles:

#     df = adv.twitter.lookup_user(screen_name=club, tweet_mode='extended')

#     user_dfs.append(df)



# user_dfs = pd.concat(user_dfs, sort=False)

# user_dfs.to_csv('user_dfs.csv', index=False)
user_dfs = pd.read_csv('../input/user_dfs.csv')

user_dfs.head(3)
(user_dfs

 .sort_values('followers_count', ascending=False)

 [['created_at', 'screen_name', 'followers_count', 'statuses_count']]

 .style.format({'followers_count': '{:,}','statuses_count':  '{:,}'}))
club_handles = user_dfs['screen_name'].tolist()

mentioned_handles = adv.extract_mentions(user_dfs['description'])['mentions_flat']

mentioned_handles = [m.replace('@', '') for m in mentioned_handles]

all_handles = sorted(club_handles + mentioned_handles)

print('number of accounts:', len(all_handles))

print('sample:')

all_handles[:5] + all_handles[-5:]
# clubs_tweets_dfs = []

# for acct in all_handles:

#     df = adv.twitter.get_user_timeline(screen_name=acct, count=3500, tweet_mode='extended')

#     clubs_tweets_dfs.append(df)

# (pd.concat(clubs_tweets_dfs, sort=False, ignore_index=True)

#  .to_csv('clubs_tweets.csv', index=False))
club_tweets = pd.read_csv('../input/clubs_tweets.csv', parse_dates=['tweet_created_at', 'user_created_at'],

                          low_memory=False)

print(club_tweets.shape)

club_tweets.head(2)
category_cols = club_tweets.columns[club_tweets.nunique().lt(250)] 

dtypes = dict(zip(category_cols, ['category' for i in range(len(category_cols))]))

club_tweets = club_tweets.astype(dtypes)
main_sub_accts = dict(zip(user_dfs['screen_name'].values, 

                          adv.extract_mentions(user_dfs['description'])['mentions'])) 

for acct, subacct in main_sub_accts.items():

    print(acct, ':',  *subacct, sep=' ')

    print()
from collections import defaultdict

dd = defaultdict()



for k, v in main_sub_accts.items():

    for val in v:

        dd[val.replace('@', '').lower()] = handles_clubs[k]

        dd[k.lower()] = handles_clubs[k]

dd
club_tweets['club_name'] = club_tweets['user_screen_name'].str.lower().map(dd)
club_tweets[['user_screen_name', 'club_name']].sample(10)
(club_tweets

 [['user_screen_name', 'tweet_created_at', 'club_name', 'user_statuses_count']]

 .groupby(['club_name', 'user_screen_name'])

 .agg({'tweet_created_at': ['min', 'max'], 'user_statuses_count': 'count'})

 .assign(date_range=lambda df: df[('tweet_created_at', 'max')] - 

         df[('tweet_created_at', 'min')])

 .sort_values('date_range'))
weekyl_count = (club_tweets

               .set_index('tweet_created_at')

               .resample('W')['tweet_full_text']

               .count())

weekyl_count.head()
fig = go.Figure()

fig.add_bar(x=weekyl_count.index, y=weekyl_count.values)

fig.layout.title = 'Number of Weekly Tweets for Top European Football Clubs\' Twitter Accounts'

fig.layout.paper_bgcolor = '#E5ECF6'

fig.show()
from plotly.subplots import make_subplots

club_names = list(club_tweets['club_name'].unique())

fig = make_subplots(rows=7, cols=1, x_title='Week', shared_xaxes=True,

                    y_title='Number of Tweets', subplot_titles=club_names)

for i, club in enumerate(club_names):

        weekly = (club_tweets[club_tweets['club_name']==club]

                  .set_index('tweet_created_at').resample('W')['tweet_full_text'].count())

        fig.add_bar(x=weekly.index, y=weekly.values,

                    showlegend=False,

                    marker={'line': {'color': '#000000'}},

                    row=i+1, col=1)

fig.layout.title = 'Number of Weekly Tweets by Club (last 3,200 tweets)'

fig.layout.paper_bgcolor = '#E5ECF6'

fig.layout.height = 750

fig
(club_tweets['tweet_lang']

 .value_counts()

 .to_frame()

 .assign(perc=lambda df: df['tweet_lang'].div(df['tweet_lang'].sum()),

         cum_perc=lambda df: df['perc'].cumsum())

 .head(10)

 .style.format({'tweet_lang': '{:,}', 'perc': '{:.1%}', 'cum_perc': '{:.1%}'}))
tweets_by_wkday = (club_tweets

                   .groupby(club_tweets['tweet_created_at']

                           .dt.weekday_name)['tweet_full_text']

                   .count().to_frame().sort_values('tweet_full_text'))

fig = go.Figure()

fig.add_bar(y=tweets_by_wkday.index, x=tweets_by_wkday['tweet_full_text'], 

            orientation='h')

fig.layout.title = 'Number of Tweets per Day of Week'

fig.layout.xaxis.title = 'Number of Tweets'

fig.layout.paper_bgcolor = '#E5ECF6'

fig
import plotly.express as xp

real_madrid = club_tweets[club_tweets['user_screen_name']=='realmadrid'].copy()

real_madrid['month'] = [pd.Period(x, freq='M') for x in real_madrid['tweet_created_at']]

real_madrid.loc[:,'month'] = real_madrid['month'].astype('str')



fig = xp.scatter(real_madrid[::-1], x='tweet_created_at',

                 y='tweet_retweet_count', 

                 title='@realmadrid Tweets - Monthly',

                 color='tweet_lang', opacity=0.6,

                 template='plotly_white',

                 animation_frame='month',

                 hover_data=['tweet_full_text'])

fig.layout.yaxis.title = 'Tweet Retweet Count'

fig.layout.xaxis.title = 'Tweet Creation Date'

fig.show()
hashtag_summary = adv.extract_hashtags(club_tweets['tweet_full_text'])

hashtag_summary['overview']
hashtag_summary['top_hashtags'][:15]
fig = go.FigureWidget()

fig.add_bar(x=[h[1] for h in hashtag_summary['top_hashtags'][:20][::-1]],

            y=[h[0] for h in hashtag_summary['top_hashtags'][:20][::-1]], orientation='h')

fig.layout.height = 800

fig.layout.title = 'Top Hashtags Used By All Clubs'

fig.layout.paper_bgcolor = '#E5ECF6'

fig.layout.xaxis.title = 'Number of times the hashtag was used'

fig.show()

(club_tweets

 .drop_duplicates('user_screen_name')

 .sort_values('user_followers_count', ascending=False)

 [['user_screen_name', 'user_followers_count']]

 .head(15)

 .reset_index(drop=True)

 .style.format({'user_followers_count': '{:,}'}))

top_5 = (club_tweets

         .drop_duplicates('user_screen_name')

         .sort_values('user_followers_count', ascending=False)

         ['user_screen_name']

         .head(5).tolist())

top_5
titles = []

for title in [['@' + club + ' Wtd. Freq', '@' + club + ' Absolute Freq'] for club in top_5]:

    titles.append(title[0])

    titles.append(title[1])

titles
fig = make_subplots(rows=5, cols=2, subplot_titles=titles)



for i, club in enumerate(top_5):

    df = club_tweets[club_tweets['user_screen_name']==club]

    hashtag_df = adv.word_frequency(df['tweet_full_text'], df['tweet_retweet_count'], 

                                    regex=adv.regex.HASHTAG_RAW)

    fig.add_bar(y=hashtag_df['word'][:7][::-1],

                x=hashtag_df['wtd_freq'][:7][::-1], orientation='h',

                row=i+1, col=1, showlegend=False,

                marker={'color': plotly.colors.DEFAULT_PLOTLY_COLORS[i]})

    fig.add_bar(y=hashtag_df.sort_values('abs_freq', ascending=False)['word'][:7][::-1], 

                x=hashtag_df.sort_values('abs_freq', ascending=False)['abs_freq'][:7][::-1],

                orientation='h', 

                row=i+1, col=2, showlegend=False,

                marker={'color': plotly.colors.DEFAULT_PLOTLY_COLORS[i]})



fig.layout.height = 1200

fig.layout.paper_bgcolor = '#E5ECF6'

fig.layout.title = ('<i>Top Hashtags by Twitter Account - Weighted by Number of Retweets</i><br>' +

                    '<b>Wtd. Freq:</b> number of hashtags times total retweets of tweets containing the hashtag<br>' +

                    '<b>Absolute Freq:</b> Simple count showing the number of times a hashtag was used<br>')

fig.layout.margin = {'t': 180, 'r': 10}

fig
fig = make_subplots(rows=5, cols=2, subplot_titles=titles)

for i, club in enumerate(top_5):

    df = club_tweets[club_tweets['user_screen_name']==club]

    mention_df = adv.word_frequency(df['tweet_full_text'], df['tweet_retweet_count'], 

                                    regex=adv.regex.MENTION_RAW)

    fig.add_bar(y=mention_df['word'][:7][::-1],

                x=mention_df['wtd_freq'][:7][::-1], orientation='h',

                row=i+1, col=1, showlegend=False,

                marker={'color': plotly.colors.DEFAULT_PLOTLY_COLORS[i+5]})

    fig.add_bar(y=mention_df.sort_values('rel_value', ascending=False)['word'][:7][::-1], 

                x=mention_df.sort_values('abs_freq', ascending=False)['abs_freq'][:7][::-1],

                orientation='h', 

                row=i+1, col=2, showlegend=False,

                marker={'color': plotly.colors.DEFAULT_PLOTLY_COLORS[i+5]})



fig.layout.height = 1200

fig.layout.paper_bgcolor = '#E5ECF6'

fig.layout.title = ('<i>Top Mentions by Twitter Account - Weighted by Number of Retweets</i><br>' +

                    '<b>Wtd. Freq:</b> number of mentions times total retweets of tweets containing the mention<br>' +

                    '<b>Absolute Freq:</b> Simple count showing the number of times a mention was used<br>')

fig.layout.margin = {'t': 180, 'r': 10}

fig
emoji_summary = adv.extract_emoji(club_tweets['tweet_full_text'])

emoji_summary['overview']
emoji_summary.keys()
currency_summary = adv.extract_currency(club_tweets['tweet_full_text'])

print(currency_summary.keys())

print()

currency_summary['overview']
[t for t in currency_summary['surrounding_text'] if t][:20]
from collections import Counter

Counter(currency_summary['currency_symbols_flat'])

intensity_summary = adv.extract_intense_words(club_tweets['tweet_full_text'])

print(intensity_summary.keys())

print()

intensity_summary['overview']
intensity_summary['top_intense_words'][:30]
club_tweets[club_tweets['tweet_full_text'].str.contains('go+a?l', case=False)].__len__()
question_summary = adv.extract_questions(club_tweets['tweet_full_text'])

print(question_summary.keys())

print()

question_summary['overview']
[q for q in question_summary['question_text'] if q][:20]
exclamation_summary = adv.extract_exclamations(club_tweets['tweet_full_text'])

print(exclamation_summary.keys())

print()

exclamation_summary['overview']
[x for x in exclamation_summary['exclamation_text'] if x][:10]
emoji_freq =  adv.word_frequency(club_tweets['tweet_full_text'], 

                                 club_tweets['user_followers_count'],

                                 regex=adv.emoji.EMOJI_RAW)

emoji_freq.head(20).style.format({'abs_freq': '{:,}', 'wtd_freq': '{:,}', 'rel_value': '{:,.0f}'})
print('tweets containing 🔴:')

(club_tweets

 [club_tweets['tweet_full_text'].str.contains('🔴')]

 .filter(regex='tweet_favorite_count|tweet_retweet_count')

 .describe()

 .style.format('{:,.2f}'))
print('tweets NOT containing 🔴:')

(club_tweets

 [~club_tweets['tweet_full_text'].str.contains('🔴')]

 .filter(regex='tweet_favorite_count|tweet_retweet_count')

 .describe()

 .style.format('{:,.2f}'))
(club_tweets

 [club_tweets['tweet_full_text'].str.contains('🔴')]

 ['club_name'].value_counts()

 .reset_index().style.format({'club_name': '{:,}'}))
(club_tweets

 [club_tweets['tweet_full_text'].str.contains('🔴')]

 ['user_screen_name'].value_counts()

 .head(8)

 .reset_index().style.format({'club_name': '{:,}'}))
print('Barcelona tweets containing 🔴:\n')

(club_tweets

 [club_tweets['tweet_full_text'].str.contains('🔴') & 

  (club_tweets['club_name'] == 'Barcelona')]

 .filter(regex='tweet_favorite_count|tweet_retweet_count')

 .describe()

 .style.format('{:,.2f}'))
print('Barcelona tweets NOT containing 🔴:\n')

(club_tweets

 [~club_tweets['tweet_full_text'].str.contains('🔴') & 

  (club_tweets['club_name'] == 'Barcelona')]

 .filter(regex='tweet_favorite_count|tweet_retweet_count')

 .describe()

 .style.format('{:,.2f}'))
pd.options.display.max_colwidth = 280



for x in (club_tweets[(club_tweets['club_name']=='Barcelona') & 

                      club_tweets['tweet_full_text'].str.contains('🔴')]

          [['tweet_full_text']][:10].values):

    print(*x)

    print('='*30, '\n')