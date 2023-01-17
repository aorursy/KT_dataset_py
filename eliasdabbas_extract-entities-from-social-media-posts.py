%config InlineBackend.figure_format = 'retina' # high resolution plotting

import matplotlib.pyplot as plt

import pandas as pd

import advertools as adv

pd.set_option('display.max_columns', None)

pd.set_option('display.max_colwidth', 280)

adv.__version__
tweets_users_df = pd.read_csv('../input/justdoit_tweets_2018_09_07_2.csv', )

print(tweets_users_df.shape)

tweets_users_df.head(3)
[x for x in dir(adv) if x.startswith('extract')]  # currently available extract functions
hashtag_summary = adv.extract_hashtags(tweets_users_df['tweet_full_text'])

hashtag_summary.keys()
hashtag_summary['overview']
hashtag_summary['hashtags'][:10]
hashtag_summary['hashtags_flat'][:10]
hashtag_summary['hashtag_counts'][:20]
hashtag_summary['hashtag_freq'][:15]
plt.figure(facecolor='#ebebeb', figsize=(11, 8))

plt.bar([x[0] for x in hashtag_summary['hashtag_freq'][:15]],

        [x[1] for x in hashtag_summary['hashtag_freq'][:15]])

plt.title('Hashtag frequency', fontsize=18)

plt.xlabel('Hashtags per tweet', fontsize=12)

plt.ylabel('Number of tweets', fontsize=12)

plt.xticks(range(16))

plt.yticks(range(0, 2100, 100))

plt.grid(alpha=0.5)

plt.gca().set_frame_on(False)
hashtag_summary['top_hashtags'][:10]
plt.figure(facecolor='#ebebeb', figsize=(8, 12))

plt.barh([x[0] for x in hashtag_summary['top_hashtags'][2:][:30]][::-1],

         [x[1] for x in hashtag_summary['top_hashtags'][2:][:30]][::-1])

plt.title('Top Hashtags')

plt.grid(alpha=0.5)

plt.gca().set_frame_on(False)
emoji_summary = adv.extract_emoji(tweets_users_df['tweet_full_text'])

emoji_summary.keys()
emoji_summary['overview']
emoji_summary['emoji'][:20]
emoji_summary['emoji_text'][:20]
emoji_summary['emoji_flat'][:10]
emoji_summary['emoji_flat_text'][:10]
list(zip(emoji_summary['emoji_flat'][:10], emoji_summary['emoji_flat_text'][:10]))
emoji_summary['emoji_counts'][:15]
emoji_summary['emoji_freq'][:15]
plt.figure(facecolor='#ebebeb', figsize=(8, 8))

plt.bar([x[0] for x in emoji_summary['emoji_freq'][:15]],

        [x[1] for x in emoji_summary['emoji_freq'][:15]])

plt.title('Emoji frequency', fontsize=18)

plt.xlabel('Emoji per tweet', fontsize=12)

plt.ylabel('Number of tweets', fontsize=12)

plt.grid(alpha=0.5)

plt.gca().set_frame_on(False)
emoji_summary['top_emoji'][:20]
emoji_summary['top_emoji_text'][:20]
plt.figure(facecolor='#ebebeb', figsize=(8, 8))

plt.barh([x[0] for x in emoji_summary['top_emoji_text'][:20]][::-1],

         [x[1] for x in emoji_summary['top_emoji_text'][:20]][::-1])

plt.title('Top Emoji')

plt.grid(alpha=0.5)

plt.gca().set_frame_on(False)
mention_summary = adv.extract_mentions(tweets_users_df['tweet_full_text'])

mention_summary.keys()
mention_summary['overview']
mention_summary['mentions'][:15]
mention_summary['mentions_flat'][:10]
mention_summary['mention_counts'][:20]
mention_summary['mention_freq'][:15]
plt.figure(facecolor='#ebebeb', figsize=(8, 8))

plt.bar([x[0] for x in mention_summary['mention_freq'][:15]],

        [x[1] for x in mention_summary['mention_freq'][:15]])

plt.title('Mention frequency', fontsize=18)

plt.xlabel('Mention per tweet', fontsize=12)

plt.ylabel('Number of tweets', fontsize=12)

plt.xticks(range(15))

plt.yticks(range(0, 2800, 200))

plt.grid(alpha=0.5)

plt.gca().set_frame_on(False)
mention_summary['top_mentions'][:10]
plt.figure(facecolor='#ebebeb', figsize=(8, 8))

plt.barh([x[0] for x in mention_summary['top_mentions'][:15]][::-1],

         [x[1] for x in mention_summary['top_mentions'][:15]][::-1])

plt.title('Top Mentions')

plt.grid(alpha=0.5)

plt.xticks(range(0, 1100, 100))

plt.gca().set_frame_on(False)
question_summary = adv.extract_questions(tweets_users_df['tweet_full_text'])
question_summary.keys()
question_summary['overview']
question_summary['question_mark_freq']
question_summary['top_question_marks'] # this is more interesting if you have questions in different languages where different question marks are used.
[(i,x) for i, x in  enumerate(question_summary['question_text']) if x][:15]
intense_summary = adv.extract_intense_words(tweets_users_df['tweet_full_text'], min_reps=3)
intense_summary['overview']
intense_summary['top_intense_words'][:20]
currency_summary = adv.extract_currency(tweets_users_df['tweet_full_text'])
currency_summary.keys()
currency_summary['overview']
currency_summary['top_currency_symbols']
[x for x in currency_summary['surrounding_text'] if x][:20]
word_summary = adv.extract_words(tweets_users_df['tweet_full_text'], 

                                 words_to_extract=['sport', 'football', 'athlet',],

                                 entire_words_only=False) # when set to False, it extracts the words and show how they appear within a larger word if any

                                                          # if set to True, is only extracts the exact words specified only if they appear as entire words
word_summary.keys()
word_summary['overview']
word_summary['top_words'][:20]
word_summary_politics = adv.extract_words(tweets_users_df['tweet_full_text'],

                                          ['politic', 'polic', 'trump', 'donald'])
word_summary_politics['overview']
word_summary_politics['top_words'][:20]
extracted_tweets =  (tweets_users_df[['tweet_full_text', 'user_screen_name', 'user_followers_count']]

 .assign(hashtags=hashtag_summary['hashtags'],

         hashcounts=hashtag_summary['hashtag_counts'],

         mentions=mention_summary['mentions'],

         mention_count=mention_summary['mention_counts'],

         emoji=emoji_summary['emoji'],

         emoji_text=emoji_summary['emoji_text'],

         emoji_count=emoji_summary['emoji_counts'],))

extracted_tweets.head()
word_freq_hash = adv.word_frequency(extracted_tweets['hashtags'].str.join(' '), 

                                    extracted_tweets['user_followers_count'].fillna(0))

word_freq_hash.head(10)

extracted_tweets[extracted_tweets['hashtags'].str.join(' ').str.contains('drjanegoodall|itstrue',case=False)]
word_freq_mention = adv.word_frequency(extracted_tweets['mentions'].str.join(' '), 

                                       extracted_tweets['user_followers_count'].fillna(0))

word_freq_mention.head(10)

word_freq_emoji = adv.word_frequency(extracted_tweets['emoji'].str.join(' '), 

                                       extracted_tweets['user_followers_count'].fillna(0))

word_freq_emoji.head(10)

[adv.emoji_dict.emoji_dict[k] for k in word_freq_emoji['word'][:10]]
word_freq_emoji[:10].assign(emoji_text=[adv.emoji_dict.emoji_dict[k] for k in word_freq_emoji['word'][:10]])