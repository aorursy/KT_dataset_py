#Load the boring stuff
import sqlite3
import pandas
import numpy
from matplotlib import pyplot as plt
from scipy.stats import gaussian_kde
from sqlalchemy import create_engine
from scipy.interpolate import UnivariateSpline as spline
reddit_conn = sqlite3.connect("../input/database.sqlite")
with reddit_conn:
    reddit_cursor = reddit_conn.cursor()
    reddit_cursor.execute('select sqlite_version()')
    version = reddit_cursor.fetchone()
    print("SQLite Version {}".format(version[0]))
with reddit_conn:
    reddit_cursor = reddit_conn.cursor()
    reddit_cursor.execute('select * from May2015 Limit 3')
    sample_records = reddit_cursor.fetchall()
    for rec in sample_records:
        print(rec)
reddit_engine = create_engine("sqlite:///input/database.sqlite")
print("SQLalchemy Engine is connected to {}.".format(reddit_engine.url))
reddit_subreddit_pop = pandas.read_sql_query('select '
                                             'subreddit_id '
                                             ',subreddit '
                                             ',count(*) as comments '
                                             'from May2015 '
                                             'group by subreddit_id '
                                             , reddit_conn)
# How many subreddits were there
reddit_subreddit_pop.shape[0]
# Print the top 42 subreddits by comments
top_42_subreddits = reddit_subreddit_pop.sort_values(['comments'], ascending=False).head(42)
top_42_subreddits['rank'] = top_42_subreddits['comments'].rank(ascending=False)
top_42_subreddits

plt.bar(top_42_subreddits['rank'], top_42_subreddits['comments'], width=1)
plt.xticks(top_42_subreddits['rank'], top_42_subreddits['subreddit'], rotation=90)
plt.ylabel('Comments')
plt.show()
top_n = 10
top_subreddits = reddit_subreddit_pop.sort_values(['comments'], ascending=False).head(top_n)
#Create dict mapping from subreddit id to subreddit name
top_sub_name_agg = top_subreddits.set_index('subreddit_id')['subreddit'].to_dict()
print('Total comments in top {} subreddits: {}'.format(top_n, sum(top_subreddits.comments)))
top_subreddit_list = top_subreddits.subreddit_id.tolist()
top_subreddits
# Get word count and score for each comment in the top subreddits
sql_query_words = ''.join(['select '
                           ,'subreddit_id '
                           ,',link_id '
                           ,',(length(trim(body)) - length(replace(replace(body,"\n",""), " ", "")) + 1) as words '
                           ,',score '
                           ,'from May2015 '
                           ,'where score_hidden = 0 '
                           ,'and subreddit_id in ("{}") '.format('","'.join(top_subreddit_list))
                          ])
print(sql_query_words)
reddit_word_cnt = pandas.read_sql_query(sql_query_words, reddit_conn)
reddit_word_cnt.head()
reddit_word_cnt.describe()
# Summarize the words and scores by subreddit and link
link_agg = reddit_word_cnt.groupby(['subreddit_id', 'link_id'], as_index=False).agg([numpy.mean, 'count'])
link_agg.head()
# Summarize the words and scores by subreddit and link
top_subs_agg = link_agg.groupby(level=['subreddit_id'], as_index=False).agg([numpy.mean, 'count'])
top_subs_agg.head()
top_subs_agg['link_rank'] = top_subs_agg['words','mean','count'].rank(ascending=False)
xlabels = [top_sub_name_agg[x] for x in top_subs_agg.index.values]
plt.bar(top_subs_agg['link_rank'], top_subs_agg['words','mean','count'], width=1, align='center')
plt.xticks(top_subs_agg['link_rank'], xlabels, rotation=90)
plt.show()
# Determine rank based on number of comments in each links by subreddit.
top_subs_agg['comment_per_link_rank'] = top_subs_agg['words','count','mean'].rank(ascending=False)

plt.bar(top_subs_agg['comment_per_link_rank'], top_subs_agg['words','count','mean'], width=1, align='center')
plt.xticks(top_subs_agg['comment_per_link_rank'], xlabels, rotation=90)
plt.show()
# Get tl;dr count for each link in the top subreddits
sql_query_tldr = ''.join(['select '
                           ,'subreddit_id '
                           ,',link_id '
                           ,',count(*) as tldr_cnt '
                           ,'from May2015 '
                           ,'where score_hidden = 0 '
                           ,'and (body GLOB "*tl;dr*" or body GLOB "*tldr*") '
                           ,'and subreddit_id in ("{}") '.format('","'.join(top_subreddit_list))
                           ,'group by subreddit_id, link_id '
                          ])
reddit_tldr_cnt = pandas.read_sql_query(sql_query_tldr, reddit_conn)

# Merge the average word count and score with the tldr data
link_data = pandas.merge(link_agg
                         ,reddit_tldr_cnt
                         ,left_index=True
                         ,right_on=['subreddit_id', 'link_id']
                         ,how='left'
                        )
# Make sure we have 0s instead of nulls
link_data['tldr_cnt'].fillna(0, inplace=True)

# Cap the tldr count. The data is too sparse out there.
link_data.loc[link_data['tldr_cnt']>20, 'tldr_cnt'] = 20

# Summarize the data by the tldr counts
tldr_agg = link_data.groupby('tldr_cnt', as_index=False).agg([numpy.mean, 'count'])
tldr_agg
# Plot the relationship between tldr flags in comments against the average score and word count.
# Smooth out the plot where sample size is low using splines.
tldr_score_spline = spline(tldr_agg.index, tldr_agg['score', 'mean', 'mean'], tldr_agg['score', 'count', 'count'], s=10000)
spline_linspace = numpy.linspace(0,20,200)
plt.plot(spline_linspace, tldr_score_spline(spline_linspace), lw=2)
plt.ylabel('Average Score')
plt.xlabel('tl;dr Occurences')
plt.show()