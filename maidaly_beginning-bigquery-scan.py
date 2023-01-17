# import our bq_helper package

import bq_helper 
# create a helper object for our bigquery dataset

hacker_news = bq_helper.BigQueryHelper(active_project= "bigquery-public-data", 

                                       dataset_name = "hacker_news")
hacker_news.list_tables()
hacker_news.table_schema("stories")
hacker_news.head("stories", selected_columns='title', num_rows = 15)
query = """SELECT title

            FROM `bigquery-public-data.hacker_news.stories`

            """
# check how big this query will be

size = hacker_news.estimate_query_size(query)

print ("The size of the query is %.3f GB" % size)
hacker_news_stories_title = hacker_news.query_to_pandas_safe(query, max_gb_scanned=1)
hacker_news_stories_title = hacker_news_stories_title.dropna(how='all')
hacker_news_stories_title.info()
import wordcloud

import matplotlib.pyplot as plt

droped_words = ['ask', 'show', 'hn', '<', '>', 'day', 'look', 'hn:','say', 'help', 'without'

               'will', 'using', 'use', 'may', 'next']

words = str()

for title in hacker_news_stories_title.title:

    words += ' '.join([i for i in title.lower().split() if i not in droped_words])

cloud = wordcloud.WordCloud(background_color='black',

                            max_font_size=200,

                            width=1600,

                            height=800,

                            max_words=200,

                            relative_scaling=.5).generate(words)

plt.figure(figsize=(20,10))

plt.axis('off')

plt.savefig('comments-hackernews.png')

plt.imshow(cloud)