import bq_helper
hacker_news = bq_helper.BigQueryHelper(active_project='bigquery-public-data',
                                       dataset_name='hacker_news')
hacker_news.list_tables()
hacker_news.head('comments')
query = '''SELECT parent, COUNT(id) AS num_children
           FROM `bigquery-public-data.hacker_news.comments`
           GROUP BY parent
           HAVING COUNT(id) > 10
           ORDER BY num_children DESC'''
hacker_news.estimate_query_size(query)
popular_stories = hacker_news.query_to_pandas_safe(query)
popular_stories.head()
hacker_news.head('full')
query = '''SELECT type, COUNT(id) AS num_stories
           FROM `bigquery-public-data.hacker_news.full`
           GROUP BY type
           ORDER BY num_stories DESC'''
hacker_news.estimate_query_size(query)
num_stories_by_type = hacker_news.query_to_pandas_safe(query)
num_stories_by_type
num_stories_by_type.plot(x='type', y='num_stories', kind='bar')
hacker_news.head('comments')
query = '''SELECT deleted, COUNT(id) as num_comments
           FROM `bigquery-public-data.hacker_news.comments`
           GROUP BY deleted
           ORDER BY num_comments DESC
           '''
hacker_news.estimate_query_size(query)
num_comments_by_deleted = hacker_news.query_to_pandas_safe(query)
num_comments_by_deleted
num_comments_by_deleted.plot(x='deleted', y='num_comments', kind='bar')
query = '''SELECT deleted, COUNT(id) as num_comments
           FROM `bigquery-public-data.hacker_news.comments`
           GROUP BY deleted
           HAVING deleted = True
           '''
hacker_news.estimate_query_size(query)
num_comments_deleted = hacker_news.query_to_pandas_safe(query)
num_comments_deleted
query = '''SELECT COUNT(id) as num_comments
           FROM `bigquery-public-data.hacker_news.comments`
           WHERE deleted = True
           '''
hacker_news.estimate_query_size(query)
num_comments_deleted = hacker_news.query_to_pandas_safe(query)
num_comments_deleted
query = '''SELECT author, AVG(LENGTH(text)) as avg_length
           FROM `bigquery-public-data.hacker_news.comments`
           GROUP BY author
           ORDER BY avg_length DESC
           '''
hacker_news.estimate_query_size(query)
avg_length_by_author = hacker_news.query_to_pandas_safe(query)
avg_length_by_author = hacker_news.query_to_pandas(query)
avg_length_by_author.head()
avg_length_by_author.head().plot(x='author', y='avg_length', kind='bar')