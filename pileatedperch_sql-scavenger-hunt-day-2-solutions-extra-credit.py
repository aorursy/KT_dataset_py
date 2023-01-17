import bq_helper
hacker_news = bq_helper.BigQueryHelper(active_project='bigquery-public-data', 
                                       dataset_name='hacker_news')
hacker_news.list_tables()
hacker_news.table_schema('comments')
hacker_news.head('comments')
hacker_news.query_to_pandas_safe('''
SELECT parent, COUNT(id) AS number_replies
FROM `bigquery-public-data.hacker_news.comments`
GROUP BY parent
HAVING COUNT(id) > 10
ORDER BY COUNT(id) DESC
''')
hacker_news.query_to_pandas_safe('''
SELECT type, COUNT(id) AS count
FROM `bigquery-public-data.hacker_news.full`
GROUP BY type
ORDER BY count DESC
''')
hacker_news.query_to_pandas_safe('''
SELECT COUNT(deleted)
FROM `bigquery-public-data.hacker_news.comments`
WHERE deleted IS TRUE
''')
hacker_news.query_to_pandas_safe('''
SELECT deleted, COUNT(id)
FROM `bigquery-public-data.hacker_news.comments`
GROUP BY deleted
''')
hacker_news.query_to_pandas_safe('''
SELECT COUNTIF(deleted)
FROM `bigquery-public-data.hacker_news.comments`
''')