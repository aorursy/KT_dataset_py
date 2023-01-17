# setup
import bq_helper

hacker_news = bq_helper.BigQueryHelper(active_project='bigquery-public-data',
                                      dataset_name='hacker_news')
# explore tables
hacker_news.list_tables()
# explore table 'full'
hacker_news.head('full')
query1 = '''SELECT COUNT(DISTINCT(id)) AS stories
        FROM `bigquery-public-data.hacker_news.full`
        '''
result1 = hacker_news.query_to_pandas_safe(query1)
result1
query2 = '''SELECT type, COUNT(*) AS count
        FROM `bigquery-public-data.hacker_news.full`
        WHERE deleted = True
        GROUP BY type
        HAVING type = 'comment'
'''
result2 = hacker_news.query_to_pandas_safe(query2)
result2
