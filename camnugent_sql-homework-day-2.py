import bq_helper

hn_dat = bq_helper.BigQueryHelper(active_project='bigquery-public-data',
                                dataset_name='hacker_news')
hn_dat.list_tables()
hn_dat.head('stories')
q1 = """
SELECT type, COUNT(id)
FROM `bigquery-public-data.hacker_news.full`
GROUP BY type
"""
hn_dat.estimate_query_size(q1)
hn_summary = hn_dat.query_to_pandas_safe(q1)
hn_summary
hn_dat.head('comments')
#(If a comment was deleted the "deleted" column in the comments table will have the value "True".)
q2 = """
SELECT deleted, COUNT(id)
FROM `bigquery-public-data.hacker_news.comments`
GROUP BY deleted
"""
comment_summary = hn_dat.query_to_pandas_safe(q2)
comment_summary