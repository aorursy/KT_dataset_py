import bq_helper
hacker_news = bq_helper.BigQueryHelper(active_project='bigquery-public-data',
                                       dataset_name='hacker_news'
                                      )
hacker_news.list_tables()
hacker_news.table_schema('full')
hacker_news.head('full')
hacker_news.table_schema('comments')
hacker_news.head('comments')
q1 = """
select
    type
    , count(id)
from
    `bigquery-public-data.hacker_news.full`
group by
    type
order by
    type
"""
hacker_news.estimate_query_size(q1)
stories = hacker_news.query_to_pandas_safe(q1)
stories
q2 = """
select
    count(deleted)
from
    `bigquery-public-data.hacker_news.comments`
where
    deleted = True
"""
hacker_news.estimate_query_size(q2)
hacker_news.query_to_pandas_safe(q2)
q3 = """
select
    type
    , max(id) max_id
    , min(id) min_id
from
    `bigquery-public-data.hacker_news.full`
group by
    type
order by
    type
"""
hacker_news.estimate_query_size(q3)
hacker_news.query_to_pandas_safe(q3)