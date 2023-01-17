# import bq helper
import bq_helper
# create hacker_news variable
hacker_news = bq_helper.BigQueryHelper(active_project="bigquery-public-data", 
                                       dataset_name="hacker_news")
# list tables
hacker_news.list_tables()
hacker_news.head("comments")
# most repeated comment query
most_rep_comm_query = """
select parent, count(id) as total_replies
from `bigquery-public-data.hacker_news.comments`
group by parent
having total_replies > 10
"""
# estimate the query size
hacker_news.estimate_query_size(most_rep_comm_query)
# fetch data and create the dataframe
most_rep_comm = hacker_news.query_to_pandas_safe(most_rep_comm_query)
# find the sample from the dataframe
most_rep_comm.head()
# print information on all the columns in the "full" table
# in the hacker_news dataset
hacker_news.table_schema("full")
# query to find total story by type 
t_s_b_type_query = """
select count(id) as total_story, type 
from `bigquery-public-data.hacker_news.full`
group by type
"""
hacker_news.estimate_query_size(t_s_b_type_query)
total_story = hacker_news.query_to_pandas_safe(t_s_b_type_query)
total_story.head()
# total story by type which are deleted
tsb_type_del_query = """
select count(id) as total_story, type 
from `bigquery-public-data.hacker_news.full`
where deleted is True
group by type
"""
hacker_news.estimate_query_size(tsb_type_del_query)
total_del_story = hacker_news.query_to_pandas_safe(tsb_type_del_query)
total_del_story.head()
# find total deleted comments 
del_comments = """
select count(id) as del_comments, type
from `bigquery-public-data.hacker_news.full`
where deleted is True and type = 'comment'
group by type
"""
total_del_comments = hacker_news.query_to_pandas_safe(del_comments)
total_del_comments.head()
# query to find total story by type 
max_score_in_type_query = """
select type, id, title, max(score) as top_scorer
from `bigquery-public-data.hacker_news.full`
group by type, id, title
"""
hacker_news.estimate_query_size(max_score_in_type_query)
