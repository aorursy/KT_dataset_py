# import package with helper functions 
import bq_helper

# create a helper object for this dataset
hacker_news = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="hacker_news")

# print the first couple rows of the "comments" table
hacker_news.head("comments")
# query to pass to 
query = """SELECT parent, COUNT(id)
            FROM `bigquery-public-data.hacker_news.comments`
            GROUP BY parent
            HAVING COUNT(id) > 10
        """
# the query_to_pandas_safe method will cancel the query if
# it would use too much of your quota, with the limit set 
# to 1 GB by default
popular_stories = hacker_news.query_to_pandas_safe(query)
popular_stories.head()
# Your code goes here :)
import bq_helper

hacker_news = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                      dataset_name="hacker_news")

query_stories_by_type = """
select
  type        as story_type
, count(id)   as story_count
from
  `bigquery-public-data.hacker_news.full`
group by
  type
order by
  count(id) desc
"""

stories_by_type = hacker_news.query_to_pandas_safe(query_stories_by_type)

print(stories_by_type)
# plot the output
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

fig, ax = plt.subplots(1,1)
sns.barplot(
    x='story_type',
    y='story_count',
    data=stories_by_type,
    ax=ax
)
ax.set_xlabel('Story Type')
ax.set_ylabel('Story Count')
ax.get_yaxis().set_major_formatter(
    FuncFormatter(lambda x, p: format(int(x), ',')))
ax.set_title('Hacker News Story Type by Count')
query_comment_status_count = """
select
  `bigquery-public-data.hacker_news.comments`.by        as comment_author
, deleted   as comment_deleted_status
, count(id) as comment_count
from
  `bigquery-public-data.hacker_news.comments`
group by
  `bigquery-public-data.hacker_news.comments`.by
, deleted
order by
  count(id) desc
"""

comment_status = hacker_news.query_to_pandas_safe(query_comment_status_count)
comment_status.comment_deleted_status.fillna('False', inplace=True)

print(comment_status.groupby('comment_deleted_status').sum())
fig, ax = plt.subplots()
sns.barplot(
    x='comment_deleted_status',
    y='comment_count',
    data=comment_status.groupby('comment_deleted_status').sum().reset_index(),
    ax=ax
)
ax.set_xlabel('Comment Status (Deleted?)')
ax.set_ylabel('Count of Comments')
ax.get_yaxis().set_major_formatter(
    FuncFormatter(lambda x, p: format(int(x), ',')))
ax.set_title('Count of Comments by Deleted Status')
fig, ax1 = plt.subplots(figsize=(10,5))
sns.pointplot(
    x='comment_author',
    y='comment_count',
    data=comment_status[comment_status.comment_deleted_status=='False'].head(15),
    hue='comment_deleted_status',
    kind='scatter',
    join=False,
    markers='x',
    ax=ax1
)
ax1.xaxis.set_tick_params(rotation=45)
ax1.set_xlabel('Comment Author')
ax1.set_ylabel('Number of Comments')
ax1.get_yaxis().set_major_formatter(
    FuncFormatter(lambda x, p: format(int(x), ',')))
ax1.set_title('Commenters With Most Comments')
extra_credit_query = """
select
  sum(
    case
      when deleted = True
      then 1
      else 0
    end
  ) as deleted_comments
from
  `bigquery-public-data.hacker_news.comments`
where
  deleted = True
"""

ec = hacker_news.query_to_pandas_safe(extra_credit_query)
print(ec)