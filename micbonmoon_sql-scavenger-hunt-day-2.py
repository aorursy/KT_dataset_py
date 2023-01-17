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
query = """
    select type, count(id) as count
    from `bigquery-public-data.hacker_news.full`
    group by type
"""

count_by_type = hacker_news.query_to_pandas_safe(query)
print(count_by_type)
from numpy import log
from bokeh.io import output_notebook
from bokeh.plotting import figure
from bokeh.plotting import show
from bokeh.models import HoverTool
from bokeh.models import ColumnDataSource
output_notebook()
count_by_type['y'] = count_by_type['count']/10000000
count_by_type['size'] = log(count_by_type['count'])*3
source = ColumnDataSource(data=count_by_type)
hover = HoverTool(tooltips=[
    ('count', '@count')
])
p = figure(x_range=list(count_by_type['type']),
           y_range=[-0.2, 2.0],
           title='Number of stories by type (10^7)',
           plot_width=400, plot_height=300,
           tools=[hover, 'box_zoom', 'reset']
          )
p.circle(
    x='type', y='y',
    size='size', alpha=0.7,
    source=source)
show(p)
query = """
    select  count(id) as count
    from `bigquery-public-data.hacker_news.comments`
    where deleted
"""

count_deleted = hacker_news.query_to_pandas_safe(query)
print(count_deleted, ' comments were deleted.')
query = """
    select  type, max(timestamp) as most_recent_delete_time
    from `bigquery-public-data.hacker_news.full`
    where deleted
    group by type
"""

most_recent_deleted_by_type = hacker_news.query_to_pandas_safe(query)
print(most_recent_deleted_by_type)