# Import query package, as well as pandas, and visualization packages
import bq_helper
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
%matplotlib inline 


# create a helper object for this dataset
hacker_news = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="hacker_news")
# The first query I'm running will count the number of stories there are per type
# I sorted the data by volume of stories, and am printing the top 5. 

type_query = """select distinct type, count(id) stories 
                    from `bigquery-public-data.hacker_news.full`
                group by 1 order by 2 desc"""
number_of_stories = hacker_news.query_to_pandas_safe(type_query)
print(number_of_stories.head())

# For the second query, I'm counting the distinct number of comments deleted, 
# and calculating it as a percent of the total. 
deleted_query = """select
                    count(case when deleted = TRUE then id else null end) deleted,
                    (count(case when deleted = TRUE then id else null end) / count(id)) *100  pct_deleted
                    from `bigquery-public-data.hacker_news.comments`
                  """
deleted_comments = hacker_news.query_to_pandas_safe(deleted_query)
print(deleted_comments.head())

# For the third query, I'm calculating the average number of comments per author per type. 
avg_query = """select type, count(distinct author) authors, avg(comments) avg_comments
                    from
                       (
                       select distinct a.by as author,type, count(distinct id) comments
                        from `bigquery-public-data.hacker_news.full` a 
                       group by 1,2
                       ) 
                    a group by 1 order by 1 desc 
                  """
avg_commenttypes = hacker_news.query_to_pandas_safe(avg_query)
print(avg_commenttypes.head())
# I wonder if there is a typical # of days that people drop off after.  The data is on the user, day level.
# there is a lot of data so I'm only looking at people who first interacted after 2018-01-01, but since
# I have to filter the data from the innermost, I added a filter to the outside that'll make it so I 
# cut out people who joined any day before 1/1/18, skewing the data. 
# also filtering for people without usernames (represents about 3% of the data, might be an interesting average metric)

counting_query = """
                select distinct agg_counts.by, interaction_date, total_days_as_user,
                index_days_as_user, running_interaction_count, total_interactions,
                running_interaction_count / total_interactions as pct_of_interactions
                from 
                (
                    select distinct date_diff.by, interaction_date,
                          date_diff(most_recent_interaction,first_interaction, day) total_days_as_user, 
                          date_diff(interaction_date,first_interaction, day) index_days_as_user,
                          max(date_diff(most_recent_interaction, interaction_date, day))over(partition by date_diff.by) max_days,
                          running_interaction_count, 
                          total_interactions
                          from
                           (
                           select distinct a.by, cast(timestamp as date) interaction_date,
                               min(cast(timestamp as date))over(partition by a.by) first_interaction,
                               max(cast(timestamp as date))over(partition by a.by) most_recent_interaction,
                               count(id)over(partition by a.by order by cast(timestamp as date)) running_interaction_count,
                               count(id)over(partition by a.by) total_interactions
                            from `bigquery-public-data.hacker_news.full` a 
                            where a.by is not null and a.by != ''
                            and cast(timestamp as date) >= cast('2018-01-01' as date) 
                            order by 1,2 
                           )
                           date_diff
                    ) 
                    agg_counts 
                    where max_days = total_days_as_user
                    order by 1,4
                """
time_as_user = hacker_news.query_to_pandas_safe(counting_query)
print(time_as_user.head())
time_as_user[time_as_user.by =='000000000000001']
