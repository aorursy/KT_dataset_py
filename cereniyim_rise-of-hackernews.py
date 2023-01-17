import os

import pandas as pd

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# view first rows of stories table

from google.cloud import bigquery



# Create a "Client" object

client = bigquery.Client()



# Construct a reference to the "hacker_news" dataset

dataset_ref = client.dataset("hacker_news", project="bigquery-public-data")



# Construct a reference to the "stories" table

table_ref_stories = dataset_ref.table("stories")



# API request - fetch the table

table_stories = client.get_table(table_ref_stories)



# Preview the first five lines of the "stories" table

client.list_rows(table_stories, max_results=5).to_dataframe()
table_stories.schema
# Construct a reference to the "comments" table

table_ref_comments = dataset_ref.table("comments")



# API request - fetch the table

table_comments = client.get_table(table_ref_comments)



# Preview the first five lines of the "comments" table

client.list_rows(table_comments, max_results=5).to_dataframe()
table_comments.schema
# look at the parent_id column in the comments table

# then search for parent_ids who does not represent a comment_id in the comments table

# if this results are no empty, it means parent column represent parend comment id and one another id

query_for_id_selection = """ 

                            SELECT DISTINCT(parent)

                            FROM `bigquery-public-data.hacker_news.comments`

                            WHERE parent NOT IN (

                                                    SELECT id

                                                    FROM `bigquery-public-data.hacker_news.comments`)

            

"""

# set quota not to exceed limits

safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=10**10)



# create job to execute query

query_for_id_selection_job = client.query(query_for_id_selection, job_config=safe_config)



#load results into a dataframe

query_for_id_result = query_for_id_selection_job.to_dataframe()



# turn dataframe to a a series, then list, then list of strings to use in the query

non_comment_ids = ",".join(map(str,query_for_id_result.head(20).parent.tolist()))
# function to write query results to a dataframe

# without exceeding the 1 GB quota per query

def query_to_dataframe(query_name):

    safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=10**10)

    query_name_job = client.query(query_name, job_config=safe_config)

    return query_name_job.to_dataframe()
print("Number of parent_ids that are not comment_ids:", query_for_id_result.parent.nunique())
non_comment_ids
# are parent_ids that does not represent a comment_id, do they belong to a story?

non_comment_id_query = """ 

                            SELECT *

                            FROM `bigquery-public-data.hacker_news.stories` AS stories

                            WHERE id IN ({})

                    """.format(non_comment_ids)



# print query results

query_to_dataframe(non_comment_id_query).head()
# look at parent id row 4318042 to see how many comments are dependant on it

# in the stories table, number of  descendants are 113

query_number_of_comments = """

                            SELECT *

                            FROM `bigquery-public-data.hacker_news.comments`

                            WHERE parent = 4318042

"""



print("Number of comments that story 4318042 has:", len(query_to_dataframe(query_number_of_comments)))

# it did not give 122 rows because of the tree structure.

# it only give the first level comment id which is 12.
id_query = """ 

                SELECT id

                FROM `bigquery-public-data.hacker_news.stories`

                WHERE id IN (

                            SELECT id

                            FROM `bigquery-public-data.hacker_news.comments`

                            )

"""



query_to_dataframe(id_query)
# to have a look we are going to count the number of comments and posts created 

# for not deleted and dead ones

active_users_query = """

                        WITH active_users_from_stories AS (

                            SELECT author,

                                COUNT(*) AS number_of_stories

                            FROM `bigquery-public-data.hacker_news.stories`

                            WHERE deleted IS NOT TRUE AND dead IS NOT TRUE

                            GROUP BY author

                            ORDER BY number_of_stories DESC

                        ),

                        active_users_from_comments AS (

                            SELECT author,

                                COUNT(*) AS number_of_comments

                            FROM `bigquery-public-data.hacker_news.comments`

                            WHERE deleted IS NOT TRUE AND dead IS NOT TRUE

                            GROUP BY author

                            ORDER BY number_of_comments DESC

                        )

                        SELECT active_users_from_comments.author,

                            number_of_stories,

                            number_of_comments

                        FROM active_users_from_stories

                        FULL JOIN active_users_from_comments

                         ON active_users_from_stories.author = active_users_from_comments.author

                        ORDER BY number_of_stories DESC

                        LIMIT 10

"""

# if you change last ORDER BY clause to number_of_comments will list the users who commented most

safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=10**10)

active_users_query_job = client.query(active_users_query, job_config=safe_config)

active_users_query_result = active_users_query_job.to_dataframe()

active_users_query_result
# look at the total number of stories in the whole hackernews which is not dead or deleted

total_stories = """

                    SELECT COUNT(id)

                    FROM `bigquery-public-data.hacker_news.stories`

                    WHERE deleted IS NOT TRUE AND dead IS NOT TRUE

"""

total_stories_df = client.query(total_stories).result().to_dataframe()



percentage = active_users_query_result.number_of_stories.sum()/total_stories_df.f0_.iloc[0]



print("Most active users in terms of number of stories created have created the {} of the whole stories"

      .format(round(percentage,2)))
# before providing an answer to this question investigate and understand full table



# Construct a reference to the "full" table

table_ref_full = dataset_ref.table("full")



# API request - fetch the table

table_full = client.get_table(table_ref_full)



# Preview the first five lines of the "full" table

client.list_rows(table_full, max_results=5).to_dataframe()
# if those keywords occur in the full table's text or title column we can say they got covered

coverage_query = """WITH startup_ranking_score AS (

                    SELECT CASE 

                        WHEN title LIKE "%Airbnb%" OR text LIKE "%Airbnb%" THEN "Airbnb"

                        WHEN title LIKE "%Stripe%" OR text LIKE "%Stripe%" THEN "Stripe"

                        WHEN title LIKE "%Dropbox%" OR text LIKE "%Dropbox%" THEN "Dropbox"

                        WHEN title LIKE "%Zapier%" OR text LIKE "%Zapier%" THEN "Zapier"

                        WHEN title LIKE "%Reddit%" OR text LIKE "%Reddit%" THEN "Reddit"

                        END AS popular_startup_name,

                    ranking,

                    score

                    FROM `bigquery-public-data.hacker_news.full`

                                                                )

                    SELECT popular_startup_name,

                        SUM(ranking) AS total_ranking,

                        SUM(score) AS total_score

                    FROM startup_ranking_score

                    GROUP BY popular_startup_name

                    ORDER BY total_score DESC

"""

query_to_dataframe(coverage_query)
# to look at this first total number of comments generated per day will be investigated

# then the average of the year will be aggregated

average_daily_comments_per_year = """ WITH total_comments_generated_per_day AS (

                                        SELECT EXTRACT(DAYOFYEAR FROM time_ts) AS day,

                                            EXTRACT(YEAR FROM time_ts) AS year,

                                            COUNT(id) AS total_comments

                                        FROM `bigquery-public-data.hacker_news.comments`

                                        GROUP BY year, day

                                        )

                                        SELECT year, 

                                            AVG(total_comments) AS average_daily_comments

                                        FROM total_comments_generated_per_day

                                        GROUP BY year

                                        ORDER BY year

"""

query_to_dataframe(average_daily_comments_per_year)
# this question will be investigated in the full table 

number_of_users = """

                    SELECT EXTRACT(YEAR FROM timestamp) AS year,

                        COUNT(DISTINCT f.by) AS number_of_users

                    FROM `bigquery-public-data.hacker_news.full` AS f

                    WHERE timestamp IS NOT NULL

                    GROUP BY year

                    ORDER BY year     

"""

query_to_dataframe(number_of_users)
# to answer this question stories and comments tables will be joined and 

# and timedifference of the time_ts will be investigated

time_to_receive_comment = """

                            WITH time_difference AS (

                                SELECT stories.id AS story_id,

                                    MIN(TIMESTAMP_DIFF(comments.time_ts, stories.time_ts, SECOND)) AS second

                                FROM `bigquery-public-data.hacker_news.stories` AS stories

                                LEFT JOIN `bigquery-public-data.hacker_news.comments` AS comments

                                    ON stories.id = comments.parent

                                GROUP BY story_id

                                ORDER BY second ASC)

                            SELECT *

                            FROM time_difference

                            WHERE second >= 0

"""

query_to_dataframe(time_to_receive_comment).head(10)
print("Average number of hours passed for a story to receive a comment:",

      round(query_to_dataframe(time_to_receive_comment).second.mean()/3600,2))
# to answer this question we are going to look at the ratio of 

# comments receiving subcomments to all comments



# total number of comments in the comments table

total_num_comments = """

                        SELECT COUNT(DISTINCT id)

                        FROM `bigquery-public-data.hacker_news.comments`

                    """



# total number of comments having sub_comments in the comments table

total_num_comments_w_sub_comments = """ WITH comments_w_subcomment_list AS (

                                            SELECT id,

                                                CASE

                                                    WHEN id IN (

                                                        SELECT DISTINCT(parent)

                                                        FROM `bigquery-public-data.hacker_news.comments`) 

                                                            THEN 1

                                                    ELSE 0

                                                END AS is_commented

                                            FROM `bigquery-public-data.hacker_news.comments`)

                                        SELECT SUM(is_commented)

                                        FROM comments_w_subcomment_list

"""
percent_of_commented_comments = 100 * (query_to_dataframe(total_num_comments_w_sub_comments).f0_.iloc[0] 

                                      / query_to_dataframe(total_num_comments).f0_.iloc[0])



print("Percentage of comments with replies {}".format(round(percent_of_commented_comments,2)))
# we are going to investigate this question in the union of comments and stories table

# with necessary attributes

user_and_creation_date = """ WITH authors_creation_times AS (

                                SELECT author, 

                                    time_ts,

                                        CASE

                                            WHEN author IS NOT NULL THEN "story"

                                                ELSE NULL

                                        END AS type

                                FROM `bigquery-public-data.hacker_news.stories` AS stories

                                UNION ALL

                                SELECT author,

                                    time_ts,

                                        CASE

                                            WHEN author IS NOT NULL THEN "comment"

                                                ELSE NULL

                                    END AS type

                                FROM `bigquery-public-data.hacker_news.comments` AS comments),

                            first_creation_date AS (

                                SELECT author,

                                    type,

                                    MIN(time_ts) AS first_activity_time

                                FROM authors_creation_times

                                GROUP BY author, type)

                            SELECT type,

                                COUNT(author) AS number_of_users

                            FROM first_creation_date

                            GROUP BY type        

"""

query_to_dataframe(user_and_creation_date)
# to answer this question

# first users who make their first activity on HackerNews on January 2014 will be identified

# then their activity will be matched from full table

# to make the query more efficient CTEs will be used rather than joining multiple tables at once



users_w_first_activity_2014_01 = """WITH users_from_2014_01 AS (

                                        SELECT f.by AS author,

                                            MIN(timestamp) AS first_activity

                                        FROM `bigquery-public-data.hacker_news.full` AS f

                                        WHERE timestamp >= '2014-01-01' AND timestamp < '2014-02-01'

                                        GROUP BY f.by)

                                    SELECT users_from_2014_01.author,

                                        users_from_2014_01.first_activity,

                                        f.type

                                    FROM users_from_2014_01

                                    LEFT JOIN `bigquery-public-data.hacker_news.full` AS f

                                    ON users_from_2014_01.author = f.by 

                                        AND users_from_2014_01.first_activity = f.timestamp   

                                """

query_to_dataframe(users_w_first_activity_2014_01).head(10)
# before answering this question lets look at the first rows of full_201510 table

# Construct a reference to the "full" table

table_ref_full = dataset_ref.table("full_201510")



# API request - fetch the table

table_full = client.get_table(table_ref_full)



# Preview the first five lines of the "full" table

client.list_rows(table_full, max_results=5).to_dataframe()
# to answer this we are going to use full_201510 table

users_posted_201510 = """ 

                        SELECT COUNT(DISTINCT f.by) AS number_of_users

                        FROM `bigquery-public-data.hacker_news.full_201510` AS f

                     """

query_to_dataframe(users_posted_201510)
# number of posts created temporary table will be created using full table

# and then moving averages per date and post category will be calculated

# using analytic functions

moving_average_query = """ WITH num_posts_per_day_type AS ( 

                            SELECT EXTRACT(DATE FROM timestamp) AS date,

                                type,

                                COUNT(id) AS num_posts

                            FROM `bigquery-public-data.hacker_news.full` 

                            WHERE timestamp >= "2018-01-01"

                            GROUP BY date, type

                            )

                          SELECT date,

                            type,

                            AVG(num_posts) OVER (

                                PARTITION BY type

                                ORDER BY num_posts

                                ROWS BETWEEN 7 PRECEDING AND 7 FOLLOWING) AS moving_average

                          FROM num_posts_per_day_type

                          ORDER BY date

"""

query_to_dataframe(moving_average_query).head(10)
# scores will be grouped and order per day created from scores table

# and a rank will be assigned using analytic functions

scores_query = """

                    SELECT id,

                        score,

                        EXTRACT(DATE FROM time_ts) AS date,

                        RANK() OVER(

                            PARTITION BY EXTRACT(DATE FROM time_ts)

                            ORDER BY score) AS score_rank

                    FROM `bigquery-public-data.hacker_news.stories`

                    WHERE score >= 0   

"""

query_to_dataframe(scores_query).head(10)