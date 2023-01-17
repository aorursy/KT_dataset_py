import bq_helper 



# create a helper object for our bigquery dataset

hacker_news = bq_helper.BigQueryHelper(active_project= "bigquery-public-data", 

                                       dataset_name = "hacker_news")
# print a list of all the tables in the hacker_news dataset

hacker_news.list_tables()
# print information on all the columns in the "full" table

# in the hacker_news dataset

hacker_news.table_schema("full")
# preview the first couple lines of the "full" table

hacker_news.head("full")
# preview the first ten entries in the by column of the full table

hacker_news.head("full", selected_columns="by", num_rows=10)
# this query looks in the full table in the hacker_news

# dataset, then gets the score column from every row where 

# the type column has "job" in it.

query = """SELECT score 

            FROM `bigquery-public-data.hacker_news.full`

            WHERE type = "job" """



# check how big this query will be (in GB)

hacker_news.estimate_query_size(query)
# check out the scores of job postings (if the query is smaller than 1 gig)

job_post_scores = hacker_news.query_to_pandas_safe(query)

job_post_scores.head()
# average score for job posts

job_post_scores.score.mean()
# import package with helper functions 

import bq_helper



# create a helper object for this dataset

open_aq = bq_helper.BigQueryHelper(active_project="bigquery-public-data",

                                   dataset_name="openaq")



# print all the tables in this dataset (there's only one!)

open_aq.list_tables()
# print the first couple rows of the "global_air_quality" dataset

open_aq.head("global_air_quality",100)
# query to select all the items from the "city" column where the

# "country" column is "us"

query = """SELECT city

            FROM `bigquery-public-data.openaq.global_air_quality`

            WHERE country = 'US'

        """



# the query_to_pandas_safe will only return a result if it's less

# than one gigabyte (by default)

us_cities = open_aq.query_to_pandas_safe(query)
# What five cities have the most measurements taken there?

us_cities.city.value_counts().head()
# Your code goes here :)

query = """

        Select distinct pollutant FROM `bigquery-public-data.openaq.global_air_quality`

        WHERE value = 0

        """

nonppm_countries = open_aq.query_to_pandas_safe(query)

nonppm_countries.head()

# import package with helper functions 

import bq_helper



# create a helper object for this dataset

hacker_news = bq_helper.BigQueryHelper(active_project="bigquery-public-data",

                                   dataset_name="hacker_news")



# print the first couple rows of the "comments" table

hacker_news.head("comments")
# query to pass to 

query = """SELECT parent, COUNT(id) as count

            FROM `bigquery-public-data.hacker_news.comments`

            GROUP BY parent

            HAVING COUNT(id) > 10

        """



# the query_to_pandas_safe method will cancel the query if

# it would use too much of your quota, with the limit set 

# to 1 GB by default

popular_stories = hacker_news.query_to_pandas_safe(query)



popular_stories.head()
# Your code goes here:

query = """SELECT COUNT(id), deleted

            FROM `bigquery-public-data.hacker_news.comments`

            WHERE deleted = True

        """

stories_count = hacker_news.query_to_pandas_safe(query)

stories_count.head()
# import package with helper functions 

import bq_helper



# create a helper object for this dataset

accidents = bq_helper.BigQueryHelper(active_project="bigquery-public-data",

                                   dataset_name="nhtsa_traffic_fatalities")



# query to find out the number of accidents which 

# happen on each day of the week

query = """SELECT COUNT(consecutive_number) AS num_of_accidents, 

                  EXTRACT(DAYOFWEEK FROM timestamp_of_crash) AS day_of_week

            FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`

            GROUP BY day_of_week

            ORDER BY num_of_accidents DESC

        """



# the query_to_pandas_safe method will cancel the query if

# it would use too much of your quota, with the limit set 

# to 1 GB by default

accidents_by_day = accidents.query_to_pandas_safe(query)
# library for plotting

import matplotlib.pyplot as plt



# make a plot to show that our data is, actually, sorted:

plt.plot(accidents_by_day.num_of_accidents)

plt.title("Number of Accidents by Rank of Day \n (Most to least dangerous)")



print(accidents_by_day)
# Your code goes here :)

query = """SELECT COUNT(consecutive_number) AS num_of_accidents, 

                  EXTRACT(DAYOFWEEK FROM timestamp_of_crash) AS day_of_week

            FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`

            GROUP BY day_of_week

            ORDER BY num_of_accidents DESC

        """



# the query_to_pandas_safe method will cancel the query if

# it would use too much of your quota, with the limit set 

# to 1 GB by default

accidents_by_day = accidents.query_to_pandas_safe(query)



# import package with helper functions 

import bq_helper



# create a helper object for this dataset

github = bq_helper.BigQueryHelper(active_project="bigquery-public-data",

                                              dataset_name="github_repos")



# You can use two dashes (--) to add comments in SQL

query = ("""

        SELECT L.license, COUNT(sf.path) AS number_of_files

        FROM `bigquery-public-data.github_repos.sample_files` as sf

        INNER JOIN `bigquery-public-data.github_repos.licenses` as L 

            ON sf.repo_name = L.repo_name

        GROUP BY L.license

        ORDER BY number_of_files DESC

        """)



file_count_by_license = github.query_to_pandas_safe(query, max_gb_scanned=6)



# print out all the returned results

print(file_count_by_license)
# Your code goes here :)

from google.cloud import bigquery

import pandas as pd

import bq_helper 

import numpy as np

%matplotlib inline

from wordcloud import WordCloud

import matplotlib.pyplot as plt



import spacy

nlp = spacy.load('en')



# create a helper object for our bigquery dataset

hacker_news = bq_helper.BigQueryHelper(active_project= "bigquery-public-data", 

                                       dataset_name = "hacker_news")



%matplotlib inline
hacker_news.head("stories")