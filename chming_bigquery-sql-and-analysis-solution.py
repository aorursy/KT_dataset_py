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

open_aq.head("global_air_quality")
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
query = """SELECT DISTINCT country

            FROM `bigquery-public-data.openaq.global_air_quality`

            WHERE unit != 'ppm'

        """

open_aq.query_to_pandas_safe(query)
# Your code goes here :)

query = """SELECT DISTINCT pollutant

            FROM `bigquery-public-data.openaq.global_air_quality`

            WHERE value = 0.0

        """

open_aq.query_to_pandas_safe(query)
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

query = """SELECT type, COUNT(1) as count

            FROM `bigquery-public-data.hacker_news.full`

            GROUP BY type

        """

hacker_news.query_to_pandas_safe(query)
query = """SELECT COUNT(1) as count

            FROM `bigquery-public-data.hacker_news.comments`

            WHERE deleted is True

        """

hacker_news.query_to_pandas_safe(query)
# import package with helper functions 

import bq_helper



# create a helper object for this dataset

accidents = bq_helper.BigQueryHelper(active_project="bigquery-public-data",

                                   dataset_name="nhtsa_traffic_fatalities")



# query to find out the number of accidents which 

# happen on each day of the week

query = """SELECT COUNT(consecutive_number), 

                  EXTRACT(DAYOFWEEK FROM timestamp_of_crash)

            FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`

            GROUP BY EXTRACT(DAYOFWEEK FROM timestamp_of_crash)

            ORDER BY COUNT(consecutive_number) DESC

        """



# the query_to_pandas_safe method will cancel the query if

# it would use too much of your quota, with the limit set 

# to 1 GB by default

accidents_by_day = accidents.query_to_pandas_safe(query)
# library for plotting

import matplotlib.pyplot as plt



# make a plot to show that our data is, actually, sorted:

plt.plot(accidents_by_day.f0_)

plt.title("Number of Accidents by Rank of Day \n (Most to least dangerous)")



print(accidents_by_day)
# Your code goes here :)

query = """SELECT EXTRACT(HOUR FROM timestamp_of_crash) as hour,

                  COUNT(consecutive_number) as num_accidents

            FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`

            GROUP BY hour

            ORDER BY num_accidents DESC

        """

accidents.query_to_pandas_safe(query)
# Your code goes here :)

query = """SELECT registration_state_name, 

                  COUNT(1) as count

            FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2015`

            WHERE hit_and_run = 'Yes'

            GROUP BY registration_state_name

            ORDER BY count DESC

        """

accidents.query_to_pandas_safe(query)
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

query = """

        SELECT C.repo_name, count(1) as num_commits

        FROM `bigquery-public-data.github_repos.sample_commits` as C

        INNER JOIN `bigquery-public-data.github_repos.sample_files` as F 

            ON C.repo_name = F.repo_name

        WHERE f.path LIKE '%.py'

        GROUP BY C.repo_name

        ORDER BY num_commits DESC

        """



github.query_to_pandas_safe(query, max_gb_scanned=6)
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
## So we will declare the query variable

## we will be using same name for queries in entire notebook hope this doesnt create confusion

query = """

SELECT MIN(score) as min_score, MAX(score) as max_score, AVG(score) as avg_score

FROM `bigquery-public-data.hacker_news.stories`

"""



hacker_news.query_to_pandas_safe(query)
query = """

SELECT 

SUM(score) as total_score, author,

SUM(descendants) as total_decendants,

count(id) as total_stories

FROM `bigquery-public-data.hacker_news.stories`

GROUP BY author

ORDER BY total_score DESC

LIMIT 30

"""



top_users = hacker_news.query_to_pandas_safe(query)

top_users.fillna(0, inplace=True)

# Look at the first 10 top scores

top_users.sort_values("total_score")[::-1][:10].plot(x="author",y=["total_score", "total_decendants", "total_stories"], 

                                                   kind="bar",figsize=(14,6),

                                                   title='Top users stats').set_ylabel("Score")
query = """

SELECT 

COUNT(*) as total_stories

FROM `bigquery-public-data.hacker_news.stories`

"""



count = hacker_news.query_to_pandas_safe(query)

print("Total number of stories {}".format(count.total_stories.values[0]))
query = """

SELECT 

SUM(score) as total_score, author,

SUM(descendants) as total_decendants,

count(id) as total_stories

FROM `bigquery-public-data.hacker_news.stories`

WHERE author IS NOT NULL

GROUP BY author

ORDER BY total_stories DESC

LIMIT 30

"""

most_freq_users = hacker_news.query_to_pandas_safe(query)



# Look at the first 10 top scores

most_freq_users.sort_values("total_stories")[::-1][:10].plot(x="author",y=["total_score", "total_decendants", "total_stories"], 

                                                   kind="bar",figsize=(14,6),

                                                   title='Top users stats').set_ylabel("Score")
print("Total number of stories {} posted by 30 most active users".format(most_freq_users.total_stories.sum()))

print("Percentage stories posted by 30 most active users : {}%".format(most_freq_users.total_stories.sum()*100/count.total_stories.values[0]))
# Writing query string for our specific need here

query = """

SELECT score, title

FROM `bigquery-public-data.hacker_news.stories`

WHERE REGEXP_CONTAINS(title, r"Y Combinator|YCombinator")

ORDER BY score  DESC

"""



hacker_news.estimate_query_size(query)

HN_scores = hacker_news.query_to_pandas_safe(query)



HN_scores.sort_values("score")[::-1][:10].plot(x="title",y=["score"], 

                                                   kind="bar",figsize=(14,6),

                                                   title='Top scoring YC Stories').set_ylabel("Score")
print("Total number of stories with mention of YC {}".format(len(HN_scores)))

print("Avg score of stories with mention of YC {}".format(HN_scores.score.mean()))
# Writing query string for our specific need here

query = """

SELECT AVG(score) as avg_score

FROM `bigquery-public-data.hacker_news.stories`

WHERE id NOT IN (SELECT id FROM `bigquery-public-data.hacker_news.stories` WHERE REGEXP_CONTAINS(title, r"Y Combinator|YCombinator")) 

"""



hacker_news.estimate_query_size(query)

Non_HN_df = hacker_news.query_to_pandas_safe(query)

print("Total average score of stories without mention of HN {}".format(Non_HN_df.avg_score.values[0]))
# you can replace Uber with other companies

query ="""

SELECT count(*) as total_stories, SUM(score) as total_score,

EXTRACT(YEAR FROM time_ts) AS year

FROM `bigquery-public-data.hacker_news.stories`

WHERE REGEXP_CONTAINS(title, r"(U|u)ber")

GROUP BY year

ORDER BY year

"""



hacker_news.estimate_query_size(query)

uber_stories_df = hacker_news.query_to_pandas_safe(query)

uber_stories_df.plot(x="year",y=["total_stories","total_score"],

                     kind="bar",figsize=(14,6),

                     title='Uber stories per year').set_ylabel("Score")
companies_list = ["scribd","weebly" ,"songkick","ninite", "startuply","airbnb"]



query ="""

SELECT title, score, time_ts

FROM `bigquery-public-data.hacker_news.stories`

WHERE REGEXP_CONTAINS(title, r"Scribd |Weebly |Songkick |Ninite |Startuply |AirBnb ") 

ORDER BY time_ts

"""



companies_stories_df = hacker_news.query_to_pandas_safe(query)

companies_stories_df["title"] =companies_stories_df["title"].str.lower()

companies_stories_df['year'] = companies_stories_df['time_ts'].dt.year
fig, ax = plt.subplots()



for company in companies_list:

    comp_df = companies_stories_df.loc[companies_stories_df["title"].str.contains(company)]

    if len(comp_df)>0 :

        agg_d = {'score':'sum'}

        comp_plot = comp_df.groupby(["year"], as_index=False).agg(agg_d)

        comp_plot["counts"] = comp_df.groupby(["year"], as_index=False).size().reset_index(name='counts').counts

        comp_plot.plot(ax=ax, x="year",y="score", label=company, 

                     kind="line",figsize=(14,6),

                     title='Scores by companies per year')



plt.show()
query ="""

SELECT title

FROM `bigquery-public-data.hacker_news.stories`

ORDER BY score DESC

LIMIT 1000

"""



top_stories_df = hacker_news.query_to_pandas_safe(query)
%%time



def find_entity(title):

    

    parsed_title = nlp(title)

    entity_arr = []

    for num, entity in enumerate(parsed_title.ents):

        entity_arr.append(entity.text)

    return  ' '.join(entity_arr)



top_stories_df["entity"]= top_stories_df["title"].apply(find_entity)
top_stories_df = top_stories_df.loc[~ (top_stories_df['entity'] == "")]

top_stories_df.head()
top_entities = top_stories_df.groupby(["entity"]).size().reset_index(name='counts')

top_entities_p =top_entities.sort_values(['counts'])[::-1][:15]
top_entities_p.plot(x="entity",y=["counts"],

                     kind="bar",figsize=(14,6),

                     title='Top stories topic').set_ylabel("counts")
cloud = WordCloud(width=1440, height= 1080,max_words= 200).generate(' '.join(top_entities.entity.values))

plt.figure(figsize=(20, 15))

plt.imshow(cloud)

plt.axis('off');

plt.show()