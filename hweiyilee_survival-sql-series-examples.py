# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# This block of code helps you to navigate the BigQuery environment -- to access and view data, and to run queries.
# Import BigQuery to access the public data
from google.cloud import bigquery
# Create a "Client" object
client = bigquery.Client()

# This function enables you to construct a reference to a dataset in BigQuery, and view the table(s).
# It takes in the dataset name as a string, and the list of tables you want to view as a list of strings.
# It also returns a dictionary of data frames containing the first 5 rows of each table.
def view_tables(dataset_name, table_list = None):
    dataset_ref = client.dataset(dataset_name, project = "bigquery-public-data")
    dataset= client.get_dataset(dataset_ref)
    tables_dict = dict()
    if table_list == None:
        table_list = [str(item.table_id) for item in client.list_tables(dataset)]
    for table_name in table_list:
        print(table_name)
        table_ref = dataset_ref.table(table_name)
        table = client.get_table(table_ref)
        first_five_rows = client.list_rows(table, max_results = 5).to_dataframe()
        print(first_five_rows)
        tables_dict[table_name] = first_five_rows
    return tables_dict

# This function enables you to run a query on a BigQuery dataset and returns the results as a dataframe.
# The input is the name of the query that you have created.
def run_query(query_name):
    # Ensures that the quota is not exceeded
    safe_config = bigquery.QueryJobConfig(maximum_bytes_billed = 10**10)
    this_query_job = client.query(query_name, safe_config)
    this_query_results = this_query_job.to_dataframe()
    return this_query_results

# This function enables you to save the output of a query to a CSV file.
# You need to specify the query name, but if you don't give a file name it will save to a file named 'query_output.csv'.
# The data will appear under your "Data" dropdown, in the "output" folder, and you can download it from there.
def save_query(query_name, filename = 'query_output'):
    query_results = run_query(query_name)
    query_results.to_csv('{}.csv'.format(filename), index = False)
# Chapter 3 -- Filtering the data you want: The "WHERE" Clause
# Example comes from the Global Air Quality dataset in BigQuery

openaq_tables = view_tables("openaq",["global_air_quality"])
# Find out all the distinct countries
distinct_countries_query = '''
              SELECT
              DISTINCT country
              FROM `bigquery-public-data.openaq.global_air_quality`
              '''
# Run the query and view the results. You will see that the country names are listed as 2-letter country codes
# and that there are 98 distinct countries in the table.
run_query(distinct_countries_query)
# Find out the country and city names where the unit used to measure air quality is 'ppm'
country_city_query = '''
              SELECT
              country,
              city
              FROM `bigquery-public-data.openaq.global_air_quality`
              where unit = 'ppm'
              GROUP BY
              country,
              city
              ''' 

# Run the query and view the results. You will see that different city names are capitalized differently.
run_query(country_city_query)
# Use upper() to set all the letters in the city name to uppercase.
# This way, you don't have to worry about how an individual city name is formatted. 
# You can achieve the same outcome by using lower() and filtering for where lower(city) in ('bayamon', 'illawarra')
two_cities_query = '''
              SELECT
              *
              FROM `bigquery-public-data.openaq.global_air_quality`
              where upper(city) in ('BAYAMON', 'ILLAWARRA')
              ''' 

# Run the query and view the results. You will see that the data for both cities was returned, without
# you having to remember and replicate exactly how the city names were capitalized in the table.
run_query(two_cities_query)
# Chapter 4 -- Aggregating Metrics: Using "GROUP BY"
# Example comes from the Hacker Comments dataset in BigQuery

hacker_tables = view_tables("hacker_news",["comments"])
# Use date functions to extract the year and month that the comment was made.
# Count the number of comments that were made by year and month.
comments_yrmth_query = '''
              SELECT
              EXTRACT(YEAR from time_ts) AS comment_year,
              EXTRACT (MONTH from time_ts) AS comment_month,
              COUNT(id) AS num_comments
              FROM `bigquery-public-data.hacker_news.comments`
              GROUP BY
              EXTRACT(YEAR from time_ts),
              EXTRACT(MONTH from time_ts)
              ;
              '''
# Run this query and view the results. The number of comments is aggregated by year-month combination because
# we grouped by year and month. We needed to put the year and month in both the SELECT and the GROUP BY statements to achieve this.
run_query(comments_yrmth_query)
# Compare the results of count(id) vs. count(*)
# This query uses count(id) to count the number of comments. It counts the number of rows that have
# a non-null value for the comment ID.
comments_yrmth_countid_query = '''
              SELECT
              *
              FROM
              (SELECT
              EXTRACT(YEAR from time_ts) AS comment_year,
              EXTRACT (MONTH from time_ts) AS comment_month,
              COUNT(id) AS num_comments
              FROM `bigquery-public-data.hacker_news.comments`
              GROUP BY
              EXTRACT(YEAR from time_ts),
              EXTRACT(MONTH from time_ts)) AS x
              ORDER BY 
              x.comment_year,
              x.comment_month
              ;
              '''

# This query uses count(*) to count the number of comments. It counts the number of rows in the table,
# regardless of whether there is a value in the id column or not.
comments_yrmth_countstar_query = '''

              SELECT
              *
              FROM
              (SELECT
              EXTRACT(YEAR from time_ts) AS comment_year,
              EXTRACT (MONTH from time_ts) AS comment_month,
              COUNT(*) AS num_comments
              FROM `bigquery-public-data.hacker_news.comments`
              GROUP BY
              EXTRACT(YEAR from time_ts),
              EXTRACT(MONTH from time_ts)) AS x
              ORDER BY 
              x.comment_year,
              x.comment_month
              ;
              
              '''

# Creates a data frame for the results of each of the two queries
countid_data = run_query(comments_yrmth_countid_query)
countstar_data = run_query(comments_yrmth_countstar_query)

# Compares all the values in the 2 data frames and returns True if all values are the same, i.e. the 2 data frames are identical.
# When would you expect the results to be identical? It's when every row in the id column has a non-null value (i.e. every
# comment has a valid ID).
(countid_data == countstar_data).all().all()

# Chapter 5 -- Define your own metrics: Operators & conditionals
# Example comes from the Chicago Taxi Trips dataset in BigQuery
chicago_taxi_trips_tables = view_tables("chicago_taxi_trips")

# Pandas truncates the number of columns you can see when printing out the table, in order to fit the page size.
# View all the column names in the 'taxi_trips' table.
chicago_taxi_trips_tables['taxi_trips'].columns
# Let's create 2 new metrics:
# 1) The total revenue that went to the taxi company i.e. fares + tips
# 2) The average amount paid per mile i.e. trip total divided by the number of miles
trip_revenues_query = '''
                      SELECT
                      EXTRACT (YEAR FROM trip_start_timestamp) AS trip_year,
                      EXTRACT (MONTH FROM trip_start_timestamp) AS trip_month,
                      SUM(fare + tips) AS taxi_driver_rev,
                      ROUND(SUM(trip_total)/ SUM(trip_miles),2) AS total_per_mile
                      FROM `bigquery-public-data.chicago_taxi_trips.taxi_trips`
                      GROUP BY
                      EXTRACT (YEAR FROM trip_start_timestamp),
                      EXTRACT (MONTH FROM trip_start_timestamp);
                      '''

run_query(trip_revenues_query)
# Let's classify the trips into 3 categories: short, medium and long. We can then calculate the 
# amount paid per mile and the tips as a percentage of fares for each category.
trip_categories_query =  '''
                         SELECT
                         (CASE WHEN trip_seconds < 300 THEN 'SHORT'
                               WHEN trip_seconds > 900 THEN 'LONG'
                               ELSE 'MID' END )as trip_length,
                         ROUND(AVG((100*tips)/fare),2) AS tip_percentage
                         FROM `bigquery-public-data.chicago_taxi_trips.taxi_trips`
                         WHERE EXTRACT (YEAR FROM trip_start_timestamp) = 2020
                         AND EXTRACT (MONTH FROM trip_start_timestamp) = 7
                         AND trip_seconds > 0
                         AND fare > 0
                         GROUP BY
                         (CASE WHEN trip_seconds < 300 THEN 'SHORT'
                               WHEN trip_seconds > 900 THEN 'LONG'
                               ELSE 'MID' END) ;
                         '''

run_query(trip_categories_query)
# Let's compare the weighted and simple averages of the tip percentage. You will see that the numbers are slightly different.
tip_pct_compare_query =  '''
                         SELECT
                         (CASE WHEN trip_seconds < 300 THEN 'SHORT'
                               WHEN trip_seconds > 900 THEN 'LONG'
                               ELSE 'MID' END )as trip_length,
                         ROUND(AVG((100*tips)/fare),2) AS tip_percentage_simple,
                         ROUND(SUM(100*tips)/SUM(fare),2) AS tip_percentage_weighted
                         FROM `bigquery-public-data.chicago_taxi_trips.taxi_trips`
                         WHERE EXTRACT (YEAR FROM trip_start_timestamp) = 2020
                         AND EXTRACT (MONTH FROM trip_start_timestamp) = 7
                         AND trip_seconds > 0
                         AND fare > 0
                         GROUP BY
                         (CASE WHEN trip_seconds < 300 THEN 'SHORT'
                               WHEN trip_seconds > 900 THEN 'LONG'
                               ELSE 'MID' END) ;
                         '''

run_query(tip_pct_compare_query)
# Chapter 6 -- When to use more complex joins
# For the outer joins example, we're using the posts_questions and posts_answers tables. 
# This is an overview of the data in each of these tables.
q_and_a = view_tables('stackoverflow',['posts_questions','posts_answers'])
# This query pulls the number of questions in the questions table, and the number of answers in the answers table.
# Because we are doing a full join, we will get all questions and all answers.
# For a question without an answer, the answer ID will be null for that row.
# For an answer without a question, the question ID will be null for that row.

stackoverflow_answer_query_full = '''

               SELECT
               EXTRACT(YEAR FROM q.creation_date) AS qn_year,
               EXTRACT(MONTH FROM q.creation_date) AS qn_month,
               COUNT(q.id) AS question_count,
               COUNT(a.id) AS answer_count,
               SUM (CASE WHEN a.id IS NULL THEN 1 ELSE 0 END) AS q_without_a,
               SUM(CASE WHEN q.id IS NULL THEN 1 ELSE 0 END) AS a_without_q
               FROM `bigquery-public-data.stackoverflow.posts_questions` AS q
               FULL JOIN `bigquery-public-data.stackoverflow.posts_answers` AS a
               ON (q.id = a.parent_id)
               WHERE EXTRACT(YEAR FROM q.creation_date) = 2020
               GROUP BY
               EXTRACT(YEAR FROM q.creation_date),
               EXTRACT(MONTH FROM q.creation_date)
               ;
               
               '''
# Run the query and view the results. You will see that there are questions without answers, but no answers without questions.
run_query(stackoverflow_answer_query_full)
# Next, we will look at the stackoverflow_posts and posts_answers table, which matches answers to a given post.
# If you compare the two tables, they have the same set of column names, but parent_id is a foreign key in the 
# posts_answers table that links a parent post to a particular answer.
# Similarly, accepted_answer_id is a foreign key in the posts table which will link an answer as the accepted answer
# for a given post.
posts_and_a = view_tables('stackoverflow',['stackoverflow_posts','posts_answers'])
print(posts_and_a['stackoverflow_posts'].columns)
print(posts_and_a['posts_answers'].columns)
# This query counts the number of post-accepted answer pairs per owner, for every owner who has a valid display name.


stackoverflow_accepted_answers_query = '''

               SELECT
               p.owner_display_name,
               COUNT(p.id) AS post_count,
               COUNT(a.id) AS answer_count
               FROM `bigquery-public-data.stackoverflow.stackoverflow_posts` AS p
               JOIN `bigquery-public-data.stackoverflow.posts_answers` AS a
               ON (p.id = a.parent_id AND p.accepted_answer_id = a.id)
               WHERE 
               p.owner_display_name != 'None'
               GROUP BY
               p.owner_display_name
               ORDER BY
               COUNT(p.id) DESC
               ;
               
               '''

# In the results, note that the number of posts and the number of accepted answers is the same, 
# because every post has only one accepted answer.
run_query(stackoverflow_accepted_answers_query)
# This query joins the posts table to the answers table on only one foreign key - primary key pair instead
# of both. It only joins on whether the post ID was the parent ID of the answer. 

stackoverflow_all_answers_query = '''

               SELECT
               p.owner_display_name,
               COUNT(p.id) AS post_count,
               COUNT(a.id) AS answer_count
               FROM `bigquery-public-data.stackoverflow.stackoverflow_posts` AS p
               JOIN `bigquery-public-data.stackoverflow.posts_answers` AS a
               ON (p.id = a.parent_id)
               WHERE 
               p.owner_display_name != 'None'
               GROUP BY
               p.owner_display_name
               ORDER BY
               COUNT(p.id) DESC
               ;
               
               '''

# The results will capture repetitions of the post data for every answer, including the ones that were not accepted.
# Hence, the posts are duplicated, resulting in a much higher post count.
run_query(stackoverflow_all_answers_query)
# Counts the number of unique posts that 'anon' made in the posts table, to compare which of the above two queries is giving the
# accurate number of posts. 
stackoverflow_unique_posts_query = '''
               SELECT
               p.id AS post_id,
               p.accepted_answer_id
               FROM `bigquery-public-data.stackoverflow.stackoverflow_posts` AS p
               WHERE 
               p.owner_display_name ='anon'
               AND p.accepted_answer_id IS NOT NULL
               GROUP BY
               p.id,
               p.accepted_answer_id
               ;
               '''

# You will see that the number of posts is closer to the first query (joined on parent_id in answers and accepted_answer_id in posts)
# than in the second query. Hence, you know that the second query has duplicated the data.
run_query(stackoverflow_unique_posts_query)
# Chapter 7 -- Sort your data with "ORDER BY"
# Find the cities with the highest pollution measured by 'ppm'.
most_polluting_cities_query = '''
              SELECT
              country,
              city,
              value,
              unit
              FROM `bigquery-public-data.openaq.global_air_quality`
              WHERE unit = 'ppm'
              AND value > 0
              GROUP BY
              country,
              city,
              value,
              unit
              ORDER BY value DESC
              ''' 

# You can see that the data is ordered by descending value, and that there are repeats. This suggests that
# you may need to apply one more layer of aggregation to the values to get a unique result for each city.
run_query(most_polluting_cities_query)

# This is an example where we are using ORDER BY to sort text data alphabetically.
cities_in_australia_query = '''

              SELECT
              city,
              unit,
              MAX(value) AS max_pollution
              FROM `bigquery-public-data.openaq.global_air_quality`
              WHERE country = 'AU'
              AND value > 0
              GROUP BY
              city,
              unit
              ORDER BY city
              
              ''' 

# The cities now appear in ascending alphabetical order. The default sort order of SQL is ascending, so if we want
# an ascending sort we don't have to specify it. Only if we want a descending sort, then we need to use DESC.
# Interestingly, some cities have reported their pollution in different units.
run_query(cities_in_australia_query)
# To sort on a calculated value, you may need to create a subquery.
# In the original table, the comment time is recorded as a timestamp. SQL has to pull out the year and month
# based on calculations that you have asked it to make. It is not smart enought to 
# You can then select data from x and order it by the year and month columns.

comments_yrmth_countid_query = '''
              SELECT
              *
              FROM
              (SELECT
              EXTRACT(YEAR from time_ts) AS comment_year,
              EXTRACT (MONTH from time_ts) AS comment_month,
              COUNT(id) AS num_comments
              FROM `bigquery-public-data.hacker_news.comments`
              GROUP BY
              EXTRACT(YEAR from time_ts),
              EXTRACT(MONTH from time_ts)) AS x
              ORDER BY 
              x.comment_year,
              x.comment_month
              ;
              '''

run_query(comments_yrmth_countid_query)
# This is an example which uses all the SQL keywords, namely SELECT, FROM, WHERE, GROUP BY, HAVING, ORDER BY, and LIMIT.
# Objective: Get the top 5 biggest polluters in Australia
top_5_polluters_australia_query = '''

              SELECT
              city,
              unit,
              MAX(value) AS max_pollution
              FROM `bigquery-public-data.openaq.global_air_quality`
              WHERE country = 'AU'
              AND value > 0
              GROUP BY
              city,
              unit
              HAVING max_pollution > 5
              ORDER BY max_pollution DESC
              LIMIT 5;
              
              ''' 

# LIMIT 5 is the line that tells SQL to bring back only 5 rows. Because we sorted the rows in descending order of polluting
# values, the first 5 rows in our output have the top 5 highest amounts of pollution.
run_query(top_5_polluters_australia_query)
# Chapter 8 -- Use window functions to automate heavily manual Excel tasks

# A common business task is to pull the top performers by geographical market or category. Especially when you are dealing with 
# geographical data, this can take a long time to do manually in Excel.
# The RANK() window function allows you to rank a particular value that you have pulled in a SQL query.
# In this case, we're pulling the maximum pollution value that a city has, and ranking the cities in order of maximum pollution.
# We then pull the top 3 ranks only for this output, in order to get the top 3 polluting cities per country.

top_3_polluting_cities_by_country_query = '''
              
              SELECT
              *
              
              FROM
              (SELECT
              city,
              country,
              unit,
              MAX(value) AS max_pollution,
              RANK() OVER
              (PARTITION BY country
              ORDER BY MAX(value) DESC) AS pollution_rank
              FROM `bigquery-public-data.openaq.global_air_quality`
              WHERE unit = 'ppm'
              AND city != 'N/A'
              AND value > 0
              GROUP BY
              city,
              country,
              unit
              ORDER BY 
              country,
              pollution_rank) AS x
              
              WHERE x.pollution_rank <=3
              ;
              
              ''' 

# Notice that some countries have less than 3 cities that measure pollution using ppm.
# There are also cases where multiple cities have the same pollution measure, and therefore share the same rank.
# If you have more than one city sharing a rank, you could get more than 3 cities per country e.g. Israel (IL)
# which has 4 cities sharing the number 2 rank.
run_query(top_3_polluting_cities_by_country_query)
# This is an example of tie-breaking using just the data for Israel.
# The only thing that's changed is to add another column to order by, that will break ties between rows that have the same
# value for the pollution metric.
israel_tiebreaking_query = '''
              
              SELECT
              *
              
              FROM
              (SELECT
              city,
              country,
              unit,
              MAX(value) AS max_pollution,
              RANK() OVER
              (PARTITION BY country
              ORDER BY MAX(value) DESC, city) AS pollution_rank
              FROM `bigquery-public-data.openaq.global_air_quality`
              WHERE unit = 'ppm'
              AND country = 'IL'
              AND city != 'N/A'
              AND value > 0
              GROUP BY
              city,
              country,
              unit
              ORDER BY 
              country,
              pollution_rank) AS x
              
              WHERE x.pollution_rank <=3
              ;
              '''

run_query(israel_tiebreaking_query)
# The AVG() window function allows you to compute a moving average. This is another commonn business task which 
# is very time-consuming in Excel. We will use the COVID-19 open data to compute a moving average. 
# Here are the data columns that we have in this table. It contains the global COVID-19 case data.

covid19_table_details = view_tables('covid19_open_data',['covid19_open_data'])
covid19_table_details['covid19_open_data'].columns
daily_new_cases_italy_query = '''
                                  SELECT
                                   c.date,
                                   SUM(c.new_confirmed) as new_cases
                                   FROM `bigquery-public-data.covid19_open_data.covid19_open_data` AS c
                                   WHERE c.date >='2020-08-01'
                                   AND c.country_code = 'IT'
                                   GROUP BY
                                   c.date
                                   ORDER BY
                                   c.date
                                   '''

run_query(daily_new_cases_italy_query)
# Now that we have nailed down a query to pull the daily new cases, we can layer on another calculation: the 7 days
# moving average of daily new cases. This is the AVG() OVER window function in the query.
# You have to arrange the dates in the correct chronological order (from earliest to latest, i.e. ascending) for SQL
# to know which dates are the last 7 dates.
# The 7th date is the current date, so that is why you ask SQL to average the dates from 6 rows ago till the current date
# (with each date being one row) to get to the last 7 days moving average.

moving_l7d_italy_query = '''
                                   SELECT
                                   it.date,
                                   it.new_cases,
                                   AVG(it.new_cases) OVER
                                   (ORDER BY date
                                   ROWS BETWEEN 6 PRECEDING AND CURRENT ROW) AS l7d_avg_new_cases
                                   FROM
                                   (SELECT
                                   c.date,
                                   SUM(c.new_confirmed) as new_cases
                                   FROM `bigquery-public-data.covid19_open_data.covid19_open_data` AS c
                                   WHERE c.date >='2020-08-01'
                                   AND c.date <='2020-09-02'
                                   AND c.country_code = 'IT'
                                   GROUP BY
                                   c.date
                                   ORDER BY
                                   c.date) as it
                                   GROUP BY
                                   it.date,
                                   it.new_cases
                                   '''

# In the output, you will notice that for every date that did not have 7 days of previous history, you will get an 
# average of all the history to date i.e. day 1 will be itself, day 2 is the average of days 1 and 2, etc.
run_query(moving_l7d_italy_query)
# This example expands from just Italy into all countries, using PARTITION BY to ensure that the last 7 days' moving
# average is calculated by country. You can save the output to a .csv file using the save_query function that was written
# in cell 2 of this notebook. This will allow you to play with it in Excel and verify that the moving averages work for 
# every country.

moving_l7d_allctries_query = '''
                                   SELECT
                                   x.date,
                                   x.country_code,
                                   x.new_cases,
                                   AVG(x.new_cases) OVER
                                   (PARTITION BY x.country_code
                                   ORDER BY x.date
                                   ROWS BETWEEN 6 PRECEDING AND CURRENT ROW) AS l7d_avg_new_cases
                                   FROM
                                   (SELECT
                                   c.date,
                                   c.country_code,
                                   SUM(c.new_confirmed) as new_cases
                                   FROM `bigquery-public-data.covid19_open_data.covid19_open_data` AS c
                                   WHERE c.date >='2020-08-01'
                                   AND c.date <='2020-09-02'
                                   GROUP BY
                                   c.date,
                                   c.country_code
                                   ORDER BY
                                   c.date) as x
                                   GROUP BY
                                   x.date,
                                   x.country_code,
                                   x.new_cases
                                   '''

# You need to specify a file name to save the data to when calling the save_query function.
save_query(moving_l7d_allctries_query,'moving_l7d_allctries_query')
# Chapter 9 -- Using subqueries and common table expressions
# In the Stack Overflow data set, we found 1,118 posts by 'anon' which were matched to accepted answers in the answers table. To find the mismatches, we can combine
# subqueries to pull the unique posts, then the posts matched to answers, and then find the post ID's that were in the first subquery but not in the second one.

anon_mismatched_posts_query = '''

               SELECT
               anon_p.post_id AS post_id,
               anon_p.accepted_answer_id AS accepted_answer_id_for_post,
               a1.id AS answer_associated_with_post
               
               FROM
               
               (SELECT
               p.id AS post_id,
               p.accepted_answer_id
               FROM `bigquery-public-data.stackoverflow.stackoverflow_posts` AS p
               WHERE 
               p.owner_display_name ='anon'
               AND p.accepted_answer_id IS NOT NULL
               ) AS anon_p
               
               LEFT JOIN
               
               (SELECT
                a.id as answer_id,
                p1.id as post_id
                FROM `bigquery-public-data.stackoverflow.stackoverflow_posts` AS p1
                JOIN `bigquery-public-data.stackoverflow.posts_answers` AS a
                ON p1.accepted_answer_id = a.id AND a.parent_id = p1.id
                WHERE p1.owner_display_name = 'anon'
                GROUP BY
                a.id,
                p1.id) AS anon_p_a ON anon_p.post_id = anon_p_a.post_id  
                
                LEFT JOIN
                
                `bigquery-public-data.stackoverflow.posts_answers` AS a1
                ON anon_p.post_id = a1.parent_id
                
            
                WHERE anon_p_a.post_id IS NULL
                ;
                
                '''

run_query(anon_mismatched_posts_query)
anon_mismatched_cte_query = '''

               WITH anon_p AS
               (SELECT
               p.id AS post_id,
               p.accepted_answer_id
               FROM `bigquery-public-data.stackoverflow.stackoverflow_posts` AS p
               WHERE 
               p.owner_display_name ='anon'
               AND p.accepted_answer_id IS NOT NULL
               ),
               
               anon_p_a AS
               (SELECT
                a.id as answer_id,
                p1.id as post_id
                FROM `bigquery-public-data.stackoverflow.stackoverflow_posts` AS p1
                JOIN `bigquery-public-data.stackoverflow.posts_answers` AS a
                ON p1.accepted_answer_id = a.id AND a.parent_id = p1.id
                WHERE p1.owner_display_name = 'anon'
                GROUP BY
                a.id,
                p1.id) 
                
                SELECT
                anon_p.post_id,
                anon_p.accepted_answer_id AS accepted_answer_id_for_post,
                a1.id AS answer_associated_with_post
                
                FROM
                anon_p LEFT JOIN anon_p_a ON anon_p.post_id = anon_p_a.post_id
                
                LEFT JOIN
                
                `bigquery-public-data.stackoverflow.posts_answers` AS a1
                ON anon_p.post_id = a1.parent_id
                
                WHERE anon_p_a.post_id IS NULL
                
                ;
                '''

run_query(anon_mismatched_cte_query)
