# import our bq_helper package
import bq_helper 
# create a helper object for our bigquery dataset (OpenAQ = openaq)
hacker_news = bq_helper.BigQueryHelper(active_project= "bigquery-public-data", 
                                       dataset_name = "openaq")
# print a list of all the tables in the hacker_news dataset
hacker_news.list_tables()
# print information on all the columns in the "global_air_quality" table
# in the openaq dataset
hacker_news.table_schema("global_air_quality")
# preview the first couple lines of the "global_air_quality" table (using head())
hacker_news.head("global_air_quality")
# first query
query1 = """SELECT distinct country
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE unit != "pmm" """
# check how big this query will be (using estimate_query_size())
hacker_news.estimate_query_size(query1)
# check out the contry of unit not pmm (if the 
# query is smaller than 1 gig)
job_post_country = hacker_news.query_to_pandas_safe(query1)
# save our dataframe as a .csv 
job_post_country.to_csv("job_post_country.csv")
# second query
query2 = """SELECT distinct pollutant
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE value = 0 """
# check how big this query will be (using estimate_query_size())
hacker_news.estimate_query_size(query2)
# check out the pollutant of value 0 (if the 
# query is smaller than 1 gig)
job_post_pollutant = hacker_news.query_to_pandas_safe(query2)
# save our dataframe as a .csv 
job_post_pollutant.to_csv("job_post_pollutant.csv")