#open a bq_helper package to start working with BigQuery dataset
import bq_helper
#create a helper object for our bigquery dataset
open_aq = bq_helper.BigQueryHelper(active_project="bigquery-public-data", dataset_name="openaq")

#print all the tables in this dataset
open_aq.list_tables()
#print first few rows to see what data looks like :)
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
us_cities.city.value_counts().head(5)
#Select countries where the unit is other than ppm
query = """ SELECT country
                   FROM `bigquery-public-data.openaq.global_air_quality`
                   WHERE unit != 'ppm'
               """
#create a dataframe executing query
country_df = open_aq.query_to_pandas_safe(query)
#select all unique entries from the dataframe column country
country_df.country.unique()
#Select distinct countries where the unit is other than ppm
query2 = """ SELECT distinct country
                   FROM `bigquery-public-data.openaq.global_air_quality`
                   WHERE unit != 'ppm'
               """
country_df2 = open_aq.query_to_pandas_safe(query2)
print(country_df2)
open_aq.estimate_query_size(query)
open_aq.estimate_query_size(query2)
#which pollutants have value of exactly 0?
query3 = """ SELECT distinct pollutant
                   FROM `bigquery-public-data.openaq.global_air_quality`
                   WHERE value = 0
               """
pollutant_df = open_aq.query_to_pandas_safe(query3)
print(pollutant_df)
query4 = """ SELECT pollutant
               FROM `bigquery-public-data.openaq.global_air_quality`
               WHERE value = 0
            """
open_aq.estimate_query_size(query4)
pollutant_df = open_aq.query_to_pandas_safe(query4)
print(pollutant_df)
pollutant_df.pollutant.unique()
query3 = """
    SELECT country, pollutant, SUM(value) AS total
    FROM `bigquery-public-data.openaq.global_air_quality`
   GROUP BY country, pollutant
   HAVING total = 0
"""
pollutant3 = open_aq.query_to_pandas_safe(query3)
pollutant3
#does not work as expected. Index error after trying to get pandas DF
query4 = """SELECT pollutant, SUM(value) as tot_value
        FROM `bigquery-public-data.openaq.global_air_quality`
        GROUP BY pollutant
        HAVING tot_value = 0
    """ 