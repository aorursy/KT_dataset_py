# import our bq_helper package
import bq_helper 

# create a helper object for this dataset
open_aq = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="openaq")

# print all the tables in this dataset (there's only one!)
open_aq.list_tables()# print the first couple rows of the "global_air_quality" dataset
open_aq.head("global_air_quality")
# print the first couple rows of the "global_air_quality" dataset
open_aq.head("global_air_quality")
# Question 1 Which countries use a unit other than ppm to measure any type of pollution? (Hint: to get rows where the value isn't something, use "!=")
query = """SELECT country
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE unit != 'ppm'
        """
countries = open_aq.query_to_pandas_safe(query)
countries.country.value_counts().head()
#Question 2 Which pollutants have a value of exactly 0?
query2 = """SELECT DISTINCT pollutant, value
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE value = 0
        """
pollutants = open_aq.query_to_pandas_safe(query2)
pollutants