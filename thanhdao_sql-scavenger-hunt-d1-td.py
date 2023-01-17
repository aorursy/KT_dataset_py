# Scavenger Hunt Questions
## 1. Which countries use a unit other than ppm to measure any type of pollution? (Hint: to get rows where the value *isn't* something, use "!=")
## 2. Which pollutants have a value of exactly 0?

# import our bq_helper package
import bq_helper 
# create a helper object for our bigquery dataset
OpenAQ = bq_helper.BigQueryHelper(active_project= "bigquery-public-data", 
                                       dataset_name = "openaq")
# print a list of all the tables in the hacker_news dataset
OpenAQ.list_tables()

# print information on all the columns in the "full" table 
# in the hacker_news dataset
# telling collumn name, datatype, mode, descriptions
OpenAQ.table_schema("global_air_quality")

# preview the first ten entries in the by column of the full table
OpenAQ.head("global_air_quality")
OpenAQ.head("global_air_quality", selected_columns="unit", num_rows=10)

query = """SELECT DISTINCT country
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE unit != "ppm" """
# check how big this query will be -- The query size is returned in gigabytes.
OpenAQ.estimate_query_size(query)
country = OpenAQ.query_to_pandas_safe(query)
print(country)
country.to_csv("Country_without_ppm_unit.csv")
## 2. Which pollutants have a value of exactly 0?
query2 = """SELECT DISTINCT pollutant,value
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE value = 0"""
OpenAQ.estimate_query_size(query2)
pollutants = OpenAQ.query_to_pandas_safe(query2)
print(pollutants)
pollutants.to_csv("pollutants_with_no_value.csv")
