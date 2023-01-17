# import package with helper functions 
import bq_helper

# create a helper object for this dataset
open_aq = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="openaq")

# print all the tables in this dataset (there's only one!)
open_aq.list_tables()

# print the first couple rows of the "global_air_quality" dataset
open_aq.head("global_air_quality")
#Which countries use a unit other than ppm to measure pollution?
query = """SELECT country, unit
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE unit != 'ppm'
        """

met_countries = open_aq.query_to_pandas_safe(query)
#display how many counts per country
met_countries.country.value_counts().head()
query1 = """SELECT country, 
COUNT(*)
FROM `bigquery-public-data.openaq.global_air_quality`
WHERE unit != 'ppm'
GROUP BY country"""

unique_countries = open_aq.query_to_pandas_safe(query1)
#display how many counts per country
unique_countries

query2 = """SELECT unit
            FROM `bigquery-public-data.openaq.global_air_quality`
            GROUP BY unit
        """
unique_units = open_aq.query_to_pandas_safe(query2)

unique_units.head()
#Which pollutants have a value of 0?
query3 = """SELECT pollutant
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE value = 0.00
        """
no_pollute = open_aq.query_to_pandas_safe(query3)
no_pollute.pollutant.value_counts().head()