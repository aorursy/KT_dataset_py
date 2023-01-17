# import package with helper functions 
import bq_helper

# create a helper object for this dataset
open_aq = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="openaq")

# print all the tables in this dataset (there's only one!)
open_aq.list_tables()
open_aq.head("global_air_quality")
# Your code goes here :)

query2 = """
        SELECT country
        FROM `bigquery-public-data.openaq.global_air_quality`
        WHERE unit != "ppm"
        """

no_ppm_countries = open_aq.query_to_pandas_safe(query2)
no_ppm_countries.head()
countries = no_ppm_countries['country'].unique()
print(countries)
print("No. of countries which use a unit other than ppm to measure pollution: " + str(len(countries)))
query3 = """
        SELECT pollutant
        FROM `bigquery-public-data.openaq.global_air_quality`
        WHERE value = 0
        """

zero_pollutants = open_aq.query_to_pandas_safe(query3)
zero_pollutants.head()
zero = zero_pollutants['pollutant'].unique()
print(zero)
print("No. of pollutants with zero value: " + str(len(zero)))
no_ppm_countries.to_csv("otp.csv")
zero_pollutants.to_csv("zero.csv")