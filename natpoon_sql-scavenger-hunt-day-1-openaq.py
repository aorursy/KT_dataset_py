# Import BigQuery package with helper functions 
import bq_helper

# Create a helper object for this OpenAQ dataset
open_aq = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="openaq")
# List all the tables in the OpenAQ dataset
open_aq.list_tables()
# Display the first couple rows of the "global_air_quality" table
open_aq.head("global_air_quality")
# Define 'query' to select all the rows from the "city" column where the
# "country" column is "US"
query = """SELECT city
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE country = 'US'
        """
# Define a new pandas dataframe with the query above.
us_cities = open_aq.query_to_pandas_safe(query)
# What five cities have the most measurements taken there?
us_cities.city.value_counts().head(10)
# Your code goes here :)
# -------
# Define my first query by selecting all rows of only the `country` and `unit` columns.
query1 = """SELECT country, unit
            FROM `bigquery-public-data.openaq.global_air_quality`
        """

# If the query above is less than 1 GB, the result will save as a new dataframe (`country_UOM`)
country_UOM = open_aq.query_to_pandas_safe(query1)
# Count the number of rows there are per each unit of measurement.
country_UOM.unit.value_counts().head()
# Using groupby to see whether or not each country uses 'ppm' or 'ug/m3'
country_UOM.groupby('country')['unit'].value_counts().unstack().fillna('-')
# Define second query to select only the 'country' column where 'ppm' is NOT one of the units.
query2 = """SELECT country
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE unit != 'ppm'
        """

# Create a new dataframe
non_ppm_countries = open_aq.query_to_pandas_safe(query2)
# Print a list of all countries that use a unit other than 'ppm' to measure air pollutants.
print(set(non_ppm_countries.country))
# Define the query that selects the columns below and only the rows where value = 0.
query = """SELECT city, country, pollutant, value
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE value = 0
        """
# Create a new dataframe from the result ('zero_vals')
zero_vals = open_aq.query_to_pandas_safe(query)
# Verify that all the rows have a value of exactly 0.
zero_vals['value'].sum()
# Print a list of which pollutants have a value of exactly 0.
print(set(zero_vals['pollutant']))
# Print how many times each pollutant has had a value of exactly 0.
zero_vals['pollutant'].value_counts()
zero_vals.groupby('country').pollutant.value_counts().unstack().fillna('-')
# (Welcoming any suggestions on how to make a cool plot from this result =D)