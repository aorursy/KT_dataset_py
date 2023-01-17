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
#query to select the countries which are not using ppm to measure pollution 
query = """SELECT DISTINCT country, unit
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE unit != 'ppm'
            ORDER BY country
        """

#estimate query size
open_aq.estimate_query_size(query)

#Run the query to get the dataframe of result, 
#query_to_pandas_safe will return the result only if output is less than 1gb
diff_unit_countries = open_aq.query_to_pandas_safe(query)

#displaying first few rows
diff_unit_countries.head()

# countries from result as pandas data series
#diff_unit_countries['country']

#Result as list
diff_unit_countries.country.tolist()
#query to select the pollutants with value exactly 0
query = """SELECT DISTINCT pollutant
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE value = 0.00
        """

#estimating the size of the query
open_aq.estimate_query_size(query)

#run the query to select the 0 value pollutants
zero_val_pollutants = open_aq.query_to_pandas_safe(query)

#displaying first few rows
zero_val_pollutants.head()

#listing the result
zero_val_pollutants.pollutant.tolist()