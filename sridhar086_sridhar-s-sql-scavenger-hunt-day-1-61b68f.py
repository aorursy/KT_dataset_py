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
query_unit = """SELECT country,unit
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE unit != 'ppm'
        """


pollution_units_country = open_aq.query_to_pandas_safe(query_unit)
countries = pollution_units_country.drop_duplicates().reset_index(drop=True).set_index('country').to_dict()['unit']
print ("There are ",len(countries)," with a unit of ", list(countries.values())[0],":",countries )
#countries
query_value = """SELECT DISTINCT pollutant
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE value = 0
        """


pollutant_value = open_aq.query_to_pandas_safe(query_value)
print ( "The list of pollutants that recorded 0 values are ",
list(pollutant_value.to_dict()['pollutant'].values())
    
)