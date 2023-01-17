#importing relevent packages, creating a helper object, and listing the tables in the dataset
import bq_helper 
open_aq = bq_helper.BigQueryHelper(active_project= "bigquery-public-data", 
                                       dataset_name = "openaq")
open_aq.list_tables()
# look at first couple of rows in global_air_quality
open_aq.head("global_air_quality")
# What US cities are included in the dataset?
query = """SELECT city
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE country = 'US'
        """
us_cities = open_aq.query_to_pandas_safe(query)
# What five cities have the most measurements taken there?
us_cities.city.value_counts().head()
# Which countries use a unit other than ppm to measure any type of pollution?

# My preview of the dataset does not show a row where unit = 'ppm'.
# In this case, I am curious to what other untis are used.

query = """ SELECT DISTINCT unit
            FROM `bigquery-public-data.openaq.global_air_quality`
        """
units_distinct = open_aq.query_to_pandas_safe(query)
units_distinct.unit.head()
# Cool. Looks like only 1 other unit. Now I just need to answer the first question.

query = """ SELECT DISTINCT country
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE unit != 'ppm'
        """
no_ppm = open_aq.query_to_pandas_safe(query)
no_ppm
#Which pollutants have a value of exactly 0?

query = """ SELECT DISTINCT pollutant
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE value = 0.00
        """
zero_value = open_aq.query_to_pandas_safe(query)
zero_value