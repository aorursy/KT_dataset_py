# import package with helper functions 

import bq_helper



# create a helper object for this dataset

open_aq = bq_helper.BigQueryHelper(active_project="bigquery-public-data",

                                   dataset_name="openaq")



# print all the tables in this dataset (there's only one!)

open_aq.list_tables()



# print look at top few rows

open_aq.head('global_air_quality')

# Query to select all the items from the "city" column where the "country" column is 'US'

us_city = """SELECT city

            FROM `bigquery-public-data.openaq.global_air_quality`

            WHERE country = 'US'

        """

open_aq.query_to_pandas_safe(us_city)
# Your Code Goes Here

pollution_unit = """ SELECT DISTINCT country

                FROM `bigquery-public-data.openaq.global_air_quality`

                WHERE unit != 'ppm'

             """



pollution_unit_sample = open_aq.query_to_pandas_safe(pollution_unit)



# Show first few rows from the dataset

print(pollution_unit_sample.head())
# Your Code Goes Here

zero_value_pollutants = """ 

                        SELECT * FROM `bigquery-public-data.openaq.global_air_quality`       

                        WHERE value = 0

                       """



zero_value_sample = open_aq.query_to_pandas_safe(zero_value_pollutants)



print(zero_value_sample.head())
