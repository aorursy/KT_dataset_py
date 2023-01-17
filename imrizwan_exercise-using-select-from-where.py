# import package with helper functions 
import bq_helper

# create a helper object for this dataset
open_aq = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="openaq")

# print all the tables in this dataset (there's only one!)
open_aq.list_tables()

open_aq.head("global_air_quality")
query_1 = """SELECT country, unit
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE unit != 'ppm'
        """
unit_other = open_aq.query_to_pandas_safe(query_1)
unit_other.head()
unit_other.count()
unit_other.country.value_counts().head()
query_2 =  """SELECT pollutant, value
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE value = 0
        """
pollutants_zero = open_aq.query_to_pandas_safe(query_2)
pollutants_zero.head(10)