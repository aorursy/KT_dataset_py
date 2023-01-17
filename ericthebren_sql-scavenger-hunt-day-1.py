# import package with helper functions 
import bq_helper

# create a helper object for this dataset
open_aq = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="openaq")

# print all the tables in this dataset (there's only one!)
open_aq.list_tables()
# print the first couple rows of the "global_air_quality" dataset
open_aq.head("global_air_quality")
##Countries that do NOT use PPM
query_ppm = """
            SELECT country
            FROM (
                SELECT country
                    ,unit
                FROM `bigquery-public-data.openaq.global_air_quality`
                WHERE unit != 'ppm'
                GROUP BY country
                    ,unit
                ) AS T
            GROUP BY country
            ORDER BY country
        """
pollutant_by_country = open_aq.query_to_pandas_safe(query_ppm)
pollutant_by_country
## Pollutants that have 0

query_pollutant = """
                    SELECT pollutant
                        ,sum(value) AS value1
                    FROM `bigquery-public-data.openaq.global_air_quality`
                    GROUP BY pollutant
                    ORDER BY value1 ASC
                    """
pollutant_zero = open_aq.query_to_pandas_safe(query_pollutant)
pollutant_zero
