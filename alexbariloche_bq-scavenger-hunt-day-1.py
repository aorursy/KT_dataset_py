# import bq_helper package
import bq_helper

# create a helper object for our bigquery dataset
openaq = bq_helper.BigQueryHelper(active_project= "bigquery-public-data", 
                                       dataset_name = "openaq")
query = """SELECT country FROM
            (SELECT country, count(*) AS units_used 
            FROM (
                SELECT country, unit
                FROM `bigquery-public-data.openaq.global_air_quality`
                GROUP BY country, unit)
            GROUP BY country)
            WHERE units_used > 1 """

countries_using_more_than_one_unit = openaq.query_to_pandas_safe(query)
countries_using_more_than_one_unit.head(20)
countries_using_more_than_one_unit.to_csv( "countries_using_more_than_one_unit.csv")
query = """SELECT DISTINCT country 
                FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE unit != 'ppm'"""

countries_using_micrograms_over_m3_unit = openaq.query_to_pandas_safe(query)
countries_using_micrograms_over_m3_unit.head(20)
countries_using_micrograms_over_m3_unit.to_csv( "countries_using_micrograms_over_m3_unit.csv")
countries_using_micrograms_over_m3_unit.count()
#query = """
query = """SELECT count(*) FROM (
                SELECT pollutant, SUM( value) AS qty
                FROM `bigquery-public-data.openaq.global_air_quality` 
                GROUP BY pollutant
                )
            WHERE qty = 0.0 """

pollutant_at_zero = openaq.query_to_pandas_safe(query)
pollutant_at_zero.head()
query = """SELECT pollutant, country, city, location 
            FROM `bigquery-public-data.openaq.global_air_quality` 
            WHERE value = 0.0 
            ORDER BY pollutant, country, city, location"""

pollutant_at_zero_detail = openaq.query_to_pandas_safe(query)
pollutant_at_zero_detail.head(30)
pollutant_at_zero_detail.to_csv("pollutant_at_zero_detail.csv")
