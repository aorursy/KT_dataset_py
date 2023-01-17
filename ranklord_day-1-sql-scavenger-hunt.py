# Google's BigQuery helper
import bq_helper as bq 

# create a helper object for our bigquery dataset
air_quality = bq.BigQueryHelper("bigquery-public-data", "openaq")
# Which countries use a unit other than ppm to measure any type of pollution?
query = """SELECT DISTINCT country, unit
           FROM `bigquery-public-data.openaq.global_air_quality`
           WHERE LOWER(unit) != 'ppm' ORDER BY country """

# execute query safely (it's less than 1GB) and display the result
not_ppm = air_quality.query_to_pandas_safe(query)
not_ppm
# Which pollutants have a value of exactly 0?
query = """SELECT DISTINCT pollutant, value
           FROM `bigquery-public-data.openaq.global_air_quality`
           WHERE VALUE = 0 ORDER BY pollutant """

zero_pollutant = air_quality.query_to_pandas_safe(query)
zero_pollutant
# Total pollution by country and by pollutant
query = """SELECT country, pollutant, FORMAT("%.2f", SUM(value)) as pollution
           FROM `bigquery-public-data.openaq.global_air_quality`
           GROUP BY country, pollutant 
           ORDER BY country, pollutant, pollution DESC """

test = air_quality.query_to_pandas_safe(query)
test.style.set_properties(**{'text-align': 'right'})
# Total pollution by time
query = """SELECT timestamp, FORMAT("%.2f", SUM(value)) as pollution
           FROM `bigquery-public-data.openaq.global_air_quality`
           GROUP BY timestamp """

test3 = air_quality.query_to_pandas_safe(query)
test3