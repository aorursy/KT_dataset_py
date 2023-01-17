import pandas as pd
from google.cloud import bigquery
from bq_helper import BigQueryHelper

# sample query from:
# https://cloud.google.com/bigquery/public-data/openaq#which_10_locations_have_had_the_worst_air_quality_this_month_as_measured_by_high_pm10
QUERY = """
        SELECT location, city, country, value, timestamp
        FROM `bigquery-public-data.openaq.global_air_quality`
        WHERE pollutant = "pm10" AND timestamp > "2017-04-01"
        ORDER BY value DESC
        LIMIT 1000
        """

#execute query using the bq_helper library:
bq_assistant = BigQueryHelper('bigquery-public-data', 'openaq')
df = bq_assistant.query_to_pandas(QUERY)
df.head(3)

Count = """
        SELECT COUNT(location)
        FROM `bigquery-public-data.openaq.global_air_quality`
        LIMIT 1000
        """

bq_assistant = BigQueryHelper('bigquery-public-data', 'openaq')
df = bq_assistant.query_to_pandas(Count)
df.head(3)
Units = """
        SELECT DISTINCT unit
        FROM `bigquery-public-data.openaq.global_air_quality`
        ORDER BY unit DESC
        LIMIT 1000
        """
bq_assistant = BigQueryHelper('bigquery-public-data', 'openaq')
df = bq_assistant.query_to_pandas(Units)
df.head(3)
#How many distinct countries?
Countries = """
        SELECT COUNT(DISTINCT country)
        FROM `bigquery-public-data.openaq.global_air_quality`
        LIMIT 1000
        """
bq_assistant = BigQueryHelper('bigquery-public-data', 'openaq')
df = bq_assistant.query_to_pandas(Countries)
df.head(3)
#How many distinct countries use a unit other than ppm?
countries_ppm = """
        SELECT COUNT(DISTINCT country)
        FROM `bigquery-public-data.openaq.global_air_quality`
        WHERE unit != 'ppm'
        LIMIT 1000
        """
bq_assistant = BigQueryHelper('bigquery-public-data', 'openaq')
df = bq_assistant.query_to_pandas(countries_ppm)
df.head(3)
#List how many countries use each unit
unit_list = """
        SELECT unit, COUNT(DISTINCT country) as number_of_countries
        FROM `bigquery-public-data.openaq.global_air_quality`
        GROUP BY unit
        ORDER BY unit
        LIMIT 1000
        """
bq_assistant = BigQueryHelper('bigquery-public-data', 'openaq')
df = bq_assistant.query_to_pandas(unit_list)
df.head(70)
#I'll start by looking at all readings for each pollutant
readings_by_poll = """
        SELECT DISTINCT pollutant, COUNT(value) as number_of_readings
        FROM `bigquery-public-data.openaq.global_air_quality`
        GROUP BY pollutant
        ORDER BY pollutant
        LIMIT 1000
        """

bq_assistant = BigQueryHelper('bigquery-public-data', 'openaq')
df = bq_assistant.query_to_pandas(readings_by_poll)
df.head(10)
zero_readings = """
        SELECT DISTINCT pollutant, COUNT(value) as number_of_readings
        FROM `bigquery-public-data.openaq.global_air_quality`
        WHERE value=0
        GROUP BY pollutant
        ORDER BY pollutant
        LIMIT 1000
        """

bq_assistant = BigQueryHelper('bigquery-public-data', 'openaq')
df = bq_assistant.query_to_pandas(zero_readings)
df.head(10)