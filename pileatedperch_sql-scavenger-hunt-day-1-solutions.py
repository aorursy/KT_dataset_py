import bq_helper
openaq = bq_helper.BigQueryHelper(active_project = 'bigquery-public-data', dataset_name='openaq')
openaq.list_tables()
openaq.head('global_air_quality')
us_cities = openaq.query_to_pandas_safe("""
SELECT DISTINCT city
FROM `bigquery-public-data.openaq.global_air_quality`
WHERE LOWER(country) = 'us'
ORDER BY city
""")
us_cities
not_ppm = openaq.query_to_pandas_safe("""
SELECT DISTINCT country, unit
FROM `bigquery-public-data.openaq.global_air_quality`
WHERE LOWER(unit) != 'ppm'
ORDER BY country
""")
not_ppm
absent_pollutants = openaq.query_to_pandas_safe("""
SELECT DISTINCT pollutant
FROM `bigquery-public-data.openaq.global_air_quality`
WHERE value = 0.0
ORDER BY pollutant
""")
absent_pollutants