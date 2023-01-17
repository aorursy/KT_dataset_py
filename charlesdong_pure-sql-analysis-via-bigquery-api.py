import pandas as pd
from google.cloud import bigquery
client = bigquery.Client()
# Check the quality first!
query1 = """
SELECT avg(value) AS avg_value, min(value) AS min_value, MAX(value) AS max_value, pollutant
FROM `bigquery-public-data.openaq.global_air_quality`
GROUP BY pollutant
"""
list(client.query(query1).result(timeout=100))
# Check how many rows containing negative value
query2 = """
SELECT COUNT(*) FROM `bigquery-public-data.openaq.global_air_quality`
WHERE value < 0
"""
list(client.query(query2).result(timeout=100))[0].values()
# Check which is the most severe pollutants in the world

query3 = """
SELECT count(*) AS freq, pollutant
FROM `bigquery-public-data.openaq.global_air_quality`
WHERE value >= 0
GROUP BY pollutant
ORDER BY freq DESC
"""
list(client.query(query3).result(timeout=100))
# find the biggest pollution source in each country, 
# and result is presented in alphebetical order

query4 = """
SELECT f.freq, f.country, f.pollutant
FROM
(
SELECT COUNT(*) AS freq, country, pollutant, 
       ROW_NUMBER() OVER (PARTITION BY country ORDER BY COUNT(*) DESC) AS rownum
FROM `bigquery-public-data.openaq.global_air_quality`
WHERE value >= 0
GROUP BY country, pollutant
ORDER BY freq DESC
) f
WHERE f.rownum = 1
ORDER BY f.country
"""
list(client.query(query4).result(timeout=100))

def print_queried_cities_pollution(pollutant):
    
    client = bigquery.Client()
    query = """
        SELECT city, AVG(value) AS value, AVG(latitude) AS latitude, AVG(longitude) AS longitude
        FROM `bigquery-public-data.openaq.global_air_quality`
        WHERE value>=0 and pollutant = @pollutant
        GROUP BY city
        """
    query_params = [
        bigquery.ScalarQueryParameter('pollutant', 'STRING', pollutant)
    ]
    job_config = bigquery.QueryJobConfig()
    job_config.query_parameters = query_params
    query_job = client.query(query, job_config=job_config)

    query_job.result()
    # Alternative to print the result row-wisely.  
    destination_table = query_job.destination
    table = client.get_table(destination_table)
    for row in client.list_rows(table):
        print(row)

print_queried_cities_pollution('o3')

