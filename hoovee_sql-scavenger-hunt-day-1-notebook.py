import bq_helper

# create helper object for the air pollution measurement data
air_q_data = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                      dataset_name="openaq")

air_q_data.list_tables()

# print the first couple of rows of the "global_air_quality" dataset
air_q_data.head("global_air_quality")
# create query
query = """SELECT DISTINCT country, unit
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE LOWER(unit) != 'ppm'
            ORDER BY country
        """
print("Size of the data:", air_q_data.estimate_query_size(query))

countries_not_ppm = air_q_data.query_to_pandas_safe(query)
# show results
countries_not_ppm
# create query
# get country, pollutant and see which county has total value of 0
query2 = """SELECT country, pollutant, SUM(value) AS total
            FROM `bigquery-public-data.openaq.global_air_quality`
            GROUP BY country, pollutant
            HAVING total = 0
         """

print("Size of the data:", air_q_data.estimate_query_size(query2))

pollutants = air_q_data.query_to_pandas_safe(query2)
# show results
pollutants
# create query
# get pollutants that have 0 value and order them
query3 = """SELECT DISTINCT pollutant
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE value = 0
            ORDER BY pollutant
         """

print("Size of the data:", air_q_data.estimate_query_size(query3))

pollutants2 = air_q_data.query_to_pandas_safe(query3)
pollutants2