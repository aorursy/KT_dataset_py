# import package with helper functions 
import bq_helper

# create a helper object for this dataset
open_aq = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="openaq")

# print all the tables in this dataset (there's only one!)
open_aq.list_tables()
# Which countries use a unit other than ppm to measure any type of pollution?
q1 = """SELECT DISTINCT country
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE unit != 'ppm'
        """
non_ppm = open_aq.query_to_pandas_safe(q1)
print("Found {} countries not using ppm".format(len(non_ppm)))
non_ppm.head()
# Which countries use a unit other than ppm to measure any type of pollution?
# (Version 2 - record a boolean per country for context)
q1 = """SELECT country, count(unit != 'ppm') > 0 AS non_ppm
            FROM `bigquery-public-data.openaq.global_air_quality`
            GROUP BY country
        """
df = open_aq.query_to_pandas_safe(q1)
print("Found {} / {} countries not using ppm".format(
    (df['non_ppm']).sum(), len(df),
))
df.head()
# Which pollutants have a value of exactly 0? [ever, I guess?]
q2 = """SELECT pollutant, count(value = 0) > 0 AS ever_zero
            FROM `bigquery-public-data.openaq.global_air_quality`
            GROUP BY pollutant
        """
df = open_aq.query_to_pandas_safe(q2)
print("Found {} / {} pollutants that have ever had a value of 0".format(
    (df['ever_zero']).sum(), len(df),
))
df.head()