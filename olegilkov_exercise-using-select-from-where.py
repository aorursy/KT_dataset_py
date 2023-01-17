# import package with helper functions 
import bq_helper

# create a helper object for this dataset
open_aq = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="openaq")

# print all the tables in this dataset (there's only one!)
open_aq.list_tables()
query = """SELECT country
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE unit != 'ppm' 
        """
countries = open_aq.query_to_pandas_safe(query, max_gb_scanned=0.1)
countries_values =[]
for c in countries.country:
    if c not in countries_values: countries_values.append(c)

print (countries_values)
query = """SELECT pollutant
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE value = 0 
        """
pollutants = open_aq.query_to_pandas_safe(query, max_gb_scanned=0.1)
pollutants_values =[]
for p in pollutants.pollutant:
    if p not in pollutants_values: pollutants_values.append(p)
print(pollutants_values)