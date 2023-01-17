# import package with helper functions 
import bq_helper

# create a helper object for this dataset
open_aq = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                              dataset_name="openaq")
# check tables' names
open_aq.list_tables()
# check head of the table
open_aq.head('global_air_quality')
# check the table schema
open_aq.table_schema('global_air_quality')
# check how many measure station are there in each country
query_1 = """SELECT country, COUNT(location) AS number_of_locations
                FROM `bigquery-public-data.openaq.global_air_quality`
                GROUP BY country
                ORDER by number_of_locations DESC"""

number_of_locations = open_aq.query_to_pandas_safe(query_1, max_gb_scanned=0.1)
number_of_locations
# save the list of countries to get area data for the from World Bank
number_of_locations.to_csv("number_of_stations.csv")
# check the kind of pollutants
query_2 = """SELECT pollutant, SUM(value) AS total_pollution
                FROM `bigquery-public-data.openaq.global_air_quality`
                GROUP BY pollutant
                ORDER by total_pollution DESC"""

pollutants = open_aq.query_to_pandas_safe(query_2, max_gb_scanned=0.1)
pollutants
query_2a = """
            SELECT *
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE value < -900
            """
minus_values = open_aq.query_to_pandas_safe(query_2a, max_gb_scanned=0.1)
minus_values
query_2b = """WITH normalised_pollution AS
                (
                SELECT pollutant,
                CASE
                    WHEN unit = 'ppm' AND pollutant ='o3' AND value > 0 THEN value
                    WHEN unit = 'µg/m³' AND pollutant ='o3' AND value > 0 THEN value/1960
                    WHEN unit = 'ppm' AND pollutant ='no2' AND value > 0 THEN value
                    WHEN unit = 'µg/m³' AND pollutant ='no2' AND value > 0 THEN value/1880
                    WHEN unit = 'ppm' AND pollutant ='co' AND value > 0 THEN value
                    WHEN unit = 'µg/m³' AND pollutant ='co' AND value > 0 THEN value/1150
                    WHEN unit = 'ppm' AND pollutant ='so2' AND value > 0 THEN value
                    WHEN unit = 'µg/m³' AND pollutant ='so2' AND value > 0 THEN value/2620
                END AS ppm_unit
                FROM `bigquery-public-data.openaq.global_air_quality`
                )
                SELECT pollutant, SUM(ppm_unit) as total_ppm_pollution
                FROM normalised_pollution
                GROUP BY pollutant
                ORDER BY total_ppm_pollution DESC"""

pollutants_above_zero = open_aq.query_to_pandas_safe(query_2b, max_gb_scanned=0.1)
pollutants_above_zero
query_4a = """WITH normalised_pollution AS
                (
                SELECT country, pollutant, location,
                    CASE
                        WHEN unit = 'ppm' AND pollutant ='o3' AND value > 0 THEN value*1960
                        WHEN unit = 'µg/m³' AND pollutant ='o3' AND value > 0 THEN value
                        WHEN unit = 'ppm' AND pollutant ='no2' AND value > 0 THEN value*1880
                        WHEN unit = 'µg/m³' AND pollutant ='no2' AND value > 0 THEN value
                        WHEN unit = 'ppm' AND pollutant ='co' AND value > 0 THEN value*1150
                        WHEN unit = 'µg/m³' AND pollutant ='co' AND value > 0 THEN value
                        WHEN unit = 'ppm' AND pollutant ='so2' AND value > 0 THEN value*2620
                        WHEN unit = 'µg/m³' AND pollutant ='so2' AND value > 0 THEN value
                END AS micro_unit_value
                FROM `bigquery-public-data.openaq.global_air_quality`
                )
                SELECT country, pollutant, (SUM(micro_unit_value)/COUNT(*)) AS current_pollution
                    FROM normalised_pollution
                    WHERE micro_unit_value > 0
                    GROUP BY country, pollutant
                    ORDER by country, pollutant """
pollution = open_aq.query_to_pandas_safe(query_4a, max_gb_scanned=0.1)
pollution.head(15)
import matplotlib.pyplot as plt
import seaborn as sns

one_pollutant = pollution.loc[pollution['pollutant'] == 'o3']
ordered_pollution = one_pollutant.sort_values(by='current_pollution')
my_range=range(1,len(ordered_pollution.index)+1)

plt.figure(figsize=(15,10))
plt.hlines(y=my_range, xmin=0, xmax=ordered_pollution['current_pollution'], color='skyblue')
plt.plot(ordered_pollution['current_pollution'], my_range, "o")
plt.yticks(my_range, ordered_pollution['country'])
plt.title("Current pollution of in various countries", loc='left')
plt.xlabel('current pollution (ug/m3)')
plt.ylabel('Country')
