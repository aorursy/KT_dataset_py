# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from ggplot import *
import matplotlib.pyplot as plt
import seaborn as sns
import bq_helper 

# create a helper object
global_aq = bq_helper.BigQueryHelper(active_project= "bigquery-public-data", dataset_name = "openaq")
# list all the tables
global_aq.list_tables()
# show table schema
global_aq.table_schema("global_air_quality")
# show first 5 rows
global_aq.head("global_air_quality")
query = """
            SELECT DISTINCT pollutant, unit
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE value > 0
        """

pollutants = global_aq.query_to_pandas_safe(query)

print(pollutants)
query = """
            SELECT value AS negative_value, COUNT(value) AS n
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE value < 0
            GROUP BY value
            ORDER BY value
        """

negative_values = global_aq.query_to_pandas_safe(query)

print(negative_values)
query = """
            WITH same_unit AS
            (
                SELECT *, 
                    CASE
                        WHEN pollutant = 'so2' AND unit = 'ppm' THEN 40.9 * 64 * value
                        WHEN pollutant = 'no2' AND unit = 'ppm' THEN 40.9 * 46 * value
                        WHEN pollutant = 'o3' AND unit = 'ppm' THEN 40.9 * 48 * value
                        WHEN pollutant = 'co' AND unit = 'ppm' THEN 40.9 * 28 * value
                        ELSE value
                    END AS value_ugm3
                FROM `bigquery-public-data.openaq.global_air_quality`
            )
            SELECT location, city, country, L.pollutant, L.value_ugm3, DATE(timestamp) AS date, source_name
            FROM same_unit AS L
            INNER JOIN
            (
                SELECT pollutant, MAX(value_ugm3) AS max_value
                FROM same_unit
                GROUP BY pollutant
            ) AS R
            ON L.pollutant = R.pollutant AND L.value_ugm3 = R.max_value
            ORDER BY L.country
        """

max_pollutant = global_aq.query_to_pandas_safe(query)
print(max_pollutant)
query = """
            WITH same_unit AS
            (
                SELECT pollutant, value, unit, EXTRACT(YEAR FROM timestamp) as year,
                    CASE
                        WHEN pollutant = 'so2' AND unit = 'ppm' THEN 40.9 * 64 * value
                        WHEN pollutant = 'no2' AND unit = 'ppm' THEN 40.9 * 46 * value
                        WHEN pollutant = 'o3' AND unit = 'ppm' THEN 40.9 * 48 * value
                        WHEN pollutant = 'co' AND unit = 'ppm' THEN 40.9 * 28 * value
                        ELSE value
                    END AS value_ugm3
                FROM `bigquery-public-data.openaq.global_air_quality`
                WHERE value >= 0
            )
            SELECT year, pollutant, COUNT(value_ugm3) AS n_measure, 
                   MIN(value_ugm3) AS min, MAX(value_ugm3) AS max,
                   AVG(value_ugm3) AS average, STDDEV_SAMP(value_ugm3) AS std, 
                   LOG10(AVG(value_ugm3)) AS log10_avg
            FROM same_unit
            GROUP BY year, pollutant
            ORDER BY year, pollutant
        """

pollutant_by_year = global_aq.query_to_pandas_safe(query)
print(pollutant_by_year)
ggplot(aes(x = 'year', y = 'log10_avg', color = 'pollutant'), data = pollutant_by_year) +\
    geom_point(size = 8) +\
    geom_line() +\
    ylab('log10 of Average level')
query = """
            WITH same_unit AS
            (
                SELECT country, city, pollutant, value, unit, DATE(timestamp) as date,
                    CASE
                        WHEN pollutant = 'so2' AND unit = 'ppm' THEN 40.9 * 64 * value
                        WHEN pollutant = 'no2' AND unit = 'ppm' THEN 40.9 * 46 * value
                        WHEN pollutant = 'o3' AND unit = 'ppm' THEN 40.9 * 48 * value
                        WHEN pollutant = 'co' AND unit = 'ppm' THEN 40.9 * 28 * value
                        ELSE value
                    END AS value_ugm3
                FROM `bigquery-public-data.openaq.global_air_quality`
                WHERE value >= 0
            ),
            summary AS
            (
                SELECT country, pollutant, COUNT(value_ugm3) AS total_values, 
                   MIN(value_ugm3) AS min_value, MAX(value_ugm3) AS max_value,
                   AVG(value_ugm3) AS avg_value, STDDEV_SAMP(value_ugm3) AS std,
                   CASE
                       WHEN AVG(value_ugm3) <= 0 THEN 0
                       ELSE LOG10(AVG(value_ugm3))
                   END AS log10_avg
                FROM same_unit
                GROUP BY country, pollutant
                HAVING avg_value > 0
                ORDER BY country
            )
            SELECT L.*
            FROM summary AS L
            INNER JOIN
            (
                SELECT pollutant, MAX(avg_value) as max_avg
                FROM summary
                WHERE total_values >= 10
                GROUP BY pollutant
            ) AS R
            ON L.pollutant = R.pollutant AND L.avg_value = R.max_avg
            ORDER BY pollutant
        """

pollutant_top_country = global_aq.query_to_pandas_safe(query)
print(pollutant_top_country)
query = """
            WITH same_unit AS
            (
                SELECT country, city, pollutant, value, unit, DATE(timestamp) as date,
                    CASE
                        WHEN pollutant = 'so2' AND unit = 'ppm' THEN 40.9 * 64 * value
                        WHEN pollutant = 'no2' AND unit = 'ppm' THEN 40.9 * 46 * value
                        WHEN pollutant = 'o3' AND unit = 'ppm' THEN 40.9 * 48 * value
                        WHEN pollutant = 'co' AND unit = 'ppm' THEN 40.9 * 28 * value
                        ELSE value
                    END AS value_ugm3
                FROM `bigquery-public-data.openaq.global_air_quality`
                WHERE value >= 0
            ),
            summary AS
            (
                SELECT country, pollutant, COUNT(value_ugm3) AS total_values, 
                   MIN(value_ugm3) AS min_value, MAX(value_ugm3) AS max_value,
                   AVG(value_ugm3) AS avg_value, STDDEV_SAMP(value_ugm3) AS std,
                   CASE
                       WHEN AVG(value_ugm3) <= 0 THEN 0
                       ELSE LOG10(AVG(value_ugm3))
                   END AS log10_avg
                FROM same_unit
                GROUP BY country, pollutant
                HAVING avg_value > 0
                ORDER BY country
            )
            SELECT *
            FROM summary 
            WHERE total_values >= 10
        """

pollutant_by_country = global_aq.query_to_pandas_safe(query)
# CO level
pollutant_co = pollutant_by_country[(pollutant_by_country.pollutant == 'co')].sort_values(by = 'avg_value', ascending = False)

fig, ax = plt.subplots(figsize = (15, 5))
country_co = sns.pointplot(x = "country", y = "log10_avg", data = pollutant_co.head(n = 10), ax = ax)
# CO level
pollutant_co = pollutant_by_country[(pollutant_by_country.pollutant == 'co')].sort_values(by = 'avg_value', ascending = False)

fig, ax = plt.subplots(figsize = (15, 5))
country_co = sns.pointplot(x = "country", y = "log10_avg", data = pollutant_co.head(n = 10), ax = ax)
country_co.set(ylabel = "Log10(Average CO level)")
# SO2 level
pollutant_so2 = pollutant_by_country[(pollutant_by_country.pollutant == 'so2')].sort_values(by = 'avg_value', ascending = False)

fig, ax = plt.subplots(figsize = (15, 5))
country_so2 = sns.pointplot(x = "country", y = "log10_avg", data = pollutant_so2.head(n = 10), ax = ax)
country_so2.set(ylabel = "Log10(Average SO2 level)")
# no2 level
pollutant_no2 = pollutant_by_country[(pollutant_by_country.pollutant == 'no2')].sort_values(by = 'avg_value', ascending = False)

fig, ax = plt.subplots(figsize = (15, 5))
country_no2 = sns.pointplot(x = "country", y = "log10_avg", data = pollutant_no2.head(n = 10), ax = ax)
country_no2.set(ylabel = "Log10(Average NO2 level)")
# O3 level
pollutant_o3 = pollutant_by_country[(pollutant_by_country.pollutant == 'o3')].sort_values(by = 'avg_value', ascending = False)

fig, ax = plt.subplots(figsize = (15, 5))
country_o3 = sns.pointplot(x = "country", y = "log10_avg", data = pollutant_o3.head(n = 10), ax = ax)
country_o3.set(ylabel = "Log10(Average O3 level)")
# black carbon level
pollutant_bc = pollutant_by_country[(pollutant_by_country.pollutant == 'bc')].sort_values(by = 'avg_value', ascending = False)

fig, ax = plt.subplots(figsize = (15, 5))
country_bc = sns.pointplot(x = "country", y = "log10_avg", data = pollutant_bc.head(n = 10), ax = ax)
country_bc.set(ylabel = "Log10(Average Black Carbon level)")
# pm10 level
pollutant_pm10 = pollutant_by_country[(pollutant_by_country.pollutant == 'pm10')].sort_values(by = 'avg_value', ascending = False)

fig, ax = plt.subplots(figsize = (15, 5))
country_pm10 = sns.pointplot(x = "country", y = "log10_avg", data = pollutant_pm10.head(n = 10), ax = ax)
country_pm10.set(ylabel = "Log10(Average PM10 level)")
# pm25 level
pollutant_pm25 = pollutant_by_country[(pollutant_by_country.pollutant == 'pm25')].sort_values(by = 'avg_value', ascending = False)

fig, ax = plt.subplots(figsize = (15, 5))
country_pm25 = sns.pointplot(x = "country", y = "log10_avg", data = pollutant_pm25.head(n = 10), ax = ax)
country_pm25.set(ylabel = "Log10(Average PM25 level)")