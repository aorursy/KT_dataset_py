import numpy as np

import pandas as pd



import os

print(os.listdir("../input"))



import bq_helper

from bq_helper import BigQueryHelper



import matplotlib.pyplot as plt

import seaborn as sns

sns.set()
San_Francisco = bq_helper.BigQueryHelper(active_project = 'biguery-public-data', dataset_name = 'san_francisco')
SF = BigQueryHelper('bigquery-public-data','san_francisco')

SF.list_tables()
SF.head('bikeshare_stations',10)
query1 = """

SELECT DISTINCT landmark, COUNT(landmark) AS No_of_stations

FROM `bigquery-public-data.san_francisco.bikeshare_stations`

GROUP BY landmark

"""



stations_per_landmark = SF.query_to_pandas_safe(query1)

stations_per_landmark
query2 = """

SELECT DISTINCT landmark, SUM(dockcount) AS docks

FROM `bigquery-public-data.san_francisco.bikeshare_stations`

GROUP BY landmark 

"""

docks_per_landmark = SF.query_to_pandas_safe(query2)

docks_per_landmark
query3 = """

SELECT name, dockcount, installation_date

FROM `bigquery-public-data.san_francisco.bikeshare_stations`

WHERE landmark = 'San Francisco' AND installation_date BETWEEN '2013-01-01' AND '2014-01-01'

"""

ins_sf_2013 = SF.query_to_pandas_safe(query3)

ins_sf_2013
ins_sf_2013['dockcount'].sum()
query4 = """

SELECT name, dockcount, installation_date

FROM `bigquery-public-data.san_francisco.bikeshare_stations`

WHERE landmark = 'San Jose' AND installation_date BETWEEN '2013-01-01' AND '2014-01-01'

"""

ins_sj_2013 = SF.query_to_pandas_safe(query4)

ins_sj_2013
SF.head('film_locations')
query5 = """

SELECT DISTINCT title

FROM `bigquery-public-data.san_francisco.film_locations`

"""

movies = SF.query_to_pandas_safe(query5)

movies
query6 = """

SELECT DISTINCT production_company

FROM `bigquery-public-data.san_francisco.film_locations`

"""



prod_com = SF.query_to_pandas_safe(query6)

prod_com
query7 = """

SELECT DISTINCT distributor

FROM `bigquery-public-data.san_francisco.film_locations`

"""

distributors = SF.query_to_pandas_safe(query7)

distributors
query8 = """

SELECT DISTINCT director

FROM `bigquery-public-data.san_francisco.film_locations`

"""

Directors = SF.query_to_pandas_safe(query8)

Directors
query9 = """

SELECT DISTINCT title, release_year

FROM `bigquery-public-data.san_francisco.film_locations`

WHERE production_company = 'Metro-Goldwyn-Mayer (MGM)' AND distributor = 'Metro-Goldwyn-Mayer (MGM)'

"""



MGM_films = SF.query_to_pandas_safe(query9)

MGM_films
query10 = """

SELECT DISTINCT title, release_year, writer

FROM `bigquery-public-data.san_francisco.film_locations`

WHERE production_company = 'Warner Bros. Pictures' AND distributor = 'Warner Bros. Pictures'

"""



WB_films = SF.query_to_pandas_safe(query10)

WB_films
query11 = """

SELECT DISTINCT release_year , title

FROM `bigquery-public-data.san_francisco.film_locations`

WHERE release_year BETWEEN 1915 AND 1950

ORDER BY release_year

"""

movies_1915_to_1950 = SF.query_to_pandas_safe(query11)

movies_1915_to_1950
query12 = """

SELECT DISTINCT release_year , title

FROM `bigquery-public-data.san_francisco.film_locations`

WHERE release_year BETWEEN 2000 AND 2018

ORDER BY release_year

"""

movies_2000_to_2018 = SF.query_to_pandas_safe(query12)

movies_2000_to_2018
SF.head('sfpd_incidents')
query13 = """

SELECT DISTINCT category, COUNT(unique_key) AS Incidents

FROM `bigquery-public-data.san_francisco.sfpd_incidents`

GROUP BY category

ORDER BY Incidents DESC

"""

sfpd_incident_types = SF.query_to_pandas_safe(query13)

sfpd_incident_types
most_frequent_incidents = sfpd_incident_types.head(10)

most_frequent_incidents.set_index('category', inplace = True)

most_frequent_incidents.plot(kind = 'barh', figsize = (12,6))
query14 = """

SELECT resolution, COUNT(resolution) AS Number

FROM `bigquery-public-data.san_francisco.sfpd_incidents`

GROUP BY resolution

ORDER BY Number DESC



"""

resolution = SF.query_to_pandas_safe(query14)

resolution
resolution.set_index('resolution', inplace = True)

resolution.head().plot(kind = 'barh', figsize = (12,6), color = 'cyan')
query15 = """

SELECT dayofweek, COUNT(unique_key) AS Incidents

FROM `bigquery-public-data.san_francisco.sfpd_incidents`

GROUP BY dayofweek 

ORDER BY Incidents DESC

"""

Inc_day_of_week = SF.query_to_pandas_safe(query15)

Inc_day_of_week
query16 = """

SELECT DISTINCT pddistrict, COUNT(unique_key) AS incidents

FROM `bigquery-public-data.san_francisco.sfpd_incidents`

GROUP BY pddistrict

ORDER BY incidents DESC

"""



districts = SF.query_to_pandas_safe(query16)

districts
query17 = """

SELECT EXTRACT (YEAR FROM timestamp) AS year, COUNT(unique_key) AS incidents

FROM `bigquery-public-data.san_francisco.sfpd_incidents`

GROUP BY year

ORDER BY year



"""

year = SF.query_to_pandas_safe(query17)

year
year.set_index('year', inplace = True)

year.drop(index = 2018, inplace = True)

plt.plot(year, '-o')