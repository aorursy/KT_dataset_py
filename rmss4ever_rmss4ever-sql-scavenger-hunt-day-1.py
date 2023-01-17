# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
# print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
#  Import packages 
import numpy as np
import pandas as pd
import bq_helper as bh

# Create helper object
open_aq = bh.BigQueryHelper(active_project = 'bigquery-public-data', dataset_name = 'openaq')
# List all tables included in openaq dataset
open_aq.list_tables()
# Show column information for table global_air_quality
open_aq.table_schema('global_air_quality')
# Print first few rows of openaq dataset to see what is inside
open_aq.head('global_air_quality')
# Select all cities monitored in the US
query_us_cities = """
                     Select city
                     From `bigquery-public-data.openaq.global_air_quality`
                     Where country = 'US'
                  """
# Run estimate on how big query might be:
open_aq.estimate_query_size(query_us_cities)
# Ok, query size is pretty small but still, let's use pandas_safe(query) just to remain safe
us_cities = open_aq.query_to_pandas_safe(query_us_cities, max_gb_scanned = 0.1)
# Top five cities having measurements
us_cities.city.value_counts()
# Question 1: Which countries use a unit other than 'ppm' to measure any type of pollution?
query_non_ppm_countries = """
                            Select country, pollutant
                            From `bigquery-public-data.openaq.global_air_quality`
                            Where unit != 'ppm'
                            Order By country Asc
                        """
# Check size of query_non_ppm_countries
open_aq.estimate_query_size(query_non_ppm_countries)
# Run query in pandas safe mode max set to 1 gb
non_ppm_countries = open_aq.query_to_pandas_safe(query_non_ppm_countries, max_gb_scanned = 0.1)
# Country and number of cites not using ppm method
non_ppm_countries.country.value_counts()
# Question 2: Which pollutants have a value of exactly 0?
query_pollutant_value_0 = """
                            Select pollutant, value
                            From `bigquery-public-data.openaq.global_air_quality`
                            Where value = 0
                        """
# Run query_pollutant_value_0 in pandas safe for less than 100M
pollutant_value = open_aq.query_to_pandas_safe(query_pollutant_value_0, max_gb_scanned = 0.1)
# Show pollutant_value query  # Show pollutant_value.pollutant query     # Show counts of pollutant_value query
pollutant_value # pollutant_value.pollutant # pollutant_value.pollutant.value_counts()
# Trying to find a 'ppm' unit
query_us_unit_value_ppm = """
                            Select country, unit
                            From `bigquery-public-data.openaq.global_air_quality`
                            Where unit = 'ppm'
                            And country = 'US'
                            Order By city ASC
                        """
# Check size of query_us_unit_value_ppm
open_aq.estimate_query_size(query_us_unit_value_ppm)
# Verifying there really is a 'ppm' unit
open_aq.query_to_pandas_safe(query_us_unit_value_ppm, max_gb_scanned = 0.1)
