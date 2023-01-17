import bq_helper 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')
# create a helper object for this dataset
open_aq = bq_helper.BigQueryHelper(active_project="bigquery-public-data", 
                                       dataset_name="openaq")

# print all the tables in this dataset
open_aq.list_tables()
open_aq.table_schema('global_air_quality')
# Query for all countries where unit is not ppm
query = """
        SELECT DISTINCT country, unit
        FROM `bigquery-public-data.openaq.global_air_quality`
        WHERE LOWER(unit) != 'ppm'
        ORDER BY country
        """

# Print estimated size of query
print("Query size estimated at {:0.4f} GB".format(open_aq.estimate_query_size(query)))
country_units = open_aq.query_to_pandas_safe(query)
country_units
# Query for all pollutants where values is 0
query = """
        SELECT DISTINCT pollutant, value
        FROM `bigquery-public-data.openaq.global_air_quality`
        WHERE value = 0
        ORDER BY pollutant
        """

# Print estimated size of query
print("Query size estimated at {:0.4f} GB".format(open_aq.estimate_query_size(query)))
# Safe query to pandas dataframe
pollutant_value = open_aq.query_to_pandas_safe(query)

# Print results
pollutant_value