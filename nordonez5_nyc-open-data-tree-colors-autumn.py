import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import pandas as pd

from google.cloud import bigquery
client = bigquery.Client()

nyc_open_dataset_ref = client.dataset("new_york", project="bigquery-public-data")

nyc_open_dataset = client.get_dataset(nyc_open_dataset_ref)



nyc_open_tables = list(client.list_tables(nyc_open_dataset))



for table in nyc_open_tables:

    print(table.table_id)
# Select tree species information and most recent tree census information.

client = bigquery.Client()

census_2015_table_ref = nyc_open_dataset_ref.table('tree_census_2015')

census_2015_table = client.get_table(census_2015_table_ref)

tree_species_ref = nyc_open_dataset_ref.table('tree_species')

tree_species = client.get_table(tree_species_ref)
client.list_rows(census_2015_table, max_results=3).to_dataframe()
client.list_rows(tree_species, max_results=3).to_dataframe()
tree_query = """

SELECT S.species_scientific_name AS Species, S.species_common_name, C.latitude, C.longitude, S.fall_color 

FROM `bigquery-public-data.new_york.tree_species` AS S

INNER JOIN `bigquery-public-data.new_york.tree_census_2015` AS C

    ON S.species_scientific_name = C.spc_latin

"""

safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=10**10)

query_job = client.query(tree_query, job_config=safe_config)



tree_loc_color = query_job.to_dataframe()
tree_loc_color
plt.subplots(figsize=(20,20))

sns.scatterplot(data=tree_loc_color, x='longitude', y='latitude', hue='fall_color', sizes = 4)