# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import bq_helper
from bq_helper import BigQueryHelper
# https://www.kaggle.com/sohier/introduction-to-the-bq-helper-package
usfs = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="usfs_fia")

bq_assistant = BigQueryHelper("bigquery-public-data", "usfs_fia")
bq_assistant.list_tables()
bq_assistant.head("plot_tree",num_rows=20)
# Note: State and county are FIPS state codes.
query1 = """
SELECT
    plot_sequence_number,
    plot_state_code,
    plot_county_code,
    measurement_year,
    latitude,
    longitude,
    tree_sequence_number,
    species_code,
    current_diameter
FROM
    `bigquery-public-data.usfs_fia.plot_tree`
WHERE
    plot_state_code_name = 'Utah'
;
        """
response1 = usfs.query_to_pandas_safe(query1, max_gb_scanned=10)
response1.head(20)
response1.plot.scatter('latitude','longitude')
import folium
from folium.plugins import HeatMap
len(response1)
latitude, longitude = 39.5, -111.5
utah_map = folium.Map(location = [latitude, longitude], zoom_start = 6,tiles= 'Stamen Terrain')
trees = response1[['latitude','longitude']]
HeatMap(trees).add_to(utah_map)
utah_map

