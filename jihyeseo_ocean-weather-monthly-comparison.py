import math
import seaborn as sns
sns.set(style="whitegrid", color_codes=True)


from wordcloud import WordCloud, STOPWORDS

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np # linear algebra
import matplotlib 
import matplotlib.pyplot as plt
import sklearn
%matplotlib inline
import matplotlib.pyplot as plt 
plt.rcParams["figure.figsize"] = [16, 12]
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
# import package with helper functions 
import bq_helper

# create a helper object for this dataset
noaa_icoads = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="noaa_icoads")

# print all the tables in this dataset (there's only one!)
noaa_icoads.list_tables()
noaa_icoads.table_schema('icoads_core_2017')
noaa_icoads.table_schema('icoads_core_1662_2000')
query = """SELECT year, month, avg(wind_speed), avg(air_temperature), avg(nightday_flag), avg(wave_height), avg(sea_surface_temp), avg(sea_level_pressure), avg(visibility)
            FROM `bigquery-public-data.noaa_icoads.icoads_core_2017` 
            GROUP BY year, month
            """  

res = noaa_icoads.query_to_pandas_safe(query)
query = """SELECT year, month, hour, avg(wind_speed), avg(air_temperature), avg(nightday_flag), avg(wave_height), avg(sea_surface_temp), avg(sea_level_pressure), avg(visibility)
            FROM `bigquery-public-data.noaa_icoads.icoads_core_2017` 
            GROUP BY year, month, hour
            """  

res = noaa_icoads.query_to_pandas_safe(query)