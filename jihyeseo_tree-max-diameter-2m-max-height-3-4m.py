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
#BigQuery Tablebigquery-public-data.usfs_fia.tree
# import package with helper functions 
import bq_helper

# create a helper object for this dataset
usfs_fia = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="usfs_fia")

# print all the tables in this dataset (there's only one!)
usfs_fia.list_tables()
usfs_fia.table_schema('tree')

query = """SELECT count(distinct condition_class_number), count(distinct tree_status_code), count(distinct species_code), count(distinct species_common_name), count(distinct species_group_code), count(distinct tree_class_code)
            FROM `bigquery-public-data.usfs_fia.tree`  
            """  

res = usfs_fia.query_to_pandas_safe(query, 1.1)
res
query = """SELECT max(current_diameter), max(actual_height)
            FROM `bigquery-public-data.usfs_fia.tree` 
            """  

res = usfs_fia.query_to_pandas_safe(query)
res
