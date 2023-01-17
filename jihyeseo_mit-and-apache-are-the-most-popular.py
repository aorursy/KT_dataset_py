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
#BigQuery Tablebigquery-public-data.libraries_io.dependencies

# import package with helper functions 
import bq_helper

# create a helper object for this dataset
libraries_io = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="libraries_io")

# print all the tables in this dataset (there's only one!)
libraries_io.list_tables()
libraries_io.table_schema('projects')
query = """SELECT distinct licenses
            FROM `bigquery-public-data.libraries_io.projects`
            """  

res = libraries_io.query_to_pandas_safe(query)
res
query = """SELECT distinct licenses, count(id) as count
            FROM `bigquery-public-data.libraries_io.projects`
            GROUP BY licenses
            ORDER BY count DESC
            LIMIT 15
            """  

res = libraries_io.query_to_pandas_safe(query)
res = res.set_index('licenses')
res.plot.bar()
