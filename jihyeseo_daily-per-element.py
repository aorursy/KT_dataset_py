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
#BigQuery Tablebigquery-public-data.ghcn_d.ghcnd_2018
# import package with helper functions 
import bq_helper

# create a helper object for this dataset
ghcn_d = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="ghcn_d")

# print all the tables in this dataset (there's only one!)
ghcn_d.list_tables()
ghcn_d.table_schema('ghcnd_2018')
query = """SELECT date, element, sum(value) as sum
            FROM `bigquery-public-data.ghcn_d.ghcnd_2018` 
            GROUP BY date, element
            """  

res = ghcn_d.query_to_pandas_safe(query)
res
