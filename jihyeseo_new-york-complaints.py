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

#BigQuery Tablebigquery-public-data..311_service_requests

# create a helper object for this dataset
new_york = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="new_york")

# print all the tables in this dataset (there's only one!)
new_york.list_tables()
new_york.table_schema('311_service_requests')
# Your Code Goes Here
query = """SELECT distinct complaint_type
            FROM `bigquery-public-data.new_york.311_service_requests` 
            """
res = new_york.query_to_pandas_safe(query)
type(res)
res