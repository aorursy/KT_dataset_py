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
#BigQuery Tablebigquery-public-data.open_images.labels
# import package with helper functions 
import bq_helper

# create a helper object for this dataset
open_images = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="open_images")

# print all the tables in this dataset (there's only one!)
open_images.list_tables()
open_images.table_schema('labels')
query = """SELECT label_name, count(image_id) as count
            FROM `bigquery-public-data.open_images.labels` 
            GROUP BY label_name
            ORDER BY count DESC
            LIMIT 10
            """  

res = open_images.query_to_pandas_safe(query, 2.5)
res
