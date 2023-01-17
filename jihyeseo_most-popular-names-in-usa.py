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
usa_names = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="usa_names")

# print all the tables in this dataset (there's only one!)
usa_names.list_tables()
usa_names.table_schema('usa_1910_2013')
usa_names.table_schema('usa_1910_current')
query = """SELECT gender, name, year, sum(number) as number 
            FROM `bigquery-public-data.usa_names.usa_1910_current` 
            GROUP BY gender, name, year
            """  
res = usa_names.query_to_pandas_safe(query, 11)
countPerGenderNameYear = res.copy()
countPerGenderNameYear.head()
countPerGenderName = countPerGenderNameYear.groupby(['gender',
                                                 'name']).number.sum()
countPerGenderName.head()
countPerGenderName = countPerGenderName.reset_index()

countPerGenderName.sort_values(by = 'number', ascending = False)
countPerGenderName = countPerGenderName.set_index('name')
countPerGenderName.head()
