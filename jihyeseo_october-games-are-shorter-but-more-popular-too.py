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
baseball = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="baseball")

# print all the tables in this dataset (there's only one!)
baseball.list_tables()
baseball.table_schema('schedules')
baseball.table_schema('games_wide')
baseball.table_schema('games_post_wide')
# Your Code Goes Here
query = """SELECT extract(month from startTime) as month, dayNight, avg(attendance) as attendance, avg(duration_minutes) as duration_minutes
            FROM `bigquery-public-data.baseball.schedules`
            GROUP BY dayNight, extract(month from startTime)
            """
res = baseball.query_to_pandas_safe(query)
res.head()
sns.barplot(x = 'month', y = 'attendance', hue = 'dayNight', data = res)
sns.barplot(x = 'month', y = 'duration_minutes', hue = 'dayNight', data = res)