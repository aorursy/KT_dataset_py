# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from matplotlib import pyplot



from google.cloud import bigquery



# Query: total number of questions per year in stackoverflow

bq_c = bigquery.Client()

query_job = bq_c.query("""SELECT DATE_TRUNC(DATE(creation_date), MONTH) as year, count(*) as total_questions

FROM `bigquery-public-data.stackoverflow.posts_questions` 

GROUP BY year

ORDER BY year""")



# Conver query result to dataframe

df = query_job.to_dataframe()
# Draw trend line

pyplot.title('stackoverflow # of posts questions per year') 



pyplot.plot(df.get('year'), df.get('total_questions'))

pyplot.xlabel('year') 

pyplot.ylabel('total') 



pyplot.show() 