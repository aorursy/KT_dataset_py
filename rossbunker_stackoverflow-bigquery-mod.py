import bq_helper

from bq_helper import BigQueryHelper

stackOverflow = bq_helper.BigQueryHelper(active_project="bigquery-public-data",

                                   dataset_name="stackoverflow")
query = """SELECT

  EXTRACT(YEAR FROM creation_date) AS Year,

  COUNT(*) AS Number_of_Questions,

FROM

  `bigquery-public-data.stackoverflow.posts_questions`

WHERE

     tags LIKE "%python%" AND (REGEXP_CONTAINS(tags, r"tensorflow|pytorch|sklearn|caffe|mxnet|keras|opencv|neural"))

GROUP BY

  Year

HAVING

  Year > 2012

ORDER BY

  Year;

        """

response = stackOverflow.query_to_pandas_safe(query)

response.head(10)
query = """SELECT

  EXTRACT(YEAR FROM creation_date) AS creation_date,

  tags

FROM

  `bigquery-public-data.stackoverflow.posts_questions`

WHERE

    tags LIKE "%python%" AND (REGEXP_CONTAINS(tags, r"tensorflow|pytorch|sklearn|caffe|mxnet|keras|opencv|neural"))

    AND EXTRACT(YEAR FROM creation_date) > 2012

        """

response = stackOverflow.query_to_pandas_safe(query)

response.head(10)

response['splits'] = response['tags'].apply(lambda x: x.split("|"))

response.head()
import pandas as pd

new = []

for idx, row in response.iterrows():

    year = row['creation_date']

    for item in row['splits']:

        t = {}

        t['year'] = year

        t['tag']  = item

        new.append(t)

df = pd.DataFrame(new)

df.head()
df.to_csv("tags_by_year.csv", index=False)