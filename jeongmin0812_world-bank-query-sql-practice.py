# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from google.cloud import bigquery

client = bigquery.Client()
dataset_ref = client.dataset('world_bank_intl_education', project='bigquery-public-data')
wb_dset = client.get_dataset(dataset_ref)
type(wb_dset)
[x.table_id for x in client.list_tables(wb_dset)]
wb_full = client.get_table(wb_dset.table('country_series_definitions'))
type(wb_full)
wb_full.schema
query = """SELECT *

                FROM `bigquery-public-data.world_bank_intl_education.country_series_definitions`

                LIMIT 5

                """



query_job = client.query(query)

query_job.to_dataframe()
query = """SELECT *

                FROM `bigquery-public-data.world_bank_intl_education.country_summary`

                LIMIT 5

                """



query_job = client.query(query)

query_job.to_dataframe()
query = """SELECT *

                FROM `bigquery-public-data.world_bank_intl_education.international_education`

                LIMIT 5

                """



query_job = client.query(query)

query_job.to_dataframe()
query = """SELECT A.country_code, A.series_code

                FROM `bigquery-public-data.world_bank_intl_education.country_series_definitions` A

                WHERE A.series_code in (select B.indicator_code from `bigquery-public-data.world_bank_intl_education.international_education` B)

                ORDER by A.country_code

                LIMIT 20

                """



query_job = client.query(query)

query_job.to_dataframe()
query = """SELECT A.country_code, A.description, A.series_code, B.indicator_code

                FROM `bigquery-public-data.world_bank_intl_education.international_education` B

                JOIN `bigquery-public-data.world_bank_intl_education.country_series_definitions` A ON (B.indicator_code=A.series_code)

                ORDER by A.country_code

                LIMIT 100

                """



query_job = client.query(query)

query_job.to_dataframe()
query = """SELECT DISTINCT A.country_code, A.description, A.series_code, B.indicator_code

                FROM `bigquery-public-data.world_bank_intl_education.international_education` B

                JOIN `bigquery-public-data.world_bank_intl_education.country_series_definitions` A ON (B.indicator_code=A.series_code)

                ORDER by A.country_code

                LIMIT 100

                """



query_job = client.query(query)

query_job.to_dataframe()
query = """SELECT DISTINCT A.country_code, B.short_name, A.series_code, A.description, C.value, C.year

                FROM `bigquery-public-data.world_bank_intl_education.country_series_definitions` A

                JOIN `bigquery-public-data.world_bank_intl_education.country_summary` B ON A.country_code = B.country_code

                JOIN `bigquery-public-data.world_bank_intl_education.international_education` C ON A.country_code = C.country_code

                ORDER by A.country_code, C.year DESC

                LIMIT 10

                """



query_job = client.query(query)

query_job.to_dataframe()
query = """SELECT DISTINCT C.year, count(*)

                FROM `bigquery-public-data.world_bank_intl_education.country_series_definitions` A

                JOIN `bigquery-public-data.world_bank_intl_education.country_summary` B ON A.country_code = B.country_code

                JOIN `bigquery-public-data.world_bank_intl_education.international_education` C ON A.country_code = C.country_code

                GROUP BY C.year

                ORDER by C.year

                """



query_job = client.query(query)

query_job.to_dataframe()
query = """SELECT DISTINCT A.country_code, B.short_name, A.series_code, A.description, C.value, C.year

                FROM `bigquery-public-data.world_bank_intl_education.country_series_definitions` A

                JOIN `bigquery-public-data.world_bank_intl_education.country_summary` B ON A.country_code = B.country_code

                JOIN `bigquery-public-data.world_bank_intl_education.international_education` C ON A.country_code = C.country_code

                WHERE C.year < 2019

                ORDER by C.year DESC

                LIMIT 10

                """



query_job = client.query(query)

query_job.to_dataframe()
query = """SELECT min(C.year)

                FROM `bigquery-public-data.world_bank_intl_education.country_series_definitions` A

                JOIN `bigquery-public-data.world_bank_intl_education.country_summary` B ON A.country_code = B.country_code

                JOIN `bigquery-public-data.world_bank_intl_education.international_education` C ON A.country_code = C.country_code

                """



query_job = client.query(query)

query_job.to_dataframe()
query = """SELECT DISTINCT A.country_code, B.short_name, sum(C.value) as Education_sum

                FROM `bigquery-public-data.world_bank_intl_education.country_series_definitions` A

                JOIN `bigquery-public-data.world_bank_intl_education.country_summary` B ON A.country_code = B.country_code

                JOIN `bigquery-public-data.world_bank_intl_education.international_education` C ON A.country_code = C.country_code

                WHERE C.year < 2019

                GROUP BY A.country_code, B.short_name

                ORDER BY Education_sum DESC

                LIMIT 10

                """



query_job = client.query(query)

query_job.to_dataframe()
query = """SELECT DISTINCT A.country_code, B.short_name, sum(C.value) as Education_sum

                FROM `bigquery-public-data.world_bank_intl_education.country_series_definitions` A

                JOIN `bigquery-public-data.world_bank_intl_education.country_summary` B ON A.country_code = B.country_code

                JOIN `bigquery-public-data.world_bank_intl_education.international_education` C ON A.country_code = C.country_code

                WHERE C.year < 2019

                GROUP BY A.country_code, B.short_name

                ORDER BY Education_sum

                LIMIT 1

                """



query_job = client.query(query)

query_job.to_dataframe()