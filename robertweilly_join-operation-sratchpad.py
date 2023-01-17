# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
from google.cloud import bigquery



client = bigquery.Client()



dataset_ref = client.dataset("github_repos", project="bigquery-public-data")



dataset  = client.get_dataset(dataset_ref)



tables = list(client.list_tables(dataset))

for table in tables:

        print(table.table_id)



licence_ref = dataset_ref.table("licenses")



licence_table = client.get_table(licence_ref)



client.list_rows(licence_table,max_results = 5).to_dataframe()
files_ref = dataset_ref.table("sample_files")



file_table = client.get_table(files_ref)



client.list_rows(file_table,max_results=5).to_dataframe()
query ="""

          SELECT L.license,count(1) as number_of_files

          FROM `bigquery-public-data.github_repos.sample_files` AS sf

          INNER JOIN `bigquery-public-data.github_repos.licenses` AS L 

            ON sf.repo_name = L.repo_name

          GROUP BY L.license

          ORDER BY number_of_files DESC

          """



safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=10**10)

query_job = client.query(query, job_config=safe_config)

file_count_by_license = query_job.to_dataframe()

file_count_by_license