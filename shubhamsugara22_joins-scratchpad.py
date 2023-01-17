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



# Create a "Client" object

client = bigquery.Client()



# Construct a reference to the "github_repos" dataset

dataset_ref = client.dataset("github_repos", project="bigquery-public-data")



# API request - fetch the dataset

dataset = client.get_dataset(dataset_ref)



# Construct a reference to the "licenses" table

licenses_ref = dataset_ref.table("licenses")



# API request - fetch the table

licenses_table = client.get_table(licenses_ref)



# Preview the first five lines of the "licenses" table

client.list_rows(licenses_table, max_results=5).to_dataframe()
# Construct a reference to the "sample_files" table

files_ref = dataset_ref.table("sample_files")



# API request - fetch the table

files_table = client.get_table(files_ref)



# Preview the first five lines of the "sample_files" table

client.list_rows(files_table, max_results=5).to_dataframe()
# Query to determine the number of files per license, sorted by number of files

query = """

        SELECT L.license, COUNT(1) AS number_of_files

        FROM `bigquery-public-data.github_repos.sample_files` AS sf

        INNER JOIN `bigquery-public-data.github_repos.licenses` AS L 

            ON sf.repo_name = L.repo_name

        GROUP BY L.license

        ORDER BY number_of_files DESC

        """



# Set up the query (cancel the query if it would use too much of 

# your quota, with the limit set to 10 GB)

safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=10**10)

query_job = client.query(query, job_config=safe_config)



# API request - run the query, and convert the results to a pandas DataFrame

file_count_by_license = query_job.to_dataframe()
# Print the DataFrame

file_count_by_license