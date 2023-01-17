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
import bq_helper
from bq_helper import BigQueryHelper
# https://www.kaggle.com/sohier/introduction-to-the-bq-helper-package
sampleTables = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="samples")
bq_assistant = BigQueryHelper("bigquery-public-data", "samples")
bq_assistant.list_tables()
#I am trying to find the people that are born on Christmas
bq_assistant.head("natality", num_rows=20)
query1 = """SELECT
  COUNT(year)
FROM
  `bigquery-public-data.samples.natality`
WHERE
  month = 12 AND day = 25 AND state = "NY";
        """
#The result of this simple query will be the people that are born on Christmas
response1 = sampleTables.query_to_pandas_safe(query1, max_gb_scanned=10)
print(response1)

#We found out that the people that are born in New York between 1969 and 2008 are 9798 
query2 = """SELECT
  COUNT(year)
FROM
  `bigquery-public-data.samples.natality`
WHERE
  month = 12 AND state = "NY" AND day IS NULL;
   """
response2 = sampleTables.query_to_pandas_safe(query2, max_gb_scanned=10)
print(response2)

#consider that in this dataset there are 352060 people that are born in December but for which the birthdate is actually not available!
query3 = """SELECT
  COUNT(year)
FROM
  `bigquery-public-data.samples.natality`
WHERE
  state = "NY";
   """
response3 = sampleTables.query_to_pandas_safe(query3, max_gb_scanned=10)
print(response3)

#All the people that were born in New York States between 1969 and 2008, are: 8670137
 