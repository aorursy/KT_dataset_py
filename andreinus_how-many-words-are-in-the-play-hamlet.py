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
tables = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="samples")
display = BigQueryHelper("bigquery-public-data", "samples")
display.list_tables()
display.head("shakespeare", num_rows=20)
# Here I want to figure out what is the name of each comp
query1 = """SELECT
  DISTINCT corpus  
FROM
  `bigquery-public-data.samples.shakespeare`
   

        """
response1 = tables.query_to_pandas_safe(query1, max_gb_scanned=10)
print(response1)

#In my second Query, I simply count the words of "Hamlet". It seems that there are 5318 words 

query2 = """SELECT
  count(corpus)  
FROM
  `bigquery-public-data.samples.shakespeare`
   WHERE corpus = "hamlet"

        """
response2 = tables.query_to_pandas_safe(query2, max_gb_scanned=10)
print(response2)