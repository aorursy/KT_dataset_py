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

import matplotlib.pyplot as plt

ny_data_set = bq_helper.BigQueryHelper(active_project="bigquery-public-data", dataset_name="new_york")
# QUESTION: What are the most common complaint types made by each zip from 2011 to 2018



answer_query = """

SELECT incident_zip, complaint_type, COUNT(1) AS num_times

FROM `bigquery-public-data.new_york.311_service_requests`

GROUP BY incident_zip, complaint_type

ORDER BY complaint_type DESC



"""

#ONE_GB = 10000*1000*1000

#safe_config = bigquery.QueryJobConfig(maximum_bytes_billied = ONE_GB)

result = ny_data_set.query_to_pandas_safe(answer_query)

result