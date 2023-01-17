# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
from google.cloud import bigquery
from bq_helper import BigQueryHelper

# Any results you write to the current directory are saved as output.
from google.cloud import bigquery
import pandas as pd

client = bigquery.Client()

# Using WHERE reduces the amount of data scanned / quota used
query = """
SELECT *
FROM `bigquery-public-data.chicago_taxi_trips.taxi_trips`
WHERE trip_total BETWEEN 1 AND 100
"""

query_job = client.query(query)
iterator = query_job.result(timeout=30)
rows = list(iterator)

# Transform the rows into a nice pandas dataframe
headlines = pd.DataFrame(data=[list(x.values()) for x in rows], columns=list(rows[0].keys()))

# Look at the first 10 headlines
headlines.head(10)
headlines.shape[0]

headlines.to_csv('csv_to_submitv2.csv', index = False)

