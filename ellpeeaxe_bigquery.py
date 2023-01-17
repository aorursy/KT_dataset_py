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

from bq_helper import BigQueryHelper

bq_assistant = BigQueryHelper("bigquery-public-data", "chicago_taxi_trips")
# INSERT QUERIES HERE

query = """

    SELECT

    DISTINCT company

    FROM `bigquery-public-data.chicago_taxi_trips.taxi_trips`

    WHERE trip_start_timestamp BETWEEN timestamp('2019-01-01') AND timestamp('2019-12-31')

"""



df = bq_assistant.query_to_pandas(query)
# PRINTS A SAMPLE OF THE DATA

df.sample(5)
# RUN THIS WHEN U WANT TO SAVE

# PLS CHANGE THE FILENAME ACCORDINGLY

df.to_csv('companies.csv')