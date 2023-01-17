# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from google.cloud import bigquery
from bq_helper import BigQueryHelper
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



# Any results you write to the current directory are saved as output.
bq_assistant = BigQueryHelper('bigquery-public-data', 'openaq')

QUERY = """
        SELECT distinct country from `bigquery-public-data.openaq.global_air_quality` where unit != 'ppm'
       
        """
df = bq_assistant.query_to_pandas(QUERY)
df
QUERY = """
        SELECT distinct pollutant from `bigquery-public-data.openaq.global_air_quality` where value = 0
       
        """
df_2 = bq_assistant.query_to_pandas(QUERY)
df_2
