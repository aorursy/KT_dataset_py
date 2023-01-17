# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import bq_helper
#create a helper object for bigquery dataset
hacker_news = bq_helper.BigQueryHelper(active_project= "bigquery-public-data", dataset_name = "hacker_news")
query = """SELECT type, COUNT (id) 
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type
        """
hacker_news.query_to_pandas_safe(query)
#import packages
import numpy as np
import pandas as pd
import bq_helper
query2 = """SELECT COUNT (deleted)
            FROM `bigquery-public-data.hacker_news.comments`
            HAVING True
        """
hacker_news.query_to_pandas_safe(query2)
