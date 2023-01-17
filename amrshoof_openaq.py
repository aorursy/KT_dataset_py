# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import bq_helper

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
air_quality = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                      dataset_name="openaq")
query = """SELECT DISTINCT country
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE unit != "ppm" """

countries_not_using_ppm = air_quality.query_to_pandas_safe(query, max_gb_scanned=1)
countries_not_using_ppm.to_csv("countries_not_using_ppm.csv")
# Any results you write to the current directory are saved as output.