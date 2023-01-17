# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

#from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))


# Any results you write to the current directory are saved as output.
import bq_helper
openAQ = bq_helper.BigQueryHelper(active_project= "bigquery-public-data", 
                                       dataset_name = "openaq")
openAQ.list_tables()
openAQ.table_schema("global_air_quality")
#Which countries use a unit other than ppm to measure any type of pollution? 

openAQ.head("global_air_quality")
query = """SELECT distinct country, unit FROM `bigquery-public-data.openaq.global_air_quality`
        WHERE unit != "ppm" 
        """ 
openAQ.estimate_query_size(query)
openAQ.query_to_pandas_safe(query, max_gb_scanned=0.1)
#Which pollutants have a value of exactly 0?
query = """SELECT distinct pollutant FROM `bigquery-public-data.openaq.global_air_quality`
        WHERE value = 0 
        """ 
openAQ.estimate_query_size(query)
openAQ.query_to_pandas_safe(query, max_gb_scanned=0.1)
