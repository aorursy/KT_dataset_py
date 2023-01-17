# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
import bq_helper
glob_a = bq_helper.BigQueryHelper(active_project= "bigquery-public-data", 
                                       dataset_name = "openaq")
glob_a.list_tables()
#glob_a.table_schema()
#glob_a.head()

glob_a.table_schema("global_air_quality")
glob_a.head("global_air_quality")
query = """ SELECT city 
             FROM `bigquery-public-data.openaq.global_air_quality`  
             where country = 'US' """
us_city= glob_a.query_to_pandas_safe(query)
us_city.head()
query2= """SELECT country
             FROM`bigquery-public-data.openaq.global_air_quality`
             WHERE unit!= 'ppm' """
no_ppm= glob_a.query_to_pandas_safe(query2)
no_ppm.head()