# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory


# Any results you write to the current directory are saved as output.

import bq_helper

openaq = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="openaq")

openaq.list_tables()
openaq.head('global_air_quality')
#Which countries use a unit other than ppm to measure any type of pollution? 
query1 = """ SELECT DISTINCT country,unit 
              FROM `bigquery-public-data.openaq.global_air_quality`
             WHERE unit != 'ppm' order by country """



openaq.query_to_pandas_safe(query1)

#Which countries use a unit other than ppm to measure any type of pollution and their units?
query2 = """ SELECT DISTINCT country,unit 
                        FROM `bigquery-public-data.openaq.global_air_quality`
                       WHERE  country in (SELECT country 
                                            FROM `bigquery-public-data.openaq.global_air_quality`
                                           WHERE unit != 'ppm') order by country """
openaq.query_to_pandas_safe(query2)
#Which pollutants have a value of exactly 0?

query3 = """ SELECT pollutant, SUM(value)
               FROM `bigquery-public-data.openaq.global_air_quality` 
           GROUP BY pollutant,value
             HAVING SUM(value)=0
               """

openaq.query_to_pandas_safe(query3)
