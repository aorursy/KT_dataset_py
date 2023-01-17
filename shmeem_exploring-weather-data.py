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
air_quality = bq_helper.BigQueryHelper(active_project='bigquery-public-data',
                                       dataset_name='epa_historical_air_quality')
air_quality.list_tables()
temp_daily = {sf.name for sf in air_quality.table_schema('temperature_daily_summary')}
annual_summary = {sf.name for sf in air_quality.table_schema('air_quality_annual_summary')}
temp_daily.intersection(annual_summary)
# Example query estimate

query = """SELECT arithmetic_mean
            FROM `bigquery-public-data.epa_historical_air_quality.temperature_daily_summary`
            WHERE state_name = "Oregon"
"""

air_quality.estimate_query_size(query)
oregon_temps = air_quality.query_to_pandas_safe(query, max_gb_scanned=0.1)
oregon_temps.head()
oregon_temps.describe()
oregon_temps.arithmetic_mean.mean()
oregon_temps.to_csv('oregon_temps_fahrenheit.csv')