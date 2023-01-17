# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-pythonfrom bq_helper import BigQueryHelper
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
from bq_helper import BigQueryHelper
QUERY = """
    SELECT
        extract(DAYOFYEAR from date_local) as day_of_year,
        aqi
    FROM
      `bigquery-public-data.epa_historical_air_quality.pm25_frm_daily_summary`
    WHERE
      city_name = "Los Angeles"
      AND state_name = "California"
      AND sample_duration = "24 HOUR"
      AND poc = 1
      AND EXTRACT(YEAR FROM date_local) = 2015
    ORDER BY day_of_year
        """
bq_assistant = BigQueryHelper("bigquery-public-data", "epa_historical_air_quality")
df = bq_assistant.query_to_pandas(QUERY)
df.plot(x='day_of_year', y='aqi', style='.');
