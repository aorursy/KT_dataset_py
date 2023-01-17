import bq_helper
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

# from subprocess import check_output
# print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

AQ=bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                dataset_name="openaq")
AQ.list_tables()
AQ.table_schema("global_air_quality")
AQ.head("global_air_quality",num_rows=10)

query="""SELECT city
FROM `bigquery-public-data.openaq.global_air_quality`
WHERE country = "US"
"""

# AQ.estimate_query_size(query)
# US_cities=AQ.query_to_pandas_safe(query)
# US_cities.city.value_counts().head()

queryUnitNotPPM="""SELECT country
FROM `bigquery-public-data.openaq.global_air_quality`
WHERE unit != "ppm"
"""

notPPM=AQ.query_to_pandas_safe(queryUnitNotPPM)
countriesWithoutPPM=pd.unique(notPPM.country)

queryPollutantZero="""SELECT pollutant
FROM `bigquery-public-data.openaq.global_air_quality`
WHERE value=0
"""
zeroPollutant=AQ.query_to_pandas_safe(queryPollutantZero)
pollutantsWithZeros=pd.unique(zeroPollutant.pollutant)