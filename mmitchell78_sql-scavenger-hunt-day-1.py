"Which countries use a unit other than ppm to measure any type of pollution?"
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import bq_helper # import our bq_helper package

AQData = bq_helper.BigQueryHelper(active_project= "bigquery-public-data", 
                                       dataset_name = "openaq")

CountryQuery =  """
                SELECT distinct country,
                                unit 
                FROM `bigquery-public-data.openaq.global_air_quality`
                WHERE unit != "ppm"
                """


AQData.query_to_pandas_safe(CountryQuery, max_gb_scanned=0.1)



AllPollutantQuery =    """
                    SELECT *
                    FROM `bigquery-public-data.openaq.global_air_quality`
                    WHERE value = 0
                    """
AQData.query_to_pandas_safe(AllPollutantQuery, max_gb_scanned=0.1)
DistinctPollutantQuery =    """
                            SELECT Distinct Pollutant
                            FROM `bigquery-public-data.openaq.global_air_quality`
                            WHERE value = 0
                         """
AQData.query_to_pandas_safe(DistinctPollutantQuery, max_gb_scanned=0.1)