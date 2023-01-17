import bq_helper
from bq_helper import BigQueryHelper
# https://www.kaggle.com/sohier/introduction-to-the-bq-helper-package
chicago_crime = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="chicago_crime")
# EXAMPLE = """SELECT * FROM `bigquery-public-data.chicago_crime.crime` LIMIT 1"""
# EXAMPLE_RESP = chicago_crime.query_to_pandas_safe(EXAMPLE)
# EXAMPLE_RESP
import pandas as pd
neighborhoods = pd.read_csv('../input/community-areas.csv')
neighborhoods.head()
getNeighborHoods = """
    SELECT arrest, domestic, community_area, COUNT(*) as count
    FROM `bigquery-public-data.chicago_crime.crime`
    GROUP BY arrest, domestic, community_area"""
responseNeighborhoods = chicago_crime.query_to_pandas_safe(getNeighborHoods)
responseNeighborhoods.head()
merged = responseNeighborhoods.merge(neighborhoods, left_on="community_area", right_on="id", how="inner")
merged.groupby(['zone', 'domestic', 'arrest']).sum()['count']