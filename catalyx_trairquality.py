# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import bq_helper 
def list_objs(obj):
    # list data objs:
    return [command for command in dir(obj) if not command.startswith('_')]
# create a helper object for this dataset
kaggleID = 'bigquery-public-data'
datasetID = 'openaq'

airq_data = bq_helper.BigQueryHelper(active_project = kaggleID, dataset_name = datasetID)
# print all the tables in this dataset (there's only one!)
airq_data.list_tables()
tbl_air = airq_data.list_tables()[0]
strFROM = kaggleID+'.'+datasetID+'.'+tbl_air
strFROM  # used to obtain the full name to be used in the query FROM clause, or
         # as a qry parameter when implemented
airq_data.table_schema(tbl_air)  # what's in there?
qry_ppm = ( 'SELECT country, unit '
          'FROM `bigquery-public-data.openaq.global_air_quality` '
          'WHERE unit = "ppm" ')
airq_data.estimate_query_size(qry_ppm)
ppm_df = airq_data.query_to_pandas(qry_ppm)
ppm_df.drop_duplicates(inplace=True)
ppm_df.shape
# check unique names: if clean data, should be the same size as df rows:
np.alltrue(len(set(ppm_df.country)) == ppm_df.shape[0])
qry_not_ppm = ( 'SELECT country, unit '
          'FROM `bigquery-public-data.openaq.global_air_quality` '
          'WHERE unit != "ppm" ')
airq_data.estimate_query_size(qry_not_ppm)
not_ppm_df = airq_data.query_to_pandas(qry_not_ppm)
not_ppm_df.drop_duplicates(inplace=True)
not_ppm_df.shape
# check
np.alltrue(len(set(not_ppm_df.country)) == not_ppm_df.shape[0])
print('D1Q1 Answer\nList of countries where the unit is not "ppm":')
not_ppm_df
qry = ( 'SELECT pollutant, value '
          'FROM `bigquery-public-data.openaq.global_air_quality` '
          'WHERE value = 0 ')
airq_data.estimate_query_size(qry)
pol0_df = airq_data.query_to_pandas(qry)
pol0_df.drop_duplicates(inplace=True)
pol0_df.shape
pol0_set = set(pol0_df.pollutant)
np.alltrue(len(pol0_set) == pol0_df.shape[0]) 
print('D1Q2 Answer\nList of pollutants with 0 value:')
pol0_df
