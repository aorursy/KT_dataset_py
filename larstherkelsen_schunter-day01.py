import pandas as pd
#Dataanalysis package that turns ie sql results intp dataframes
from google.cloud import bigquery
#g.c gives access to google cloud services
#bigquery makes analytics at scale possible
from bq_helper import BigQueryHelper
#simplifies the common read-only tasks we can do 
#Connect to the database structure: (active_project="" , dataset_name ="")
oaq = BigQueryHelper('bigquery-public-data', 'openaq')
#Ask for tablename overview in database
oaq.list_tables()
#Get information for specific table
oaq.table_schema("global_air_quality")
#Preview first rows of the choosen table
oaq.head("global_air_quality")
#Countries using other units than 'ppm' (parts pr million)
#Please notice the specific `.` around the table while '.' or "." are used for values
#Also """.""" makes it possible to write the query over several lines
sql1="""SELECT DISTINCT country
    FROM `bigquery-public-data.openaq.global_air_quality`
    WHERE unit!='ppm' ORDER BY country ASC
    """
#Estimating query size, result given in GB
oaq.estimate_query_size(sql1)
#Get the result into a panda dataframe make sure result is less than 1GB (_safe)
countries = oaq.query_to_pandas_safe(sql1)
#Get the dimensions of the dateframe
countries.shape
#Return a visual of the first five rows
countries.head(5)
#Pollutants with values 0
sql2 = """SELECT DISTINCT pollutant
    FROM `bigquery-public-data.openaq.global_air_quality`
    WHERE value=0 ORDER BY pollutant ASC
    """
oaq.estimate_query_size(sql2)
pollutant = oaq.query_to_pandas_safe(sql2)
pollutant.shape
pollutant