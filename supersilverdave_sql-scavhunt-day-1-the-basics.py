#import BigQueryHelper API
import bq_helper

#create object for open_aq queries
open_aq = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="openaq")

#print tables
open_aq.list_tables()
#Check the first lines
open_aq.head("global_air_quality")
#select all items from "city" where the country is "us"
query = """SELECT city
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE country = 'US'
        """
#estimate size of the query
open_aq.estimate_query_size(query)
#the query is small, but we'll run the bounded (max 1gb) query
us_cities = open_aq.query_to_pandas_safe(query)
#List the five cities with most measurements
us_cities.city.value_counts().head()
query_1 = '''SELECT DISTINCT country
                FROM `bigquery-public-data.openaq.global_air_quality`
                WHERE unit!= 'ppm';
            '''
open_aq.estimate_query_size(query_1)
answer_1 = open_aq.query_to_pandas_safe(query_1)
print('There are %d countries that measure a type of pollution with a unit other than ppm.' % answer_1.shape[0])
answer_1.to_csv('nonppm_countries.csv')
answer_1
query_2 = '''SELECT pollutant
                FROM `bigquery-public-data.openaq.global_air_quality`
                WHERE value=0;
            '''
open_aq.estimate_query_size(query_2)
answer_2 = open_aq.query_to_pandas_safe(query_2)
print('There are %d measurements of pollutants that have a value of 0.' % answer_2.shape[0])
print('The first are:')
answer_2.head()
query_2_detailed = '''SELECT pollutant, location, timestamp, source_name
                FROM `bigquery-public-data.openaq.global_air_quality`
                WHERE value=0;
            '''
open_aq.estimate_query_size(query_2_detailed)
answer_2_detailed = open_aq.query_to_pandas_safe(query_2_detailed)
answer_2_detailed.to_csv('zero_pollutant_measurements.csv')
answer_2_detailed
