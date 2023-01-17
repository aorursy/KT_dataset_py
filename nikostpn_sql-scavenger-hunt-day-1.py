#Setting up the dataset/helper
import bq_helper

open_aq = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="openaq")
# Your code goes here :)

#First challange
query_countries_no_ppm = """SELECT country
                            FROM `bigquery-public-data.openaq.global_air_quality`
                            WHERE unit != "ppm" """
#Run the query with a size of less than 1 gig
countries_no_ppm = open_aq.query_to_pandas_safe(query_countries_no_ppm)
#Delete Dpulicate Countries keeping only non-duplicates or the first instance of duplicates
countries_no_ppm = countries_no_ppm.drop_duplicates(keep='first')
#Output the data fram & write to .csv
countries_no_ppm.to_csv("countries_no_ppm.csv")
countries_no_ppm
#Second challange
query_pollutants_null = """SELECT pollutant
                            FROM `bigquery-public-data.openaq.global_air_quality`
                            WHERE value = 0 """
#
pollutants_null = open_aq.query_to_pandas_safe(query_pollutants_null)
#Create an array of all pollutants with a value of 0 at least in 1 location
pollutants_uniq = pollutants_null['pollutant'].unique()
#Nicely output all of the pollutants that have a value of 0 at least in 1 location
for val in pollutants_uniq:
    print(val)
