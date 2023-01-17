# import package with helper functions 
import bq_helper

# create a helper object for this dataset
open_aq = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="openaq")

# print all the tables in this dataset (there's only one!)
open_aq.list_tables()
# print the first couple rows of the "global_air_quality" dataset
open_aq.head("global_air_quality")
# query to select all the items from the "city" column where the
# "country" column is "us"
query = """SELECT city
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE country = 'US'
        """
def converted_query_size(helper,query=''):
    all_sizes=['GB','MB','KB','B']
    if type(helper) == bq_helper.BigQueryHelper:
        size =helper.estimate_query_size(query)
    i=0
    factor=1024
    while size<1:
        size*=factor
        i+=1
    return f'{size:.1f} {all_sizes[i]}'
# the query_to_pandas_safe will only return a result if it's less
# than one gigabyte (by default)
print(converted_query_size(open_aq,query))
us_cities = open_aq.query_to_pandas_safe(query)
# What five cities have the most measurements taken there?
us_cities.city.value_counts().head()
# Your code goes here :)
for t in open_aq.list_tables():
    print(f'Table: {t}\n{"-"*100}')
    for f in open_aq.table_schema(t):
        print(f'{f.name}: {f.description} [{f.field_type}]')        

query = """SELECT country,unit
            FROM `bigquery-public-data.openaq.global_air_quality`
        """
print(converted_query_size(open_aq,query))
country_unit = open_aq.query_to_pandas_safe(query)
# Which countries use a unit other than ppm to measure any type of pollution?
total1 = country_unit[country_unit.unit!='ppm'].drop_duplicates().shape[0]
print(f'{total1} countries dont use ppm as measure unit.')
#country_unit.unit.unique()
 #Which pollutants have a value of exactly 0?
query = """SELECT city,country,pollutant,value
            FROM `bigquery-public-data.openaq.global_air_quality`
            where value = 0
        """
print(converted_query_size(open_aq,query))
no_pollutant = open_aq.query_to_pandas_safe(query)
print(no_pollutant.shape[0])
no_pollutant.head()