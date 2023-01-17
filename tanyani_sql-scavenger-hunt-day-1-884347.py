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
# the query_to_pandas_safe will only return a result if it's less
# than one gigabyte (by default)
us_cities = open_aq.query_to_pandas_safe(query)
# What five cities have the most measurements taken there?
us_cities.city.value_counts().head()
# Countries that use a unit other than ppm to measure any type of pollution

query="""
    select country, count(1) 
    from `bigquery-public-data.openaq.global_air_quality`
    where unit != 'ppm'
    group by country
"""
open_aq.estimate_query_size(query) # 0.0015630675479769707
countries_no_ppmUnit = open_aq.query_to_pandas_safe(query, max_gb_scanned=0.1)

import matplotlib as mpl
import matplotlib.pyplot as plt
%matplotlib inline
from wordcloud import WordCloud

wordcloud = WordCloud(
                          background_color='white',
                          max_font_size=20, 
                          collocations=False
                         ).generate(' '.join(countries_no_ppmUnit['country']))

print(wordcloud)
fig = plt.figure(1)
plt.imshow(wordcloud)
plt.axis('off')
plt.show()
# Pollutants that have a value of exactly zero

query="""
    select pollutant 
    from `bigquery-public-data.openaq.global_air_quality`
    where value = 0.00
    
"""
open_aq.estimate_query_size(query) # 0.0015630675479769707
pollutant_zeroValue=open_aq.query_to_pandas_safe(query, max_gb_scanned=0.1)

import seaborn as sns
sns.countplot(x="pollutant", data=pollutant_zeroValue, palette="Greens_d");
