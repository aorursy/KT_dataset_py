from google.cloud import bigquery

from bq_helper import BigQueryHelper

import matplotlib.pyplot as plt

import pandas as pd

import matplotlib.pyplot as plt

# Create a "Client" object

client = bigquery.Client()



# Construct a reference to the "world_bank_intl_education" dataset

dset_ref = client.dataset("world_bank_intl_education", project="bigquery-public-data")



# API request - fetch the dataset

dataset = client.get_dataset(dset_ref)



# Construct a reference to the "international_education" table

table_ref = dset_ref.table("international_education")



# API request - fetch the table

table = client.get_table(table_ref)
# Use client.list_tables to get information about the tables within the dataset.

[x.table_id for x in client.list_tables(dset_ref)]
# Preview the first five lines of the "international_education" table

client.list_rows(table, max_results=5).to_dataframe()
# Explore indicators 

sql0 = """

    SELECT DISTINCT(indicator_name),indicator_code

    FROM `bigquery-public-data.world_bank_intl_education.international_education`

"""



df_indicators = client.query(sql0).to_dataframe()

num_indicators = len(df_indicators.index)

print ('There are %d indicators'%(num_indicators))

df_indicators.head()

# Explore countries 

sql1 = """

    SELECT DISTINCT(country_name),country_code

    FROM `bigquery-public-data.world_bank_intl_education.international_education`

"""



df_country = client.query(sql1).to_dataframe()

num_country = len(df_country.index)

print ('There are %d countries'%(num_country))

df_country.head()
sql2 = """

    SELECT DISTINCT(year)

    FROM `bigquery-public-data.world_bank_intl_education.international_education`

    ORDER BY year

"""

df_year = client.query(sql2).to_dataframe()

num_year = len(df_year.index)

print ('There are %d years'%(num_year))

df_year.head()
df_year.tail()
# Drop data for years > 2017

sql3 = """

    SELECT * 

    FROM `bigquery-public-data.world_bank_intl_education.international_education`

    WHERE year < 2017 AND indicator_code = 'NY.GDP.PCAP.CD' OR indicator_code = 'UIS.SAP.CE' 

"""

df = client.query(sql3).to_dataframe()

num_count = len(df.index)

print ('There are %d lines of data'%(num_count))

df.head()
df_2016 = df[df['year'] == 2016]

df_2016.head()
df_gdp = df_2016[df['indicator_code'] == 'NY.GDP.PCAP.CD']

df_ed = df_2016[df['indicator_code'] == 'UIS.SAP.CE']

df_ed.head()
df_new = pd.merge(df_gdp, df_ed, on='country_code')

df_new.head()
df_new = df_new[['country_name_x','value_x','value_y']]
plt.scatter(df_new['value_x'], df_new['value_y'], alpha=0.2,cmap='viridis')

plt.xlabel('GDP')

plt.ylabel('Population of compulsory school age, both sexes.');