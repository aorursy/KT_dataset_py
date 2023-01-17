

import numpy as np 

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



from google.cloud import bigquery

import bq_helper

usa_names = bq_helper.BigQueryHelper(active_project="bigquery-public-data", dataset_name="usa_names")



client = bigquery.Client()
query = """SELECT year, gender, name, sum(number) as number FROM `bigquery-public-data.usa_names.usa_1910_current` GROUP BY year, gender, name"""

names = usa_names.query_to_pandas_safe(query)

names.to_csv("usa_names.csv")

names.sample(10)



dataset_ref = client.dataset("usa_names", project="bigquery-public-data")



dataset = client.get_dataset(dataset_ref)
tables = list(client.list_tables(dataset))



for table in tables:  

    print(table.table_id)

client.list_rows(table, max_results=5).to_dataframe()
##Now for a more specific query to get the exact info I'm looking for



query = """SELECT year, gender, name, sum(number) as number FROM `bigquery-public-data.usa_names.usa_1910_current`  WHERE name = 'Hunter'  GROUP BY year, gender, name"""

hunter = usa_names.query_to_pandas_safe(query)

hunter.sample(10)
##Separating the data set into a female and male set

hunter_f = hunter[hunter.gender.isin(['F'])]

hunter_m = hunter[hunter.gender.isin(['M'])]



#print(hunter_m.head())

#print(hunter.isnull)
plt.figure(figsize=(24,12))

plt.title('Frequency of the name Hunter over the years')





sns.lineplot(y=hunter_m['number'], x=hunter_m['year'], label="Male")

sns.lineplot(y=hunter_f['number'], x=hunter_f['year'], label='Female')





#plt.xlabel("Date")

#plt.ylabel("Value($)")

#sns.set_style("white")
safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=10**10)



query = """

        SELECT year, gender, name, sum(number) as number

        FROM `bigquery-public-data.usa_names.usa_1910_current`

        WHERE year >= 1998 and year <=2019

        GROUP BY year, gender, name

        ORDER BY year



        """



names_volume = client.query(query, job_config=safe_config)

names_volume = names_volume.to_dataframe()

names_volume.sample(10)

##While the .head() function is typically used to check if a data set looks correctly configured i tend to use sample so that the data i'm looking at is always called different
mean_data =  names_volume.groupby('year')['number'].mean()

display(mean_data)

names_volume.describe()
plt.figure(figsize=(36,12))



sns.kdeplot(data=names_volume["number"], shade=True)
plt.figure(figsize=(36,12))



sns.violinplot(x=names_volume["number"], color='green')