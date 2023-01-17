# import package with helper functions 
import bq_helper

# create a helper object for this dataset
open_aq = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="openaq")

# print all the tables in this dataset (there's only one!)
open_aq.list_tables()

open_aq.head("global_air_quality")
query1 = """SELECT country
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE unit != 'ppm'
        """
other_than_ppm = open_aq.query_to_pandas_safe(query1)
other_than_ppm.head()
import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize = (20, 6))
sns.countplot(other_than_ppm['country'])
other_than_ppm['country'].unique()
query2 = """SELECT pollutant
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE value = 0
        """
zero_value = open_aq.query_to_pandas_safe(query2)
zero_value.head()
plt.figure(figsize = (20, 6))
sns.countplot(zero_value['pollutant'])
zero_value['pollutant'].unique()
other_than_ppm.to_csv("otp.csv")
zero_value.to_csv("zero.csv")
query3 = """SELECT *
            FROM `bigquery-public-data.openaq.global_air_quality`
        """
df = open_aq.query_to_pandas_safe(query3)
df.corr()
sns.heatmap(df.corr(), yticklabels=False)
sns.jointplot(df['longitude'], df['averaged_over_in_hours'], kind='reg')
