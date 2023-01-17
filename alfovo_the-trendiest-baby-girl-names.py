import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import bq_helper



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.



# Read the datset from BigQuery file

from bq_helper import BigQueryHelper

dataset = bq_helper.BigQueryHelper(active_project="bigquery-public-data",

                                   dataset_name="usa_names")



# Create a pandas dataframe from the usa_1910_current dataset

query = """

SELECT 

year,

name,

sum(number) as count

FROM `bigquery-public-data.usa_names.usa_1910_current`

WHERE gender="F"

GROUP BY year, gender, name

"""

girl_names = dataset.query_to_pandas_safe(query)
# Great, now let's take a peak at what this sucker looks like

girl_names.head()
# girl_names = names_df[names_df['gender'] == 'F'].drop(columns=['gender'])

names_by_count = girl_names.groupby('name').sum().sort_values(by='count', ascending=False)

names_by_count.loc[:, ['count']].head(10).plot(kind='bar', y="count", figsize=(10,5), title="Overall Most Popular Baby Girl Names 1910-2018")
max_count_year_dict = girl_names.groupby(['year']).agg({'count':np.max}).to_dict()['count']

def max_count(count, year):

    return count == max_count_year_dict[year]

most_popular_by_year = girl_names[girl_names.apply(lambda x : max_count(x['count'],x['year']),axis=1)]
num_years_popular = most_popular_by_year.name.value_counts().rename_axis('name').reset_index(name='number')

num_years_popular.plot(kind='barh', x='name', y='number', figsize=(15,5), title="Number of Years a Name is the Most Popular")
most_popular_by_year.boxplot(by ='name', column =['count'], grid = False, figsize=(15,5))
most_popular_by_year.plot(kind='line', x="year", y="count", figsize=(15,5), grid=True, title="Number of Babies with Most Popular Baby Girl Name by Year")
interesting_names = ['Linda', 'Lisa','Jessica', 'Ashley']

possibly_trendy_girl_names = girl_names[girl_names['name'].isin(interesting_names)]



df = possibly_trendy_girl_names.pivot(index='year', columns='name', values='count')

df.plot(figsize=(15,5), grid=True, title="Popularity of the Trendiest Names by Year")
most_popular_by_year[most_popular_by_year['name'] == 'Linda']
other_names = ['Mary', 'Jennifer', 'Emily', 'Emma', 'Sophia', 'Isabella', 'Linda', 'Lisa','Jessica', 'Ashley', 'Barbara', 'Margaret', 'Susan', 'Dorothy']

other_girl_names = girl_names[girl_names['name'].isin(other_names)]



df = other_girl_names.pivot(index='year', columns='name', values='count')

df.plot(figsize=(15,5), grid=True, title="Trendy and Almost Trendy Names by Year")
jillian_df = girl_names[girl_names['name'].isin(['Jill', 'Jillian'])].pivot(index='year', columns='name', values='count')

jillian_df.plot(figsize=(15,5), grid=True, title="The Popularity of the name Jillan and Jill ")