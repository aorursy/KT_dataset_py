# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import bq_helper
from bq_helper import BigQueryHelper
# https://www.kaggle.com/sohier/introduction-to-the-bq-helper-package
chicago_crime = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="chicago_crime")

bq_assistant = BigQueryHelper("bigquery-public-data", "chicago_crime")
bq_assistant.list_tables()
bq_assistant.table_schema("crime")
bq_assistant.table_schema("crime")['description'][12]
query1 = """SELECT 
                    date,
                    EXTRACT(HOUR from date) as hour,
                    block,
                    primary_type,
                    description,
                    location_description,
                    arrest,
                    beat,
                    district,
                    latitude,
                    longitude
            FROM
            `bigquery-public-data.chicago_crime.crime`
            where 
                Date_diff(EXTRACT(Date from date), date_add(CURRENT_DATE(),INTERVAL -12 MONTH), DAY)<365 and 
                arrest=False
                and CHAR_LENGTH(location)>0
             Limit 20
        """
response1 = chicago_crime.query_to_pandas_safe(query1)
response1.head(10)

#                    year,
#                    x_coordinate,
#                    y_coordinate,
query1 = """SELECT 
                    count(unique_key) as crimes,
                    EXTRACT(HOUR from date) as hour
            FROM
            `bigquery-public-data.chicago_crime.crime`
            where 
                Date_diff(EXTRACT(Date from date), date_add(CURRENT_DATE(),INTERVAL -12 MONTH), DAY)<365 and 
                arrest=False
                and CHAR_LENGTH(location)>0
            Group By EXTRACT(HOUR from date)
        """
response1 = chicago_crime.query_to_pandas_safe(query1)
response1 = response1.sort_values(by=['hour'])
response1.plot.bar(x='hour', y='crimes', rot=0)
query1 = """SELECT 
                    COUNT(distinct block) as count_block,
                    COUNT(distinct beat) as count_beat,
                    COUNT(distinct ward) as count_ward,
                    COUNT(distinct district) as count_district
            FROM
            `bigquery-public-data.chicago_crime.crime`
            where 
                Date_diff(EXTRACT(Date from date), date_add(CURRENT_DATE(),INTERVAL -12 MONTH), DAY)<365 and 
                arrest=False
                and CHAR_LENGTH(location)>0
        """
response1 = chicago_crime.query_to_pandas_safe(query1)
response1
query1 = """SELECT 
                    count(unique_key) as count_crimes,
                    block
            FROM
            `bigquery-public-data.chicago_crime.crime`
            where 
                Date_diff(EXTRACT(Date from date), date_add(CURRENT_DATE(),INTERVAL -12 MONTH), DAY)<365 and 
                arrest=False
                and CHAR_LENGTH(location)>0
            group by block
            order by count(unique_key) desc;
        """
response1 = chicago_crime.query_to_pandas_safe(query1)
response1.head()
response1=response1.sort_values(by='count_crimes',ascending=False)
response1.head(100).plot.bar(x='block', y='count_crimes', rot=0)
response1.head(15)
query1 = """SELECT 
                    count(unique_key) as count_crimes,
                    EXTRACT(HOUR from date) as hour,
                    block
            FROM
            `bigquery-public-data.chicago_crime.crime`
            where 
                Date_diff(EXTRACT(Date from date), date_add(CURRENT_DATE(),INTERVAL -12 MONTH), DAY)<365 and 
                arrest=False
                and CHAR_LENGTH(location)>0
            group by block, EXTRACT(HOUR from date)
        """
response1 = chicago_crime.query_to_pandas_safe(query1)
response1.head()
response1[['count_crimes','hour']].groupby(by='hour').quantile(np.around(np.arange(0.1,1.1,0.1),2))
grouped = response1[['count_crimes','hour']].groupby(by='hour').quantile(np.around(np.arange(0.1,1.1,0.1),2))
grouped.reset_index(inplace=True)
grouped.rename(columns={"level_1": "quantile"},inplace=True)
for hour_i in grouped['hour'].unique():
    print(grouped[grouped['hour']==hour_i])
grouped = response1[['count_crimes','hour']].groupby(by='hour').quantile([0.9,0.95,0.975,0.99,1])
grouped.reset_index(inplace=True)
grouped.rename(columns={"level_1": "quantile"},inplace=True)

for hour_i in grouped['hour'].unique():
    print(grouped[grouped['hour']==hour_i])
response1[response1['count_crimes']>25].shape
query1 = """SELECT 
                   Distinct
                   location_description
            FROM
            `bigquery-public-data.chicago_crime.crime`
            where 
                Date_diff(EXTRACT(Date from date), date_add(CURRENT_DATE(),INTERVAL -12 MONTH), DAY)<365 and 
                arrest=False
                and CHAR_LENGTH(location)>0
        """
response1 = chicago_crime.query_to_pandas_safe(query1)
response1.head()
len(response1.location_description)
query1 = """SELECT 
                    count(unique_key) as count_crimes,
                    block
            FROM
            `bigquery-public-data.chicago_crime.crime`
            where 
                Date_diff(EXTRACT(Date from date), date_add(CURRENT_DATE(),INTERVAL -12 MONTH), DAY)<365 and 
                arrest=False
                and CHAR_LENGTH(location)>0
            group by block
            order by count(unique_key) desc;
        """
response1 = chicago_crime.query_to_pandas_safe(query1)
response1.head()
response1.quantile([0.8,0.9,0.95,0.975,0.99,1])
# Первая смена
query1 = """SELECT 
                    count(unique_key) as count_crimes,
                    EXTRACT(Hour from date) as hour,
                    block
            FROM
            `bigquery-public-data.chicago_crime.crime`
            where 
                Date_diff(EXTRACT(Date from date), date_add(CURRENT_DATE(),INTERVAL -12 MONTH), DAY)<365 and 
                arrest=False
                and CHAR_LENGTH(location)>0
                and EXTRACT(Hour from date) in (22,23,0)
            group by block,EXTRACT(Hour from date)
            ;
        """
response1 = chicago_crime.query_to_pandas_safe(query1)
response1.head()
response1['count_crimes'].quantile([0.8,0.9,0.95,0.975,0.99,1])
response1[response1['hour']==22].sort_values(by=['count_crimes'],ascending=False).head(10)
response1[response1['hour']==23].sort_values(by=['count_crimes'],ascending=False).head(10)
response1[response1['hour']==0].sort_values(by=['count_crimes'],ascending=False).head(10)
a1 = response1[response1['hour']==0].sort_values(by=['count_crimes'],ascending=False).head(10)['block']
a2 = response1[response1['hour']==22].sort_values(by=['count_crimes'],ascending=False).head(10)['block']
a3 = response1[response1['hour']==23].sort_values(by=['count_crimes'],ascending=False).head(10)['block']
set(sum([list(a1),list(a2),list(a3)],[]))
## Улицы рядом. Найдем для них расстояния
query1 = """SELECT 
                    avg(latitude) as latitude,
                    avg(longitude) as longitude,
                    block
            FROM
            `bigquery-public-data.chicago_crime.crime`
            where 
                Date_diff(EXTRACT(Date from date), date_add(CURRENT_DATE(),INTERVAL -12 MONTH), DAY)<365 and 
                arrest=False
                and CHAR_LENGTH(location)>0
                and block in ('0000X W DIVISION ST',
                             '0000X W HUBBARD ST',
                             '001XX N STATE ST',
                             '006XX N DEARBORN ST',
                             '015XX N MILWAUKEE AVE',
                             '022XX N LINCOLN AVE',
                             '033XX N HALSTED ST',
                             '035XX N CLARK ST',
                             '035XX S RHODES AVE',
                             '063XX S DR MARTIN LUTHER KING JR DR',
                             '064XX S DR MARTIN LUTHER KING JR DR',
                             '065XX S DR MARTIN LUTHER KING JR DR',
                             '074XX S SOUTH SHORE DR',
                             '078XX S SOUTH SHORE DR',
                             '100XX W OHARE ST')
                group by block               
        """
blocks = chicago_crime.query_to_pandas_safe(query1)
blocks.head()
d = {'latitude': ['41.8458578'], 'longitude': ['-87.7053591'],'block':['home']}
df = pd.DataFrame(data=d)
df.head()
df['latitude']= df['latitude'].astype(float)
df['longitude'] = df['longitude'].astype(float)
blocks=pd.concat([blocks, df])
# to radians
blocks['latitude'] = np.radians(blocks['latitude'])
blocks['longitude'] = np.radians(blocks['longitude'])
from sklearn.neighbors import DistanceMetric
dist = DistanceMetric.get_metric('haversine')
pd.DataFrame(dist.pairwise(blocks[['latitude','longitude']].to_numpy())*6373,  columns=blocks.block.unique(), index=blocks.block.unique())
query1 = """SELECT 
                    count(*) as crimes,
                    primary_type
            FROM
            `bigquery-public-data.chicago_crime.crime`
            where 
                Date_diff(EXTRACT(Date from date), date_add(CURRENT_DATE(),INTERVAL -12 MONTH), DAY)<365 and 
                arrest=False
                and CHAR_LENGTH(location)>0
                and EXTRACT(Hour from date) in (22,23,0)
                and block in ('0000X W HUBBARD ST',
                             '035XX S RHODES AVE',
                             '063XX S DR MARTIN LUTHER KING JR DR',
                             '064XX S DR MARTIN LUTHER KING JR DR',
                             '065XX S DR MARTIN LUTHER KING JR DR')
                group by primary_type
        """
blocks = chicago_crime.query_to_pandas_safe(query1)
blocks.head()
blocks['primary_type'].values.tolist()
blocks['crime_type'] = np.where(blocks['primary_type'].isin(['CRIM SEXUAL ASSAULT', 'SEX OFFENSE', 'ASSAULT','BURGLARY','CRIMINAL DAMAGE', 'CRIMINAL SEXUAL ASSAULT', 'WEAPONS VIOLATION', 'KIDNAPPING', 'HOMICIDE']),"gun","not_gun")
blocks[['crimes','crime_type']].groupby('crime_type').sum()
