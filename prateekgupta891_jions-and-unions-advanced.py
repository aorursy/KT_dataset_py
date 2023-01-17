# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from google.cloud import bigquery

client = bigquery.Client()

dataset_ref = client.dataset("hacker_news",project="bigquery-public-data")

dataset = client.get_dataset(dataset_ref)

table_ref = dataset_ref.table("comments")

table = client.get_table(table_ref)

client.list_rows(table,max_results = 5).to_dataframe()
table_ref = dataset_ref.table('stories')

table = client.get_table(table_ref)

client.list_rows(table,max_results = 5).to_dataframe()
query =""" 

with comments as

(

select parent, count(1) as num_com

from `bigquery-public-data.hacker_news.comments` as comments

group by parent

)

select stories.id as story_id, stories.by , stories.text as story, comments.num_com as comment

from `bigquery-public-data.hacker_news.stories` as stories

left join comments on

stories.id = comments.parent

where extract(date from stories.time_ts) = '2012-01-01'

order by comment desc

"""
join_result = client.query(query).result().to_dataframe()

join_result.head()
query = """

with new_table as

(

select stories.by 

from `bigquery-public-data.hacker_news.stories` as stories

where extract(date from stories.time_ts) = '2014-01-01'

union all

select comments.by

from `bigquery-public-data.hacker_news.comments` as comments

where extract(date from comments.time_ts) = '2014-01-01'

)

select nt.by

from new_table as nt

order by nt.by desc

"""
union_results = client.query(query).result().to_dataframe()

print(union_results.head())

print(len(union_results))