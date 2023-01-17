# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
from google.cloud import bigquery

client = bigquery.Client()
javascript_query='''

CREATE TEMPORARY FUNCTION 

getDependenciesInPackageJson(content STRING)

RETURNS STRING

LANGUAGE js AS

"""



var res = '';

try {

    var x = JSON.parse(content);

    

    var list_dep = [];

    if (x.dependencies) {

      list_dep = Object.keys(x.dependencies);

    }

    

    var list_devdep = [];

    if (x.devDependencies) {

      list_devdep = Object.keys(x.devDependencies);

    }

    

    var list_alldep = list_dep.concat(list_devdep)

    res = list_alldep.join(',')

    



} catch (e) {}



return res;

""";

'''

print (javascript_query)
# --- Specify the table names 



# # Use sample tables when testing out the query

# ds_files = 'bigquery-public-data.github_repos.sample_files'

# ds_contents = 'bigquery-public-data.github_repos.sample_contents'



ds_files = 'bigquery-public-data.github_repos.files'

ds_contents = 'bigquery-public-data.github_repos.contents'





# --- Specify a list of interested testing framework

list_fw = [

    'mocha', 

    'jest',

    'jasmine',

    'qunit',

    'funcunit',

    'cypress',

    'puppeteer',

    'chai',

    'sinon'

]
my_sql_query=('''

WITH t_dep AS (

    SELECT 

        tf.id AS id, 

        tf.repo_name AS repo_name, 

        getDependenciesInPackageJson(tc.content) AS package_dep

    FROM (

        SELECT id, repo_name, path

        FROM `{}`

            WHERE path LIKE "package.json" 

    ) AS tf

    LEFT JOIN

      `{}` tc

    ON

      tf.id = tc.id

),



t_dep_check AS (

    SELECT repo_name, package_dep,

        REGEXP_CONTAINS(package_dep, r"{}") AS is_interested

    FROM t_dep

)



SELECT repo_name, package_dep

FROM t_dep_check

    WHERE is_interested

''').format(ds_files, ds_contents, '|'.join(list_fw))
final_query=javascript_query+my_sql_query

print (final_query)
my_job_config = bigquery.job.QueryJobConfig()

my_job_config.dry_run = True



my_job = client.query(final_query, job_config=my_job_config)

BYTES_PER_GB = 2**30

my_job.total_bytes_processed / BYTES_PER_GB
query_contents = client.query(final_query)



# Create a dataframe from the queried results

df_contents = query_contents.to_dataframe()
# Make a copy of this dataframe before cleaning & transforming it

df_interested = df_contents.copy()



# Sort by the 'repo_name' column

df_interested = df_interested.sort_values(by='repo_name')
# Inspect the data 

df_contents.head()
df_interested.shape
for cur_fw in list_fw:

    df_interested[cur_fw] = df_interested.package_dep.str.contains(cur_fw)
df_interested[list_fw].sum(axis=0).sort_values(ascending=False)
df_interested.to_csv("github_package_deps_June_2019.csv",index=False)
