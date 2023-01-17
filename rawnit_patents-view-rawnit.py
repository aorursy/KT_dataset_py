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



# Create a "Client" object

client = bigquery.Client()



# Construct a reference to the "openaq" dataset

dataset_ref = client.dataset("patentsview", project="patents-public-data")



# API request - fetch the dataset

dataset = client.get_dataset(dataset_ref)



# List all the tables in the "openaq" dataset

tables = list(client.list_tables(dataset))



# Print names of all tables in the dataset (there's only one!)

for table in tables:  

    print(table.table_id)



# Construct a reference to the "global_air_quality" table

table_ref = dataset_ref.table("claim")



# API request - fetch the table

table = client.get_table(table_ref)



# Preview the first five lines of the "global_air_quality" table

client.list_rows(table, max_results=5).to_dataframe()
table.schema
# Query to select all the items from the "city" column where the "country" column is 'US'

query = """

        SELECT text

        FROM `patents-public-data.patentsview.claim`

        WHERE patent_id = '8058265'

        """
# Set up the query

query_job = client.query(query)
import pandas as pd
# API request - run the query, and return a pandas DataFrame



text = query_job.to_dataframe()
text
# What five cities have the most measurements?

text.text.value_counts().head()
#!conda install nltk

import nltk

nltk.download_shell()

from nltk.corpus import stopwords

stopwords.words('english')[0:10] # Show some stop words
import string

mess = text.to_string()

mess


# function to test if something is a noun

is_noun = lambda pos: pos[:2] == 'NN'

# do the nlp stuff

tokenized = nltk.word_tokenize(mess)

nouns = [word for (word, pos) in nltk.pos_tag(tokenized) if is_noun(pos)] 



print(nouns)

# Construct a reference to the "global_air_quality" table

table_ref = dataset_ref.table("cpc_subsection")



# API request - fetch the table

table = client.get_table(table_ref)



# Preview the first five lines of the "global_air_quality" table

client.list_rows(table, max_results=5).to_dataframe()
query = """

        SELECT patentsview.cpc_subsection

        FROM `patents-public-data.patentsview.cpc_subsection` AS cpc

        INNER JOIN `patents-public-data.patentsview.claim` AS claim

            ON cpc.title = claim.text

           

        """



query_job = client.query(query)

ans = query_job.to_dataframe()

ans