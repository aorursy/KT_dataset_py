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
import time
# Set your own project id here

PROJECT_ID = 'YOUR_OWN_PROJECT_ID'

from google.cloud import bigquery

client = bigquery.Client(project=PROJECT_ID, location="US")
# Set config

path = "../input/"

dataset_id = 'titanic'

dataset_ref = client.dataset(dataset_id)

job_config = bigquery.LoadJobConfig()

job_config.source_format = bigquery.SourceFormat.CSV

job_config.skip_leading_rows = 1

job_config.autodetect = True
# Create dataset

existing_datasets = [dataset.dataset_id for dataset in client.list_datasets()]

if not dataset_id in existing_datasets:

    client.create_dataset(dataset_id)
dataset_ref = client.dataset(dataset_id)
# Create table

existing_tables = [table.table_id for table in client.list_tables(dataset_id)]

for filename in os.listdir(path):

    print (path + filename)

    table_id = filename.rstrip(".csv")

    if not table_id in existing_tables:

        table_ref = dataset_ref.table(table_id)

        with open((path + filename), "rb") as source_file:

            job = client.load_table_from_file(source_file, table_ref, job_config=job_config)

        job.result()
# Fetch train table

table_ref = dataset_ref.table("train")

train = client.get_table(table_ref)
train.schema
client.list_rows(train, max_results=5).to_dataframe()
train_query = """

CREATE OR REPLACE MODEL `titanic.logistic_reg_model`

OPTIONS (

  model_type = 'logistic_reg'

  ) AS (

SELECT

  Pclass, Name, Sex, Age, SibSp, Parch, Ticket, Fare, Cabin, Embarked,  Survived AS label

FROM

  `titanic.train`)

        """
train_query_job = client.query(train_query)
# Polling

while(1):

    if client.get_job(train_query_job.job_id).state=="DONE":

        break

    time.sleep(5)
predict_query = """

SELECT

 * 

FROM

  ML.PREDICT(MODEL `titanic.logistic_reg_model`, (

SELECT

  Pclass, Name, Sex, Age, SibSp, Parch, Ticket, Fare, Cabin, Embarked, PassengerId



FROM

  `titanic.test`

))

      """ 
predict_query_job = client.query(predict_query)
predicted = predict_query_job.to_dataframe()
predicted.head()
sub = pd.read_csv("../input/test.csv",header=0)
sub = pd.merge(sub, predicted, on='PassengerId', how='right')
sub = sub[["PassengerId","predicted_label"]].rename(index=str, columns={"predicted_label": "Survived"})
sub.head()
sub.to_csv(('submit.csv'),index=False)