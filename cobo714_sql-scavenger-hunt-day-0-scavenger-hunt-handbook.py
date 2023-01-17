# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import bq_helper

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
openAQ = bq_helper.BigQueryHelper(active_project="bigquery-public-data", dataset_name="openaq")

openAQ.list_tables()
openAQ.table_schema("global_air_quality")
openAQ.head("global_air_quality")
query = """SELECT pollutant
           FROM `bigquery-public-data.openaq.global_air_quality`
           WHERE country = "US" """
openAQ.estimate_query_size(query)
AmericanPollut = openAQ.query_to_pandas(query)
AmericanPollut.pollutant.mode()
AmericanPollut.to_csv("AmericanPollut.csv")