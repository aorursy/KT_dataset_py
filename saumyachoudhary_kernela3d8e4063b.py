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
import bq_helper

from bq_helper import BigQueryHelper



patents = bq_helper.BigQueryHelper(active_project="patents-public-data",

                                   dataset_name="patents")
bq_assistant = BigQueryHelper("patents-public-data", "patents")

bq_assistant.list_tables()
bq_assistant.head("publications")
bq_assistant.table_schema("publications").iloc[15:40]
query1 = """

SELECT

  claims.text

FROM

  `patents-public-data.patents.publications`, UNNEST(claims_localized) AS claims

LIMIT

  500;

        """

response1 = patents.query_to_pandas_safe(query1, max_gb_scanned=700)

response1
response1AsSeries = pd.Series(response1["text"])

response1AsSeries
semicolonCount = response1AsSeries.str.count(";")

semicolonCount
whereinCount = response1AsSeries.str.count("wherein")

whereinCount
saidCount = response1AsSeries.str.count("said")

saidCount
response1["semicolon count"] = semicolonCount

response1["wherein count"] = whereinCount

response1["said count"] = saidCount

response1
response1.to_csv("characterCountInClaims.csv", index=False)