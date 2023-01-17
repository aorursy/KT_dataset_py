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



import bq_helper

from bq_helper import BigQueryHelper

# https://www.kaggle.com/sohier/introduction-to-the-bq-helper-package

wdi = bq_helper.BigQueryHelper(active_project="patents-public-data",

                                   dataset_name="worldbank_wdi")
#Initialize the BigQuery Environment and print the names of all tables in our input.



bq_assistant2 = BigQueryHelper("patents-public-data", "worldbank_wdi")

bq_assistant2.list_tables()



#Print out all field names of the WDI table.



bq_assistant2.table_schema('wdi_2016')

# Create a dataframe with all data of interest.



query1 = """

SELECT *

FROM

  `patents-public-data.worldbank_wdi.wdi_2016`

WHERE

    indicator_name="GDP per capita (current US$)" OR indicator_name="Inflation, consumer prices (annual %)" 

    OR indicator_name="GDP growth (annual %)";

        """

response1 = wdi.query_to_pandas_safe(query1,max_gb_scanned=3)







# Create and save the output file.



mon_fichier = open("wdi_database.csv", "w")

mon_fichier.write(response1.to_csv())

mon_fichier.close()
