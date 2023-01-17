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
# Imports

import bq_helper
ncaa_basketball = bq_helper.BigQueryHelper(active_project="bigquery-public-data",

                                          dataset_name="ncaa_basketball")
ncaa_basketball.list_tables()
ncaa_basketball.head("mbb_games_sr")
# This query looks in the "mbb_teams_games_sr" table and gets the 'three_points_made' column

query = """SELECT *

            FROM `bigquery-public-data.ncaa_basketball.mbb_games_sr`"""



# Check how big this query will be in GB

ncaa_basketball.estimate_query_size(query)
mbb_games_sr = ncaa_basketball.query_to_pandas_safe(query, max_gb_scanned=0.1)
mbb_games_sr.mean()




# Calculate total number of cells in dataframe

totalCells = np.product(mbb_games_sr.shape)



# Count number of missing values per column

missingCount = mbb_games_sr.isnull().sum()



# Calculate total number of missing values

totalMissing = missingCount.sum()



# Calculate percentage of missing values

print("The mbb_games_sr dataset contains", round(((totalMissing/totalCells) * 100), 2), "%", "missing values and a total of ", totalMissing,".")

print("Total columns", totalCells/132,".")

missingCount[['a_points_off_turnovers', 'a_foulouts']]
# One decision to make is to determine the relevancy of the missing information 

# and if it will affect our model. If rows with missing information were removed 

#we would effectively remove our entire data set. 