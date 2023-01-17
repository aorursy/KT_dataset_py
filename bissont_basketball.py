# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
import bq_helper
# Any results you write to the current directory are saved as output.
# create a helper object for this dataset
ncaa = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                              dataset_name="ncaa_basketball")
ncaa.list_tables()
ncaa.table_schema("mbb_teams_games_sr")
ncaa.head("mbb_teams_games_sr")
# three pointerers made per season
query = "select sum(opp_three_points_made) as three_pointers, season from `bigquery-public-data.ncaa_basketball.mbb_teams_games_sr` group by season order by three_pointers desc limit 10 " 
ncaa.query_to_pandas_safe(query, max_gb_scanned=1)

