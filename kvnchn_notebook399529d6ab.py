# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import sqlite3 as sq



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
#just reading data

con = sq.connect("../input/database.sqlite")

team_att = pd.read_sql_query("SELECT * from Team_Attributes", con)

team = pd.read_sql_query("SELECT * from Team", con)

match = pd.read_sql_query("SELECT * from Match", con)

match = match[['date', 'home_team_goal', 'away_team_goal', 'home_team_api_id', 'away_team_api_id', 

              'goal', 'shoton', 'shotoff', 'foulcommit', 'card', 'cross', 'corner', 'possession',

              'B365H', 'B365D', 'B365A', 'BWH', 'BWD', 'BWA', 'IWH', 'IWD', 'IWA', 'LBH', 'LBD',

              'LBA', 'PSH', 'PSD', 'PSA', 'WHH', 'WHD', 'WHA', 'SJH', 'SJD', 'SJA', 'VCH', 'VCD',

              'VCA', 'GBH', 'GBD', 'GBA', 'BSH', 'BSD', 'BSA']]
#shuffle match rows so split tables are randomized

match = match.reindex(np.random.permutation(match.index))



#split match data into training, validation, and test sets

m_train = match.iloc[:17861]

m_valid = match.iloc[17861:21108]

m_test = match.iloc[21108:]
match