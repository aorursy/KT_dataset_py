# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.




# You can read in the SQLite datbase like this

import sqlite3

con = sqlite3.connect('../input/database.sqlite')

sample = pd.read_sql_query("""

SELECT INSTNM,

       COSTT4_A AverageCostOfAttendance,

       Year

FROM Scorecard

WHERE INSTNM='Wellesley College'""", con)

print(sample)



womenonly = pd.read_sql_query("""

SELECT WOMENONLY,

       INSTNM,

       COSTT4_A AverageCostOfAttendance,

       SAT_AVG,

       ADM_RATE,

       PCIP26

FROM Scorecard

WHERE WOMENONLY='Yes'""", con)

sw = womenonly.sort_values(womenonly.columns[2])

print(sw.describe())

print(sw.tail(7))
print (sw.head(5))
import pandas as pd

# You can read a CSV file like this

scorecard = pd.read_csv("../input/Scorecard.csv")

scorecard.tail(5)
scorecard.dtypes
# Which colleges are in the range of a SAT score

sat1400 = scorecard.query('SAT_AVG > 1400 & SAT_AVG < 1450')

#print (sat1400['CITY'])

# Which colleges are in Boston area

#bsat1400 = sat1400[sat1400['City'] == 'Boston']

#print (wsat1400['INSTNM'])

# ADM_RATE_ALL

# 

import seaborn as sns

corrmat = sat1400.corr()

sns.heatmap(corrmat, 

            xticklabels=corrmat.columns.values,

           yticklabels=corrmat.columns.values)