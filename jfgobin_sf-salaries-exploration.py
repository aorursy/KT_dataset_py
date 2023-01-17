# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import sqlite3 as sql

# Connect to the database
conn = sql.connect('../input/database.sqlite')

# Let's get a few stats on the dataset
dbcursor = conn.cursor()
print("Year   Number of entries")
print("========================")
for row in dbcursor.execute("SELECT year, count(*) as cnt from Salaries group by year"):
    print("%4s   %d"%(row[0], row[1]))
# This is fairly balanced
print("")
print("Year  Min Salary  Avg Salary  Max Salary")
print("========================================")
for row in dbcursor.execute("SELECT year, min(TotalPay) as smin, " +
                            "avg(TotalPay) as savg, max(TotalPay) as smax " + 
                            "from Salaries group by year"):
    print("%4s  %10.2f  %10.2f  %10.2f"%(row[0], row[1], row[2], row[3]))
# Surprinsingly, some people did have a negative compensation. Also the biggest salary got cut down (aboout 20%)
# and crawled slowly back up. 
# 2014 saw the same average salary as 2012, which is largely above the average from 2011.



# Done ... close the connection
conn.close()

