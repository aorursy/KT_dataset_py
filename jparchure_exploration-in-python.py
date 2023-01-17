# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import sqlite3



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.

#Loading database

experiments = sqlite3.connect("../input/database.sqlite")



cur = experiments.cursor()

#Checking if the connection works by calculating nrow(table)

for i in cur.execute('SELECT COUNT(*) FROM resultsdata15;'):

    print(i)



results = pd.read_sql('SELECT * FROM resultsdata15',con=experiments)

print(results.head())



samples = pd.read_sql('SELECT * FROM sampledata15',con=experiments)

print(samples.head())
print(results.columns) #Get columns

print(samples.columns)