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



# WildFire in US

import pandas as pd

import sqlite3

import matplotlib.pyplot as plt

import seaborn as sns



conn = sqlite3.connect('../input/FPA_FOD_20170508.sqlite')

df = pd.read_sql_query("SELECT * FROM 'Fires'", conn)

df.head()
#Creating a boolean w T/Fs

ca_only = df['STATE'] == 'CA'



#Slicing dataset into only fires within CA

ca_only_ds = df[ca_only]





list(ca_only_ds)
# 

fire_year = df.groupby('FIRE_YEAR').size()

plt.bar(range(len(fire_year)), fire_year.values, width = 0.5)

plt.xticks(range(len(fire_year)), fire_year.index, rotation = 50)

plt.show()
fire_date = df.groupby('DISCOVERY_DOY').size()

plt.scatter(fire_date.index, fire_date.values)

plt.show()
sns.heatmap(pd.crosstab(df.FIRE_YEAR, df.STAT_CAUSE_DESCR))

plt.show()