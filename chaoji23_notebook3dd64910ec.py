# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import matplotlib.pyplot as plt

import sqlite3

import datetime

from collections import Counter



sql_conn = sqlite3.connect('../input/database.sqlite')

df = pd.read_sql('SELECT created_utc FROM May2015 WHERE created_utc % 10 = 5', sql_conn)

df = list(df.created_utc)



counter = dict()

for time in df:

    date = datetime.datetime.fromtimestamp(time)

    str_rep = str(date.month) + '-' + '{0:02d}'.format(date.day)

    if str_rep not in counter:

        counter[str_rep] = 0

    counter[str_rep] += 1    



label = counter.keys()

label = sorted(label)

counts = map(lambda x: counter[x], label)

print(counts)

#plt.bar(np.arange(1, 33) - 0.2, counts, 0.4, tick_label = label, color='#00ffff')

#plt.xlabel('dates')

#plt.ylabel('Num of reddits')

#plt.show()

#plt.savefig('periodic.png')



# Any results you write to the current directory are saved as output.