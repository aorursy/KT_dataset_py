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
import sqlite3, time, csv, re, random

sql_conn = sqlite3.connect('../input/database.sqlite')

lmt = 10000
data = sql_conn.execute("SELECT body, score, subreddit FROM May2015\
                                WHERE gilded = 1 \
                                ORDER BY score desc\
                                LIMIT " + str(lmt))
for x in range(0, 100):
    print(data.fetchone())
    print('')
