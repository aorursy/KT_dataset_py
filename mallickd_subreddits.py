# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



# Any results you write to the current directory are saved as output.



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import sqlite3



conn = sqlite3.connect("../input/database.sqlite")

c = conn.cursor()

num_entries = 0



for row in c.execute("SELECT * FROM May2015"):

    num_entries += 1



print(num_entries)
