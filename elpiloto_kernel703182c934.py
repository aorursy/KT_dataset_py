# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Let's Inspect what is going on in the csv file by printing out the first 10 rows.
# You can't load and manipulate data if you're not familiar with its contents, so this is always the first step.
import csv
csv_file = open('/kaggle/input/cat-in-the-dat/train.csv')
csv_reader = csv.reader(csv_file)
for _ in range(10):
    row = next(csv_reader)
    print(row)
csv_file.close()
csv_file = open('/kaggle/input/cat-in-the-dat/train.csv')
csv_reader = csv.reader(csv_file)
header = next(csv_reader)
row = next(csv_reader)
csv_file.close()

num_columns = len(header)
# print out name and value and type of data
for c in range(num_columns):
    col = header[c]
    value = row[c]
    print('======================')
    print('column name: {}'.format(col))
    print('value: {}'.format(value))
    # I'm printing out the type of the value here to emphasize that it csv loads everything as a string
    # and that we'll have to convert it to the correct data format later if we want to feed this data
    # into any algorithms.
    print('type: {}'.format(type(value)))  
# use easy import
df_train=pd.read_csv('../input/cat-in-the-dat/train.csv')
print(df_train.head())


ordinal_columns = {}
for c in df_train.columns:
    if 'ord' in c:
        ordinal_columns[c] = df_train[c].unique()
from pprint import pprint
pprint(ordinal_columns)
df_train.keys()
import matplotlib.pyplot as plt
days = np.array(range(1,13))

print(np.sin(days), np.cos(days))

plt.scatter(np.sin(days), np.cos(days))