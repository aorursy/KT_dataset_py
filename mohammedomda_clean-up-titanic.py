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
df = pd.read_csv('../input/titanic/test.csv', index_col=0)
df.head()
# Clean up the dataset & handle the missing values: 
missing_values = df.isnull().sum()
missing_values

# Figure out the percent of missing data : 
total_cells = np.product(df.shape)
total_missing = missing_values.sum()
# percent is : 
percent = (total_missing / total_cells) * 100
print(percent)
# remove all columns with at least one missing value : 
col_with_one_na = df.dropna(axis=1)
col_with_one_na.head()
# Detecting how much data we lose : 
print("Columns with original dataset: %d \n" % df.shape[1])
print("Columns with na's was dropped out: %d \n" % col_with_one_na.shape[1] )
# replace all NA's with the value that comes directly after it in the same columns ,
# then repalce all the remaining na's with 0
df.fillna(method='bfill', axis=0).fillna(0)
