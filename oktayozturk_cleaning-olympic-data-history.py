# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/athlete_events.csv')
# Head shows first 5 rows
data.head()
# Head shows last 5 rows
data.tail()
# Columns gives column names of features
data.columns
# Shape gives number of rows and columns in a tuble
data.shape
# Info gives data type like dataframe, number of sample or row, number of feature or column, feature types and memory usage
data.info()
# For example lets look frequency of Team types
print(data["Team"].value_counts(dropna = False))
# Describe show the statistics features
data.describe()
data.boxplot(column = 'Height', by = 'Age') 
# Show the data types
data.dtypes
# Information about the data
data.info()
# Lets check Weight for NaN values
data["Weight"].value_counts(dropna = False)
# Lets drop nan values
data_1 = data
data_1["Weight"].dropna(inplace = True)

assert data_1["Weight"].notnull().all()
data["Weight"].fillna('empty',inplace = True)
assert data["Weight"].notnull().all()
data["Weight"].value_counts(dropna = False)
