# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# read from file, csv (comma separated values), into a pandas dataframe data type

df = pd.read_csv("/kaggle/input/medium-articles-dataset/medium_data.csv", na_values = ['Read'])

# what is this data set?

df.head()
# what are the datatypes?

for (columnName, columnData) in df.iteritems():

    print("%s: %s" % (columnName, df[columnName].dtypes))

    

print(df.responses.unique())

print(df['responses'].dtypes)



df.responses.fillna(0)

print(df.responses.fillna(0).unique())



df.responses = df.responses.fillna(0).astype(int)

print(df['responses'].dtypes)

print(df.date.unique())

print(df['date'].dtypes)



df.date = pd.to_datetime(df.date, format='%Y-%m-%d')

print(df['date'].dtypes)
