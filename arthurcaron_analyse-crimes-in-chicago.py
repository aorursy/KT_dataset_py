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
df = pd.read_csv("/kaggle/input/crimes-in-chicago/Chicago_Crimes_2012_to_2017.csv")
df.head()
# Define display rules
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)


# Drop the unused columns.
df.columns

unused_columns = ['Updated On', 'X Coordinate', 'Y Coordinate']
for col in unused_columns:
    del df[col]
df.groupby(['Year'])['Date'].agg(['min', 'max'])

# The year 2017 is too short (first date : 01/01/2017 / last date : 01/18/2017), we need to delete it.
df = df[df['Year'] != 2017]

# Multiple Primary Type ('NON - CRIMINAL', 'NON-CRIMINAL (SUBJECT SPECIFIED)' and 'NON-CRIMINAL') group in the Primary Type 'NON-CRIMINAL'
df['Primary Type'] = ['NON-CRIMINAL' if e in ['NON - CRIMINAL', 'NON-CRIMINAL (SUBJECT SPECIFIED)'] else e for e in df['Primary Type']]


# Create the crosstab
crosstab = pd.crosstab(df['Primary Type'], df['Year'], values = df['Description'], aggfunc = 'count')

# Clean the crosstab
crosstab = crosstab.fillna(0)
crosstab = crosstab.astype(int)


# Create the new columns 'min', 'max' and 'mean'
count_primary_type_by_year = df.groupby(['Year', 'Primary Type'])['Description'].count().reset_index(name = 'count')
crosstab_agg = pd.DataFrame(count_primary_type_by_year.groupby(['Primary Type'])['count'].agg(['min', 'max', 'mean']))

# Join the crosstab with the crosstab_agg
crosstab = crosstab.join(crosstab_agg, on='Primary Type')
crosstab
df.head()