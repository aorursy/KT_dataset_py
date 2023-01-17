# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Read the county data

df_county = pd.read_csv('/kaggle/input/us-census-demographic-data/acs2017_county_data.csv')

df_county.head()
# Find Mecklenburg County

df_county.loc[(df_county['County'] == 'Mecklenburg County') & (df_county['State'] == 'North Carolina')]
df_tract = pd.read_csv('/kaggle/input/us-census-demographic-data/acs2017_census_tract_data.csv')

df_tract.head()
df_meck_tract = df_tract.loc[(df_tract['County'] == 'Mecklenburg County') & (df_tract['State'] == 'North Carolina')]

df_meck_tract.shape
df_meck_tract.columns