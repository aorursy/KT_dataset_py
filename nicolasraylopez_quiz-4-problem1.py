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
us_counties = pd.read_csv('/kaggle/input/us-census-demographic-data/acs2017_county_data.csv')

us_counties.head()
us_counties.tail()
us_counties.shape
us_counties.columns
us_counties.index
us_counties[us_counties.TotalPop >= 100000]
us_counties.set_index('County').loc['Los Angeles County']
us_counties.sort_values('TotalPop', ascending=False).iloc[round(us_counties.shape[0]/2),:]
us_counties.loc[:,'TotalPop'].sum()
us_counties.loc[:,'TotalPop'].mean()
us_counties.set_index('Transit').loc[us_counties.loc[:,'Transit'].max()]
us_counties[us_counties.Transit >= 30].loc[:,'MeanCommute'].mean()
us_counties.loc[:,'Income'].median()
us_counties.set_index('Poverty').loc[us_counties.loc[:,'Poverty'].min()]
us_counties.groupby('State').size()
us_counties.groupby('State').mean().loc[:,'MeanCommute']
us_counties.groupby('State').mean().loc[:,'Drive']
us_counties.groupby('State').sum().loc[:,'TotalPop'].sort_values(ascending=False).head(10).plot.pie()
us_counties.groupby('State').mean().loc[:,'Hispanic'].plot.bar()