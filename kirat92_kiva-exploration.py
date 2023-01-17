# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
loans = pd.read_csv("../input/kiva_loans.csv")
locations = pd.read_csv("../input/kiva_mpi_region_locations.csv")
themes = pd.read_csv("../input/loan_theme_ids.csv")
orgs = pd.read_csv("../input/loan_themes_by_region.csv")
orgs.head()
loans_india = loans.loc[loans.country == 'India',:]
loans_india.head()
#Cleaning
#Changing all region characters to lower cases
loans_india.loc[:,'region'] = loans_india.region.str.lower()
#Removing all additional information beyond the town name
loans_india.loc[:,'city'] = loans_india.region.str.split(",").str[0]


loans_india.loc[:,'city'].value_counts()

loans_india.groupby('city').loan_amount.sum().sort_values(ascending=False)
avg_by_city = loans_india.groupby('city').loan_amount.mean().sort_values(ascending=False)
avg_by_city.head(10)
loans_india.sector.value_counts()
loans_india.groupby('sector').loan_amount.mean().sort_values(ascending=False)
top_ten_total_lent = loans_india.loc[loans_india.city.isin(['jeypore','falakata','dhupguri','semliguda','titilagarh','sonepur','khurda','odagaon','surendranagar','muniguda' ])]
sector_patterns = top_ten_total_lent.groupby(['city','sector']).loan_amount.agg(['mean','count','sum'])
sector_patterns.head()
