# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
#print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# Let pandas create an index to make slicing easier --
# "Community health" index starts from 1, which might be confusing
deaths = pd.read_csv('./chicago-public-health-statistics/public-health-statistics-selected-underlying-causes-of-death-in-chicago-2006-2010.csv')
health = pd.read_csv('./chicago-public-health-statistics/public-health-statistics-selected-public-health-indicators-by-chicago-community-area.csv')
deaths.head()
np.sort(np.unique(deaths['Cause of Death']))
deaths[deaths['Community Area'] == 0]['Community Area Name'].unique()
# Look for the different field names
causes = pd.Series(deaths['Cause of Death'].unique())
cancer = causes[causes.str.lower().str.contains('cancer')]
print(cancer)
# Separate deaths by all cancer and deaths by subcategories of cancer in the data
subcancer = deaths[(deaths['Cause of Death'].isin(cancer)) & (deaths['Cause of Death'] != 'Cancer (all sites)')]
cancer_all = deaths[deaths['Cause of Death'] == 'Cancer (all sites)']
print(subcancer['Cumulative Deaths 2006 - 2010'].sum() )
print(cancer_all['Cumulative Deaths 2006 - 2010'].sum())
len(np.unique(deaths['Community Area']))
deaths.groupby('Cause of Death')['Cumulative Deaths 2006 - 2010'].sum().sort_values(ascending = False).head(10)
import matplotlib.pyplot as plt
% matplotlib inline
# Exclude the cause of death from the subtotal if it contains "all"
specific_deaths = deaths[~(deaths['Cause of Death'].str.lower().str.contains('all'))]
# Totals for city
sd_gb = specific_deaths.groupby('Cause of Death')['Cumulative Deaths 2006 - 2010'].sum()
sd_gb.sort_values(ascending = False)
fig, ax = plt.subplots(figsize=(10,5))
sd_gb.sort_values().plot(kind='barh')
plt.ylabel('')
plt.title(sd_gb.index.name + ' -  Chicago, citywide')
health.head()
health.columns
health.merge(deaths, left_on = 'Community Area', right_on = 'Community Area').head()
