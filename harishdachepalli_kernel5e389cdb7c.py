# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # plotting

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input/dataset-ardd'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
df_fatalities_data = pd.read_csv('/kaggle/input/ardd_fatalities.csv', delimiter=',')
df_fatal_data = pd.read_csv('/kaggle/input/ardd_fatal_crashes.csv', delimiter=',')
df_fatalities_data.dataframeName = 'ardd_fatalities.csv'
df_fatal_data.dataframeName = 'ardd_fatal_crashes.csv'
df_fatal_data.rename(columns={'Bus \nInvolvement': 'Bus Involvement'}, inplace=True)
df_fatal_data.head(3)


# remove rows with missing value in speed limit
df_fatal_data = df_fatal_data[df_fatal_data['Speed Limit'] != '-9']


# In[6]:


# Dataset filtered with Year > 2015
df_fatal_data_filtered = df_fatal_data[df_fatal_data['Year']>2015]


# In[7]:


# df_fatal_data_filtered = df_fatal_data[df_fatal_data['Year']>2010]
piv_deaths = pd.pivot_table(data=df_fatal_data_filtered, values='Number Fatalities', index=['Year'], columns=['Crash Type'], aggfunc='sum')
piv_deaths.head(3)
# piv_deaths = piv_deaths[piv_deaths['Year']==2005]


# In[61]:


# df_fatal_data_filtered = df_fatal_data[df_fatal_data['Year']>2010]
piv_fatality = pd.pivot_table(data=df_fatal_data_filtered, values='Number Fatalities', index=['Year'], columns=['Crash Type'], aggfunc='sum')
piv_fatality['Total Fatality'] = piv_fatality['Multiple'] + piv_fatality['Pedestrian'] + piv_fatality['Single']
piv_fatality['Multiple'] = round(100 *piv_fatality['Multiple']/piv_fatality['Total Fatality'],2)
piv_fatality['Pedestrian'] = round(100 *piv_fatality['Pedestrian']/piv_fatality['Total Fatality'],2)
piv_fatality['Single'] = round(100 *piv_fatality['Single']/piv_fatality['Total Fatality'],2)
mask = ['Multiple', 'Pedestrian', 'Single']
piv_fatality = piv_fatality[mask]
ax = piv_fatality.plot(
    kind='bar', 
    stacked=False, 
    color=('#5cb85c', '#5bc0de', '#d9534f'),
    width = 0.8, 
    figsize=(20, 8), 
    legend=True,
    fontsize=14)

ax.set_title("Fatalities over years", fontsize=16)
ax.legend(loc='upper right', frameon=True, fontsize=14)
ax.set_ylabel("Fatalities %", fontsize=14)
ax.set_xlabel("Year", fontsize=14)

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
