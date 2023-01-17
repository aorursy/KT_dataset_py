# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import seaborn as sns

import plotly.express as px

import geopandas as gpd





import folium

from folium import Choropleth, Circle, Marker

from folium.plugins import HeatMap, MarkerCluster



import matplotlib.pyplot as plt

import os

print(os.listdir("../input"))

%matplotlib inline



# Any results you write to the current directory are saved as output.
fire_file=pd.read_csv('../input/forest-fires-in-brazil/amazon.csv'

                     , encoding='latin1')

fire_file.head(5)
fire_file.size

fire_file['state'].describe()

fire_file['month'].unique()
fire_file['month'].replace(to_replace = 'Janeiro', value = 'Jan', inplace = True)

fire_file['month'].replace(to_replace = 'Fevereiro', value = 'Feb', inplace = True)

fire_file['month'].replace(to_replace = 'Março', value = 'Mar', inplace = True)

fire_file['month'].replace(to_replace = 'Abril', value = 'Apr', inplace = True)

fire_file['month'].replace(to_replace = 'Maio', value = 'May', inplace = True)

fire_file['month'].replace(to_replace = 'Junho', value = 'Jun', inplace = True)

fire_file['month'].replace(to_replace = 'Julho', value = 'Jul', inplace = True)

fire_file['month'].replace(to_replace = 'Agosto', value = 'Aug', inplace = True)

fire_file['month'].replace(to_replace = 'Setembro', value = 'Sep', inplace = True)

fire_file['month'].replace(to_replace = 'Outubro', value = 'Oct', inplace = True)

fire_file['month'].replace(to_replace = 'Novembro', value = 'Nov', inplace = True)

fire_file['month'].replace(to_replace = 'Dezembro', value = 'Dec', inplace = True)
year_month_state=fire_file.groupby(by=['year','state','month']).sum().reset_index()
print(year_month_state)
sns.set_style('whitegrid')

from matplotlib.pyplot import MaxNLocator, FuncFormatter



plt.figure(figsize=(12,4))

ax = sns.lineplot(x = 'year', y = 'number', data = year_month_state, estimator = 'sum', color = 'orange', lw = 3, 

                  err_style = None)

plt.title('Total Fires in Brazil : 1998 - 2017', fontsize = 18)

plt.xlabel('Year', fontsize = 14)

plt.ylabel('Number of Fires', fontsize = 14)



ax.xaxis.set_major_locator(plt.MaxNLocator(19))

ax.set_xlim(1998, 2017)

ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))

plt.figure(figsize=(12,4))



sns.boxplot(x = 'month', order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep','Oct', 'Nov', 'Dec'], 

            y = 'number', data = year_month_state)



plt.title('Fires in Brazil by Month', fontsize = 18)

plt.xlabel('Month', fontsize = 14)

plt.ylabel('Number of Fires', fontsize = 14)
fire_file['state'].unique()

print(fire_file[fire_file['state'] == 'Rio'].groupby(by = ['year','state', 'month']).sum().reset_index())
year_mo_state_Amazon = fire_file[fire_file['state'] == 'Pernambuco'].groupby(by = ['year','state', 'month']).sum().reset_index()





plt.figure(figsize=(12,4))



ax = sns.lineplot(x = 'year', y = 'number', data = year_mo_state_Amazon, estimator = 'sum', color = 'orange', lw = 3, 

                  err_style = None)



plt.title('Total Fires in Amazon : 1998 - 2017', fontsize = 18)

plt.xlabel('Year', fontsize = 14)

plt.ylabel('Number of Fires', fontsize = 14)



ax.xaxis.set_major_locator(plt.MaxNLocator(19))

ax.set_xlim(1998, 2017)



ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))