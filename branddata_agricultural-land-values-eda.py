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
data = pd.read_csv('/kaggle/input/agricultural-land-values-19972017/Combined_Clean.csv')
data.dtypes
data.isnull().sum()
data.head()
import seaborn as sns

import matplotlib.pyplot as plt
states_land_values = pd.DataFrame(data.groupby(['State','Year'])['Acre Value'].sum())
states_land_values = states_land_values.reset_index()
states_land_values.head()
_ = plt.figure(figsize=(20,30))

_ = sns.lineplot(x="Year", y="Acre Value", hue="State", data=states_land_values)
land_values_1997 = states_land_values[states_land_values['Year'] == 1997][['State','Acre Value']]
land_values_1997.head()
land_values_1997_series = land_values_1997.set_index('State')['Acre Value']
land_values_2017 = states_land_values[states_land_values['Year'] == 2017][['State','Acre Value']]
land_values_2017.head()
land_values_2017_series = land_values_2017.set_index('State')['Acre Value']
land_price_increase = land_values_2017_series - land_values_1997_series
land_price_increase.sort_values(ascending = False).head()
region_data = data[data['Region or State'] == 'Region']
region_data.head()
region_acre_value = region_data.groupby(['Region'])['Acre Value'].sum()
region_acre_value.head().sort_values(ascending = False)
land_type_prices = data.groupby(['LandCategory'])['Acre Value'].describe()
land_type_prices