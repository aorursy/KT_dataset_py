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
import sys, math

import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline
data = pd.read_csv('/kaggle/input/daily-temperature-of-major-cities/city_temperature.csv')

print (data.shape)

data.head()
data['Region'].value_counts()
data['Year'].value_counts()
# Year 200 and 201 are appearing in the data and are observed as typos.

# Year 2020 is dropped because, incomplete data cannot be used for measuring average temperatures.



data = data[(data['Year'] != 200) & (data['Year'] != 201) & (data['Year'] != 2020)]
data = data[~(data['AvgTemperature'] == -99.0)]
data.head()
yearly_region_data = data.groupby(['Region', 'Year']).mean().reset_index().drop(columns = ['Month', 'Day'])
yearly_region_data.head()
yearly_region_data[(yearly_region_data['Year'] == 1995) | (yearly_region_data['Year'] == 2019)]
plt.subplots(figsize=(15, 6))

sns.lineplot(x = 'Year', y = 'AvgTemperature', hue = 'Region', data = yearly_region_data)

plt.xticks(rotation = 90)

plt.legend()

plt.title('Temperature changes across regions from 1995-2020.')

plt.show()
yearly_city_data = data.groupby(['City', 'Year']).mean().reset_index().drop(columns = ['Month', 'Day'])
temp_diff = yearly_city_data.groupby(['City']).agg([('Min', 'min'), ('Max', 'max')]).reset_index().drop(columns = ['Year'])
temp_diff.columns = temp_diff.columns.map(''.join) 
temp_diff['temp_diff'] = temp_diff['AvgTemperatureMin']-temp_diff['AvgTemperatureMax']
temp_diff[(temp_diff['temp_diff'] > 10) | (temp_diff['temp_diff'] < -10)]