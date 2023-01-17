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
data = pd.read_csv('/kaggle/input/daily-temperature-of-major-cities/city_temperature.csv')

data.head()
data.info()
data.describe()
cardinality = {}

for col in data:
    cardinality[col] = data[col].nunique()

cardinality
missing = data.isna().sum() * 100 / data.shape[0]

missing
rorder = data.Region.unique()

rorder
data.Day.unique()
zeroday = data[data.Day == 0]

zeroday
data = data[data.Day != 0]
morder = data.Month.unique()

morder
yorder = data.Year.unique()

yorder
wrongyear = data[(data.Year <= 201)]

wrongyear.describe()
data.Year = data.Year.replace(200, 2000).replace(201, 2001)
data = data[data.AvgTemperature != -99]
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
region_ymin = data.groupby(['Region', 'Year'])['AvgTemperature'].min().reset_index()
region_ymax = data.groupby(['Region', 'Year'])['AvgTemperature'].max().reset_index()

region_year = data.groupby(['Region', 'Year'])['AvgTemperature'].mean().reset_index()
region_year['MinTemperature'] = region_ymin.AvgTemperature
region_year['MaxTemperature'] = region_ymax.AvgTemperature

region_year.head()
fig = px.bar(region_year, y = 'Region', x = 'AvgTemperature', color = 'Region', animation_frame = 'Year', hover_name = 'AvgTemperature', range_x = [0, 100],
             labels = {'AvgTemperature': 'Average Temperature'}, title = 'Annual Global Temperature Fluctuations')

fig.show()
sns.set_style('whitegrid')

sns.relplot(x = 'Year', y = 'AvgTemperature', hue = 'Region', data = region_year, hue_order = rorder, kind = 'line', markers = "<", height = 6, aspect = 2, palette = 'colorblind')
plt.xlabel('Year')
plt.ylabel('Average Temperature')
plt.title('Averagre Temperatures by Region per Year')
plt.show()

fig = plt.figure(figsize = (20, 30))
sns.set_style('darkgrid')

for i in range(1, 8):
    ax = fig.add_subplot(7, 1, i)
    ax = sns.boxplot(data = data[data.Region == rorder[i - 1]], x = 'Year', y = 'AvgTemperature')
    ax.set_ylim(-50, 120)
    if i == 7:
        plt.xlabel('Year')
    else:
        plt.xlabel('')
    plt.ylabel(rorder[i - 1])
plt.show()
fig = plt.figure(figsize = (30, 15))
sns.set_style('darkgrid')
sns.set_context("paper")


for i in range(1, 8):
    ax = fig.add_subplot(1, 7, i)
    ax = sns.lineplot(data = region_year[region_year.Region == rorder[i - 1]], x = 'Year', y = 'MinTemperature')
    ax = sns.lineplot(data = region_year[region_year.Region == rorder[i - 1]], x = 'Year', y = 'AvgTemperature')
    ax = sns.lineplot(data = region_year[region_year.Region == rorder[i - 1]], x = 'Year', y = 'MaxTemperature')
    ax.set_ylim(-55, 120)
    plt.xlabel(rorder[i - 1])
    if i == 1:
        plt.ylabel('Temperature')
    else:
        plt.ylabel('')
    if i == 4:
        plt.title('Distribution of Minimum, Average and Maximum Temperatures observed annualy by Region')    
plt.show()        