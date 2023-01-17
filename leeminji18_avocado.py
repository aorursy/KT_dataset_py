# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
pd.plotting.register_matplotlib_converters()

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
avocado_data = pd.read_csv('../input/avocado-prices/avocado.csv')
avocado_data
avocado_data.info()
avocado_data.describe()
pd.Series.unique(avocado_data['year'])

sns.barplot(data = avocado_data, x = 'year', y = 'Total Volume')
sns.barplot(data = avocado_data, x = 'year', y = 'AveragePrice')
plt.figure(figsize = (20,6))
chart = sns.barplot(data = avocado_data, x = 'region', y = 'Total Volume')
chart.set_xticklabels(chart.get_xticklabels(), rotation=45,horizontalalignment='right')
plt.figure(figsize = (20,6))
chart = sns.barplot(data = avocado_data, x = 'region', y = 'AveragePrice')
chart.set_xticklabels(chart.get_xticklabels(), rotation=45,horizontalalignment='right')
plt.figure(figsize = (10,10))
sns.lmplot(data = avocado_data , x = 'AveragePrice', y = 'Total Volume')
#pd.Series.unique(avocado_data['region'])
albany = avocado_data.loc[avocado_data.region == 'Albany']
california = avocado_data.loc[avocado_data.region == 'California']
sanfrancisco = avocado_data.loc[avocado_data.region == 'SanFrancisco']

plt.figure(figsize = (14,6))
chart = sns.lineplot(data = albany, x = 'Date', y = 'Total Volume', color = 'green')
sns.lineplot(data = california, x = 'Date', y = 'Total Volume', color = 'yellow')###
sns.lineplot(data = sanfrancisco, x = 'Date', y = 'Total Volume')

for ind, label in enumerate(chart.get_xticklabels()):
    if ind % 30 == 0:  # every 10th label is kept
        label.set_visible(True)
    else:
        label.set_visible(False)


#avocado_data.groupby('region').

plt.figure(figsize = (10,6))
chart = sns.barplot(data = avocado_data, x = 'type', y = 'AveragePrice')
chart.set_xticklabels(chart.get_xticklabels(), rotation=45,horizontalalignment='right')
sns.barplot(data = avocado_data, x = 'year', y = 'Total Volume', hue = 'type')
plt.figure(figsize = (14,6))
chart = sns.lineplot(data = avocado_data, x = 'Date', y = '4046', color = 'red')
sns.lineplot(data = avocado_data, x = 'Date', y = '4225', color = 'blue')
sns.lineplot(data = avocado_data, x = 'Date', y = '4770', color = 'green')##가장 작음

for ind, label in enumerate(chart.get_xticklabels()):
    if ind % 30 == 0:  # every 10th label is kept
        label.set_visible(True)
    else:
        label.set_visible(False)
plt.figure(figsize = (14,6))
chart = sns.lineplot(data = avocado_data, x = 'Date', y = 'Total Bags')
sns.lineplot(data = avocado_data, x = 'Date', y = 'Small Bags', color = 'purple')
sns.lineplot(data = avocado_data, x = 'Date', y = 'Large Bags', color = 'green')
sns.lineplot(data = avocado_data, x = 'Date', y = 'XLarge Bags', color = 'red')

for ind, label in enumerate(chart.get_xticklabels()):
    if ind % 30 == 0:  # every 10th label is kept
        label.set_visible(True)
    else:
        label.set_visible(False)