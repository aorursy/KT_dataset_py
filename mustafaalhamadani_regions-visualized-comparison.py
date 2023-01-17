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
import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv('../input/corona-virus-report/country_wise_latest.csv')
data
data.info()
data.isna().sum()
plt.style.use('seaborn')
fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, figsize=(20, 20))




plot = ax1.bar(data['WHO Region'], data['Confirmed'])
ax1.set(xlabel='Regions', ylabel='Confirmed Cases (in millions)', title='Total Confirmed cases in the Regions')
font = {'weight' : 'bold',
        'size'   : 20}
plt.rc('font', **font)


plot2 = ax2.bar(data['WHO Region'], data['Deaths'])
ax2.set(xlabel='Regions', ylabel='Death Cases',title='Total Confirmed Deaths in the Regions' )


plot3 = ax3.bar(data['WHO Region'], data['Recovered']);
ax3.set(xlabel= 'Regions', ylabel='Recovered cases (in millions)', title='Total recovered cases in the regions')
fig.suptitle('COVID-19 Regions comparison (2020/08)');
fig, ax = plt.subplots(figsize=(10,10))
plot = ax.bar(data['WHO Region'], data['Confirmed'])
ax.set(xlabel='Regions', ylabel='Confirmed Cases (in millions)', title='COVID-19 Confirmed Cases in the Regions')
font = {'weight' : 'bold',
        'size'   : 20}
plt.rc('font', **font)
ig, ax = plt.subplots(figsize=(10,10))
plot = ax.bar(data['WHO Region'], data['Deaths'])
ax.set(xlabel='Regions', ylabel='Death Cases',title='Covid-19 Confirmed Death Cases in the regions' );
fig, ax = plt.subplots(figsize=(10,10))
plot = ax.bar(data['WHO Region'], data['Recovered']);
ax.set(xlabel= 'Regions', ylabel='Recovered cases (in millions)', title='COVID-19 Confirmed Recovered cases in the Regions');
fig, ax = plt.subplots(figsize=(10,10))
plt.barh(data['WHO Region'],data['Deaths / 100 Cases'])
ax.set(xlabel='Deaths per 100 cases', ylabel='Regions',title='Covid-19 Deaths per 100 cases in the Regions' );
fig, ax = plt.subplots(figsize=(15,10))
plt.barh(data['WHO Region'],data['Recovered / 100 Cases'])
ax.set(xlabel='Recovered per 100 cases', ylabel='Regions',title='Covid-19 Recoveries per 100 cases in the Regions' );
