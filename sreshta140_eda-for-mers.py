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
import numpy as np
import seaborn as sns
country_count_latest=pd.read_csv('/kaggle/input/mers-outbreak-dataset-20122019/country_count_latest.csv')
weekly_clean=pd.read_csv('/kaggle/input/mers-outbreak-dataset-20122019/weekly_clean.csv')
weekly_clean
country_count_latest
country_count_latest=country_count_latest.sort_values(by='Confirmed', ascending=False)
country_count_latest=country_count_latest.reset_index()
country_count_latest
country_count_latest['Confirmed'].sum()
fig = plt.figure(figsize=(20,10))

ax = fig.add_axes([0,0,1,1])
ax.bar(country_count_latest['Country'], country_count_latest['Confirmed'])
plt.xlabel('Country')
plt.ylabel('Count')
plt.show()
fig = plt.figure(figsize=(20,10))

ax = fig.add_axes([0,0,1,1])
ax.bar(country_count_latest['Country'].head(5), country_count_latest['Confirmed'].head(5))
plt.xlabel('Country')
plt.ylabel('Count')
plt.show()
fig = plt.figure(figsize=(30,30))
ax = fig.add_axes([0,0,1,1])
ax.axis('equal')
langs = ['C', 'C++', 'Java', 'Python', 'PHP']
students = [23,17,35,29,12]
ax.pie(country_count_latest['Confirmed'], labels = country_count_latest['Country'],autopct='%1.2f%%')
plt.show()
weekly_clean['New Cases'].sum()
weekly_clean.groupby(['Year', 'Week'])['New Cases'].sum()
weekly_clean.groupby(['Region'])['New Cases'].sum()
weekly_cum_overall=pd.DataFrame(weekly_clean.groupby(['Year', 'Week'])['New Cases'].sum())
weekly_cum_overall=weekly_cum_overall.reset_index()
weekly_cum_overall
weekly_cum_overall['New Cases'].plot(figsize=(20,10))
plt.xlabel('Time')
plt.ylabel('New Cases')
regions=weekly_clean['Region'].unique()
for i in regions:
    print(i)
    weekly_clean[weekly_clean['Region']==i]['New Cases'].plot(figsize=(20,10))
    plt.xlabel('Time')
    plt.ylabel('New Cases')
    plt.show()
#overlapped
for i in regions:
    weekly_clean[weekly_clean['Region']==i]['New Cases'].plot(figsize=(20,10))
plt.xlabel('Time')
plt.ylabel('New Cases')
plt.show()