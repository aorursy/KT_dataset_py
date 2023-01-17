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
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import matplotlib.patches as mpatches

%matplotlib inline 

import seaborn as sns

import plotly.io as pio

import plotly.graph_objects as go

from plotly.offline import init_notebook_mode, iplot
df = pd.read_csv('../input/malaria-dataset/reported_numbers.csv')

df.head()
df.info()
df['Country'].value_counts()
#Some countries are incorrectly classified into regions. Created a new column based on accurate regions

region_dictionary ={'Africa' : 'Africa', 'Americas' : 'Americas', 'Eastern Mediterranean' : 'Eastern Mediterranean', 'Europe' : 'Central Asia', 'South-East Asia':'Asia', 'Western Pacific':'Asia'}

region_dictionary

df['New Region'] = df['WHO Region'].map(region_dictionary)
fig=plt.figure(figsize=(12,6))

plt.subplots_adjust(left=None, bottom=None, right=None, top=1.5, wspace=None, hspace=None)

plt.subplot(211)

sns.lineplot(x='Year',y='No. of cases', data=df.groupby('Year')['No. of cases'].sum().reset_index())

plt.title('No of cases over the years')

plt.xticks(np.arange(2000, 2018, 1))



plt.subplot(212)

sns.barplot(x='Year',y='No. of cases', data=df.groupby('Year')['No. of cases'].sum().reset_index(), palette='RdBu_r')

plt.title('No of cases over the years')
df1=df.dropna()

df1.head()
fig=plt.figure(figsize=(12,6))

plt.subplots_adjust(left=None, bottom=None, right=None, top=1.5, wspace=None, hspace=None)

plt.subplot(211)

sns.lineplot(x='Year',y='No. of deaths', data=df1.groupby('Year')['No. of deaths'].sum().reset_index(), color='red')

plt.title('No of malaria deaths over the years')

plt.xticks(np.arange(2000, 2018, 1))



plt.subplot(212)

sns.barplot(x='Year',y='No. of deaths', data=df1.groupby('Year')['No. of deaths'].sum().reset_index(), palette='coolwarm')

plt.title('No of malaria deaths over the years')
fig=plt.figure(figsize=(12,6))

plt.subplots_adjust(left=None, bottom=None, right=None, top=2, wspace=None, hspace=None)

plt.subplot(211)

sns.barplot(x='New Region',y='No. of cases', data=df.groupby('New Region')['No. of cases'].sum().reset_index(), palette='coolwarm')

plt.title('No of malaria cases across region')



plt.subplot(212)

sns.barplot(x='New Region',y='No. of deaths', data=df1.groupby('New Region')['No. of deaths'].sum().reset_index(), palette='coolwarm')

plt.title('No of malaria deaths across region')
df2 = df[(df['New Region']=='Asia') | (df['New Region']=='Eastern Mediterranean')]

x=['India', 'Pakistan', 'Bangladesh','Nepal', 'Sri Lanka','Myanmar']

df3 = df2[df2['Country'].isin(x)]

df3.head()
fig=plt.figure(figsize=(12,6))

plt.subplots_adjust(left=None, bottom=None, right=None, top=2, wspace=None, hspace=None)

plt.subplot(211)

plt.xticks(np.arange(2000, 2018, 1))

sns.lineplot(x='Year',y='No. of cases', hue='Country', data=df3.groupby(['Year','Country'])['No. of cases'].sum().reset_index())

plt.title('No of malaria cases in South Asia')



plt.subplot(212)

plt.xticks(np.arange(2000, 2018, 1))

sns.lineplot(x='Year',y='No. of deaths', hue='Country', data=df3.groupby(['Year','Country'])['No. of deaths'].sum().reset_index())

plt.title('No of malaria deaths in South Asia')
df4 = df.groupby(['New Region','Country'])['No. of cases'].sum().reset_index()

africa_region = df4[df4['New Region'] == 'Africa']

africa_region.head()
df5 = df1.groupby(['New Region','Country'])['No. of deaths'].sum().reset_index()

df5 = df5[df5['New Region'] == 'Africa']

df5.head()
africa_region = pd.merge(africa_region, df5[['Country','No. of deaths']], on='Country')

africa_region.head()
fig=plt.figure(figsize=(12,8))

plt.subplots_adjust(left=None, bottom=None, right=None, top=2, wspace=None, hspace=None)

plt.subplot(211)

sns.barplot(x='No. of cases', y='Country', data=africa_region.sort_values(by='No. of cases',ascending=False), palette='GnBu_d')

plt.title('Countries with high number of malaria cases in Africa')



plt.subplot(212)

sns.barplot(x='No. of deaths', y='Country', data=africa_region.sort_values(by='No. of deaths',ascending=False), palette='GnBu_d')

plt.title('Countries with high number of malaria deaths in Africa')