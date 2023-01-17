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
data = pd.read_csv('/kaggle/input/all-space-missions-from-1957/Space_Corrected.csv')
data.head()
data['Company Name'].value_counts().plot.bar(figsize=(10,5),title='Company Names')
data[' Rocket'] = pd.to_numeric(data[' Rocket'],errors='coerce')

data.dropna().groupby(['Company Name']).mean()[' Rocket'].plot.bar()

df = data.loc[data['Company Name'].isin(data['Company Name'].value_counts()[:11].index.values)]
df['Datum'] = pd.to_datetime(df['Datum'],utc=True).dt.date
df['Year'] = pd.to_datetime(df['Datum']).dt.year
# only using top 11 companies because any more would make it look to messy
df2 = df.groupby(['Year','Company Name'])['Year'].count().unstack('Company Name').fillna(0)
df2[df2.columns].plot(kind='bar', stacked=True,figsize=(10,5)).legend(loc='center left',bbox_to_anchor=(1.0, 0.5))


df2 = df.groupby(['Year','Status Mission'])['Year'].count().unstack('Status Mission').fillna(0)
df2[df2.columns].plot(kind='bar', stacked=True,figsize=(10,5)).legend(loc='center left',bbox_to_anchor=(1.0, 0.5))

df2 = df.groupby(['Company Name','Status Mission'])['Company Name'].count().unstack('Status Mission').fillna(0)
df2[df2.columns].plot(kind='bar',figsize=(10,5)).legend(loc='center left',bbox_to_anchor=(1.0, 0.5))

data.groupby(['Year']).mean()[' Rocket'].dropna().plot.bar(figsize=(10,5))


