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
df=pd.read_csv('../input/all-space-missions-from-1957/Space_Corrected.csv')

df
df.columns
df.dtypes
df.isnull().sum()
df[' Rocket'].fillna(0,inplace=True)
df['Company Name'].value_counts().head(11)
df['Location']=df['Location'].apply(lambda x: x.split(',')[-1])

df['Location']
df['Datum']=df['Datum'].apply(lambda x: x.split(',')[1])

l1=[]

for i in df['Datum']:

    l1.append(i[:5])

df['year']=l1
import matplotlib.pyplot as plt
plt.figure(figsize=(16,10))

plt.plot(df['Location'].value_counts().head(11))

plt.xlabel('Country Name',size=12)

plt.ylabel('Launch Counts',size=12)

plt.title('Top 11 Country with their Launches',size=15)
df['Status Mission'].value_counts().plot.bar(figsize=(16,10))

plt.xlabel('Mission Status',size=12)

plt.ylabel('Status Mission counts',size=12)

plt.title('Mission Status Distribution',size=15)
df.groupby('Location')[' Rocket'].count().plot.bar(figsize=(26,15))

plt.xlabel('Country',size=15)

plt.ylabel('No. of Rocketss',size=15)

plt.title('Rockets Distribution',size=20)