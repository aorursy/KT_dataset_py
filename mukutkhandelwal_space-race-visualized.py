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

import seaborn as sns

sns.set_style(style = 'whitegrid')
df = pd.read_csv('../input/all-space-missions-from-1957/Space_Corrected.csv')
df.head()
sns.heatmap(df.isnull())
df.isnull().sum()
df.drop(columns=[' Rocket','Unnamed: 0','Unnamed: 0.1'],axis = 'columns',inplace=True)
def county(x):

  a = x.split(',')

  return a[-1]

df['Country'] = df['Location'].apply(county)
def years(x):

  x = pd.to_datetime(x)

  return x.year

df['Year']= df['Datum'].apply(years)
df.head(3)
total_company = df.groupby('Company Name')['Country']

comp = []

comp_count = []

for i in total_company:

  comp.append(i[0])

  comp_count.append(len(i[1]))

df_comp_count = pd.DataFrame({'Company':comp,'Total Company': comp_count})
df_comp_count[:10]
plt.figure(figsize = (12,15))

df_comp_count = df_comp_count.sort_values('Total Company',ascending =False)

sns.barplot(x= 'Total Company',y = 'Company',data = df_comp_count )

plt.title('Total number of Mission Organization wise since 1957')
df_1 = df[df['Status Mission'] == 'Success']

plt.figure(figsize=(12,15))

sns.countplot(x = 'Country',hue= 'Status Mission',data= df_1,order = df_1['Country'].value_counts().index)

plt.xticks(rotation= 90)

plt.title('Total Successful Mission Country wise')
df_2 = df[df['Status Mission'] == 'Failure']

plt.figure(figsize=(12,15))

sns.countplot(y = 'Country',hue= 'Status Mission',data= df_2,order=df_2['Country'].value_counts().index)

plt.title('Total Mission Failed')

plt.xticks(rotation= 90)
plt.figure(figsize=(12,18))

sorted_year = df.sort_values('Year',ascending = False)

sns.countplot(y = 'Year',data = sorted_year,order = df['Year'].value_counts().index)

plt.title('Total number of Space Missions Year wise')
plt.figure(figsize = (8,18))

sns.countplot(y = 'Company Name',hue = 'Status Rocket',data = df,order = df['Company Name'].value_counts().index)

plt.legend()

plt.title('Status of rocket according to company ')