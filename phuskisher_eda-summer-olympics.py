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

df=pd.read_csv('/kaggle/input/summer-olympics-medals/Summer-Olympic-medals-1976-to-2008.csv', encoding='latin-1')

df.head()
df.info()
for vars in df.columns:

  rows=df.shape[0]

  cols=df.shape[1]

print(f'rows : {rows}')

print(f'columns : {cols}')
for vars in df.columns:

  print(f'no. of unique values in {vars} : {df[vars].nunique()}\n')

import matplotlib.pyplot as plt

import seaborn as sns

year_wise_total_medal=df.groupby('Year')['Medal'].count()

year_wise_total_medal
year_wise_total_medal.plot(kind='bar')
year_wise_medal=df.groupby('Year')['Medal'].value_counts()

year_wise_medal
plt.figure(figsize=(15,6))

sns.countplot(df['Year'], hue=df['Medal'])
year_wise_participants=df.groupby('Year')['Gender'].value_counts()

year_wise_participants
plt.figure(figsize=(15,6))

sns.countplot(df['Year'], hue=df['Gender'])
medals_by_country=df.groupby('Country')['Medal'].count().sort_values(ascending=False)[:20]

medals_by_country
medals_by_country.plot(kind='bar', figsize=(20,6))
medals_in_sports=df.groupby('Sport')['Medal'].count().sort_values(ascending=False)

medals_in_sports
medals_in_sports.plot(kind='bar', figsize=(20,6))
usa=df[df['Country']=='United States']

usa
usa_medal=usa.groupby('Year')['Medal'].count()

usa_medal
usa_medal.plot(kind='bar')
usa_medal_GSB=usa.groupby('Year')['Medal'].value_counts()

usa_medal_GSB
plt.figure(figsize=(15,6))

sns.countplot(usa['Year'], hue=usa['Medal'])
usa_participants_gender=usa.groupby('Year')['Gender'].value_counts()

usa_participants_gender
sns.countplot(usa['Year'],hue=usa['Gender'])
sports_medals=usa.groupby('Sport')['Medal'].value_counts()

sports_medals
plt.figure(figsize=(20,6))

sns.countplot(usa['Sport'], hue=usa['Medal'])

plt.xticks(rotation=90)
sports_gender=usa.groupby('Sport')['Gender'].value_counts()

sports_gender
plt.figure(figsize=(20,6))

sns.countplot(usa['Sport'], hue=usa['Gender'])

plt.xticks(rotation=90)
usa_discipline_medals=usa.groupby('Discipline')['Medal'].value_counts()

usa_discipline_medals
plt.figure(figsize=(25,6))

sns.countplot(usa['Discipline'], hue=usa['Medal'])

plt.xticks(rotation=90)
usa_discipline_gender=usa.groupby('Discipline')['Gender'].value_counts()

usa_discipline_gender
plt.figure(figsize=(25,6))

sns.countplot(usa['Discipline'], hue=usa['Gender'])

plt.xticks(rotation=90)
India=df[df['Country']=='India']

India
india_total_medals=India.groupby('Year')['Medal'].count()

india_total_medals
india_total_medals.plot(kind='bar')
sns.countplot(India['Year'], hue=India['Medal'])
sns.countplot(India['Year'], hue=India['Gender'])
sns.countplot(India['Sport'], hue=India['Gender'])
sns.countplot(India['Sport'], hue=India['Medal'])