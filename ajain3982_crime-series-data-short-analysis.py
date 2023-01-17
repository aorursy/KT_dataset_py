import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

from matplotlib import pyplot as plt

%matplotlib inline
crime = pd.read_csv('../input/Crime1.csv')

crime.head()
counts = crime.Category.value_counts()

df = pd.DataFrame(counts)

df['name'] = df.index

df.columns = ['Value','name']

df.head()
sns.factorplot(x='Value',y='name',data=df,kind='bar',size=4.25, aspect=1.9)
counts = crime.Descript.value_counts()[:10]

df = pd.DataFrame(counts)

df['name'] = df.index

df.columns = ['Value','name']

df.head()
sns.factorplot(x='Value',y='name',data=df,kind='bar',size=4.25, aspect=1.9)
sns.countplot(crime['DayOfWeek'])
counts = crime.PdDistrict.value_counts()

df = pd.DataFrame(counts)

df['name'] = df.index

df.columns = ['Value','name']

df.head()
sns.factorplot(x='name',y='Value',data=df,kind='bar',size=4.25, aspect=1.9)

plt.xticks(rotation=30)
counts = crime.Resolution.value_counts()

df = pd.DataFrame(counts)

df['name'] = df.index

df.columns = ['Value','name']

df.head()
sns.factorplot(x='name',y='Value',data=df,kind='bar',size=4.25, aspect=1.9)

plt.xticks(rotation=30)
from datetime import datetime

def processDate():

    for index,row in crime.iterrows():

        date = datetime.strptime(row['Dates'], '%m/%d/%Y %H:%M')

        crime.loc[index,'hour'] = date.hour

        crime.loc[index,'day'] = date.day
processDate()
crime.head()
plt.figure(figsize=(10,6))

sns.distplot(crime['hour'].values,kde=False,rug=False,bins=24,color='green')
plt.figure(figsize=(10,6))

sns.distplot(crime['day'].values,kde=False,rug=False,bins=24,color='blue')