# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import seaborn as sns

import numpy as np

import matplotlib.pyplot as plt
uber = pd.read_csv("../input/uberdrives/My Uber Drives - 2016.csv")
#checking the Data size

uber.shape
#checking the content of the data

uber.head()
#checking the data types of the variable

uber.dtypes
uber.tail()
uber.drop(uber.tail(1).index, inplace=True)
uber.tail()
#converting the Datatype of date

uber['START_DATE*']=uber['START_DATE*'].astype('datetime64[ns]')

uber['END_DATE*']=uber['END_DATE*'].astype('datetime64[ns]')
#creating new variable called duration to calculate the duration and speed of the drive of the journey

uber['duration']= uber['END_DATE*']-uber['START_DATE*']

uber['duration']=uber['duration'].dt.total_seconds()/60.0

uber['duration']=uber['duration'].astype('float')

uber['speed']=uber['MILES*']/(uber['duration']/60)
uber
fig,ax=plt.subplots(1,3,figsize=(38,8))

a=uber.groupby(['PURPOSE*'])['duration'].mean().sort_values(ascending=True).plot(kind='barh', ax=ax[0])

a.title.set_text('Mean of duration for each Purpose')

b=uber.groupby(['PURPOSE*'])['duration'].count().sort_values(ascending=True).plot(kind='barh', ax=ax[1])

b.title.set_text('Number of drives per each Purpose')

c=uber.groupby(['PURPOSE*'])['MILES*'].sum().sort_values(ascending=True).plot(kind='barh', ax=ax[2])

c.title.set_text('Total Number of Miles driven per each Purpose')

tab_m=uber.groupby(['PURPOSE*']).mean()

print(tab_m)
uber['speed'].median()
plt.figure(figsize=(15,8))

sns.countplot(uber['PURPOSE*'], orient="v")

## Creating Columns for Month and Hour in a Day

uber['month']=uber['START_DATE*'].dt.month

uber['hour']=uber['START_DATE*'].dt.hour
fig,ax=plt.subplots(1,2,figsize=(38,8))

i=uber.groupby('month')['MILES*'].sum().sort_values(ascending=True).plot(kind='bar', ax=ax[0], color='orange')

j=uber.groupby('month')['MILES*'].mean().sort_values(ascending=True).plot(kind='bar', ax=ax[1], color='red')

i.title.set_text('Month vs Total Miles')

j.title.set_text('Month vs Average Miles/Month')

uber['START_DATE*'].dt.month.value_counts().sort_values(ascending=True).plot(kind='bar', color='green')

plt.title('Number of rides/Month')

plt.axhline(y=uber['START_DATE*'].dt.month.value_counts().mean(), xmin=0, xmax=3.5, linestyle="--")

uber['hour'].value_counts().sort_values(ascending=False).plot(kind='bar')

plt.axhline(y=uber['hour'].value_counts().mean(), xmin=0, xmax=3.5, linestyle="--")

plt.title('Hour in a day/Number of rides')
val=uber["MILES*"]

mil_cat=["<=5","5-10","10-15","15-20",">20"]

dic=dict()

for item in mil_cat:

    dic[item]=0

for i in val.values:

    if i<=5:

        dic["<=5"]+=1

    elif i<=10:

        dic["5-10"]+=1

    elif i<=15:

        dic["10-15"]+=1

    elif i<=20:

        dic["15-20"]+=1

    else:

        dic[">20"]+=1

final=pd.Series(dic)

final.sort_values(inplace=True,ascending=False)
sns.barplot(final.index, final.values)
