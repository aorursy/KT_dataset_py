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
df = pd.read_csv("/kaggle/input/us-accidents/US_Accidents_Dec19.csv")
df.head()
df.tail()
df.sample(5)
df.shape
df.info()
#Lets check for missing values'
df.isna().sum()
((df.isna().sum())/df.shape[0])*100
#Lets start off with dropping End_Lat,End_Lng,Number,Wind chill,Precipitation(in)       

df.columns
df2 = df.drop(['End_Lat', 'End_Lng','Number','Wind_Chill(F)','Precipitation(in)'],axis =1

    )
df2.head()
#Now lets check their correlation
import seaborn as sns

import matplotlib.pyplot as plt


sns.set_palette('deep')
plt.figure(figsize = (16,12))

sns.heatmap(df2.corr(),cmap = 'Blues',annot = True,center = 0,linewidths=.5)
sns.countplot(df2['Source'])
plt.figure(figsize = (16,8))

df_pie = df2.groupby('Source').agg({'Source':'count'})

labels = df_pie.index

explode = (0, 0.1,0.1)

plt.pie(df_pie['Source'],labels = labels,explode =explode,autopct='%1.1f%%', shadow=True, startangle=140)

plt.show()
#Most of Data is from MapQuest actually followed by Bing and finally MapQuest-Bing.

df2.columns
df_1 = df2.groupby('State').agg({'Severity':'mean'})
plt.figure(figsize = (16,12))

sns.barplot(x  = df_1.Severity , y = df_1.index,orient = 'h')

plt.show()
#Seems like Tenesse,Wyomingand and Arizona on average get the most severe traffic due to accidents.
fig = plt.figure(figsize = (12,6))

fig.add_subplot(1, 2, 1)

sns.countplot(df2['Severity'])

fig.add_subplot(1, 2, 2)

df3 = df2['Severity'].value_counts()

plt.pie(df3,labels = df3.index)

plt.show()
#0 and 1 are almost negligible as compared to others
#This is a work in progress.Hopefully you Liked it.