# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib as mlp

import matplotlib.pyplot as plt

import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/BlackFriday.csv")
df.head()
df.shape
df.info()
df['Gender'].value_counts()
df['Age'].value_counts()
df['City_Category'].value_counts()
df.dtypes
df.describe(include='all')
explode = (0.1,0)

fig1,ax1 = plt.subplots(figsize=(10,5))

plt.rcParams['font.size']=10

color_palette_list = ['#80bfff', '#ff99ff' ]

ax1.pie(df['Gender'].value_counts(),explode=explode,labels=['Male','Female'],colors=color_palette_list[0:2],autopct = "%1.1f%%",shadow=True,startangle=90)

ax1.axis('equal')

plt.tight_layout()

plt.legend()

plt.title('Percentage of Transactions by genre')

plt.show()
df2 = df
df3 = df2.drop_duplicates(['User_ID'],keep='first')

df3.shape
df.shape
df3['Gender'].value_counts()
explode = (0.1,0)

fig1,ax1 = plt.subplots(figsize=(8,5))

plt.rcParams['font.size']=18

ax1.pie(df3['Gender'].value_counts(),explode=explode,shadow=True,labels=['Male','Female'],autopct="%1.1f%%",startangle=90)

ax1.axis('equal')

plt.legend()

plt.tight_layout()

plt.title('Male and Female')

plt.show()

labels = df3['Age'].value_counts().index

explode = (0.1,0)

fig1,ax1 = plt.subplots(figsize=(10,8))

plt.rcParams['font.size'] = 18

ax1.pie(df3['Age'].value_counts(),labels=labels,shadow=True,autopct = "%1.0f%%",startangle=0,pctdistance=0.6,labeldistance=1.1)

ax1.axis('equal')

plt.legend(label,

          title="Age Groups",

          loc="center left",

          bbox_to_anchor=(1.2, 0.5, 0, 0.2))

plt.title("Percentage of age Groups")

plt.tight_layout()

plt.show()
n_bins = 30

plt.subplots(figsize=(16,8))

plt.hist(df['Purchase'],bins=n_bins)

plt.xlabel("Purchase")

plt.ylabel("Count")

plt.title("Purchase Histogram")

plt.show()
df.groupby('Age')[['Purchase']].describe()
data = df.groupby('Age')[['Purchase']].mean()
data
data.plot(xticks=[1,2,3,4,5,6,7],figsize=(16,8))

plt.title("Mean Purchase by Age")

plt.show()
derived = df.groupby('Product_ID')[['Purchase']].sum()/ sum(df['Purchase'])
derived
df3['Gender'].value_counts()
df.groupby(['Product_ID','Gender'])[['Gender']].count().head()
plt.subplots(figsize=(16,8))

df.groupby(['Occupation'])['User_ID'].nunique().sort_values().plot('bar')

plt.title("User by Occupation")

plt.xlabel('Occupation')

plt.ylabel('Count')

plt.show()
plt.subplots(figsize=(16,8))

df.groupby(['City_Category'])['User_ID'].nunique().sort_values().plot('bar')

plt.title('Users by Locations')

plt.xlabel('City')

plt.ylabel('Count')

plt.show()

plt.subplots(figsize=(10,5))

df.groupby('Gender')['Purchase'].sum().sort_values().plot('bar')

plt.title('Purchase by Genre Not Normalized')

plt.xlabel('Gender')

plt.ylabel('Number of Purchases')

plt.show()
plt.subplots(figsize=(10,5))

(df.groupby('Gender')['Purchase'].sum()/df.groupby(['Gender'])['User_ID'].nunique()).plot('bar')

plt.title('Purchases by Gender - Normalized')

plt.xlabel('Gender')

plt.ylabel('Purchases')

plt.show()
plt.subplots(figsize=(10,5))

df.groupby('Age')['Purchase'].sum().plot('bar')

plt.title('Purchases by age')

plt.xlabel('Age')

plt.ylabel('Purchases')

plt.show()
plt.subplots(figsize=(10,5))

(df.groupby('Age')['Purchase'].sum()/df.groupby(['Age'])['User_ID'].nunique()).plot('bar')

plt.title('Purchases by Age - Normalized')

plt.xlabel('Age')

plt.ylabel('Purchases')

plt.show()