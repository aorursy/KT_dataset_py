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
# first we import all the important library

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.graph_objects as go

import plotly.express as px

import datetime

from plotly.offline import init_notebook_mode, iplot

from plotly.subplots import make_subplots
# read the csv

df=pd.read_csv("/kaggle/input/windows-store/msft.csv")
df.head(20)
df.shape
df.describe()
df.info()
df.isna().sum()
df.dropna(axis=0,inplace=True)
df.isna().sum()
df.shape
sns.countplot(data=df,

                  x= 'Rating',

        

                  )

plt.title('Count of different rating',size = 20)
print( len(df['Category'].unique()) , "categories")# we print the unique category



print("\n", df['Category'].unique())
print( len(df['Rating'].unique()) , "Rating")



print("\n", df['Rating'].unique())
g = sns.countplot(x="Category",data=df, palette = "Set1")

g.set_xticklabels(g.get_xticklabels(), rotation=90, ha="right")

g 

plt.title('Count of app in each category',size = 20)
g = sns.catplot(x="Category",y="Rating",data=df, kind="box", height = 10 ,

palette = "Set1")

g.set_xticklabels(rotation=90)

g = g.set_ylabels("Rating")

plt.title('Boxplot of Rating VS Category',size = 20)
df['No of people Rated'].head()
df['Reviews']=df['No of people Rated'] # we make a copy of column 'No of people Rated' to 'Review'
df.drop('No of people Rated', axis=1, inplace=True) # we delete the column 'No of people Rated'
df


g = sns.kdeplot(df.Reviews, color="Green", shade = True)

g.set_xlabel("Reviews")

g.set_ylabel("Frequency")

plt.title('Distribution of Reveiw',size = 20)
df[df.Reviews > 950].head(10)
plt.figure(figsize = (10,10))

g = sns.regplot(x="Reviews", y="Rating",color = 'green', data=df);

plt.title('Rating VS Reveiws',size = 20)
plt.figure(figsize = (10,10))

g = sns.countplot(x="Price",data=df, palette = "Set1")

g.set_xticklabels(g.get_xticklabels(), rotation=90, ha="right")

g 

plt.title('Count of app in each category',size = 20)
df['Type']=df['Price']

for row in range(df.shape[0]):

    if df.loc[row, 'Type'] != 'Free':

        df.loc[row, 'Type'] = 'Paid'

df

plt.figure(figsize = (5,5))

sns.countplot(data=df,

                  x= 'Type',)
df['Price'].value_counts().head(30)
import random



def generate_color():

    color = '#{:02x}{:02x}{:02x}'.format(*map(lambda x: random.randint(0, 255), range(3)))

    return color
flatui = []

for i in range(0,len(df['Category'].unique()),1):

    flatui.append(generate_color())
g = sns.catplot(x="Price", y="Rating", hue="Category", kind="swarm", data=df,palette = flatui,size = 10)

g.set_xticklabels(rotation=90)

plt.title('Category in each Price VS Rating',size = 20)
plt.figure(figsize=(12,7))

df2 = df.groupby(['Category'])['Reviews'].sum().sort_values(ascending=False).reset_index()

g = sns.barplot(x='Category', y='Reviews', data=df2, palette="cubehelix")

g.set_xticklabels(g.get_xticklabels(), rotation=90)

g.set_title('Total number Reviews by Category', fontsize=20)

plt.show()
plt.figure(figsize=(12,7))

df3 = df.groupby(['Category'])['Rating'].mean().sort_values(ascending=False).reset_index()

g = sns.barplot(x='Category', y='Rating', data=df3, palette="cubehelix")

g.set_xticklabels(g.get_xticklabels(), rotation=90)

g.set_title('Average rating by Category', fontsize=20)

plt.show()
df['Year'] = pd.DatetimeIndex(df['Date']).year

df_year = df.groupby(['Year', 'Category'])['Reviews'].sum()

df_year = pd.DataFrame(df_year.reset_index())

df_year
plt.figure(figsize=(16,9))

g = sns.barplot(x='Year', y='Reviews', data = df_year, palette='Set2')

g.set_title('Total number of Reviews by Year ', fontsize=20)

plt.show()

plt.figure(figsize=(20,9))

g = sns.barplot(x='Year', y='Reviews', data = df_year, hue='Category')

g.set_title('Total number of Reviews by Year by Category ', fontsize=20)

g.legend(loc='upper left')

plt.show()
df_paid=df[df['Type']=='Paid']

df_paid['Price']
sns.countplot(data=df_paid,

                  x= 'Rating',)

plt.title('Count of different rating of paid app',size = 20)
df_free=df[df['Type']=='Free']

df_free
sns.countplot(data=df_free,

                  x= 'Rating',)

plt.title('Count of different rating of free app',size = 20)
df_paid['Price'] = df_paid['Price'].str.split(expand=True)[1]#we removed the rupee symbol from price
df_paid['Price'] = df_paid['Price'].str.replace(',','') # we removed the ',' in price.

df_paid['Price'] = df_paid['Price'].astype(float) #and convert it to float.
df_top_paid = df_paid.sort_values(by='Price', ascending=False).head(30)
df_top_paid
plt.figure(figsize=(20,20))

g = sns.barplot(x='Name', y='Price', data = df_top_paid,hue='Category')

g.set_title('Top paid apps', fontsize=5)

g.set_xticklabels(g.get_xticklabels(), rotation=90)

plt.show()
dayOfWeek={0:'Monday', 1:'Tuesday', 2:'Wednesday', 3:'Thursday', 4:'Friday', 5:'Saturday', 6:'Sunday'}

df['Weekday'] = pd.DatetimeIndex(df['Date']).weekday

df['Weekday'] = df['Weekday'].map(dayOfWeek)

df_week = df.groupby(['Weekday'])['Reviews'].sum()

df_week = pd.DataFrame(df_week.reset_index())

df_week
g = sns.barplot(x='Weekday', y='Reviews', data = df_week)

g.set_title('Reviews on weekdays', fontsize=25)

g.set_xticklabels(g.get_xticklabels(), rotation=90)

plt.show()
df5=df.groupby(['Weekday'])['Name'].count().sort_values(ascending=False)

df5 = pd.DataFrame(df5.reset_index())

df5
g = sns.barplot(x='Weekday', y='Name', data = df5)

g.set_title('apps published on weekdays', fontsize=5)

g.set_xticklabels(g.get_xticklabels(), rotation=45)

plt.show()