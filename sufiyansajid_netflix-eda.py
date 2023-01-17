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
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns #importing our visualization library

import matplotlib.pyplot as plt

import plotly.express as px

%matplotlib inline

df = pd.read_csv('/kaggle/input/netflix-shows/netflix_titles.csv')

print(df.shape)
df.head()
df.isnull().sum()

sns.set()

plt.figure(figsize=(15,8))

sns.heatmap(df.isnull(),cmap = 'viridis')

plt.show()

df.isnull().sum()
df['country'].replace(np.nan, 'United States',inplace = True)

df['rating'].replace(np.nan, 'TV-MA',inplace = True)



df.drop('date_added',axis=1,inplace=True)

df.head()
sns.countplot(x='type',data=df)
plt.figure(figsize =(12,9))

sns.countplot(x='rating',data=df,order=df['rating'].value_counts().index[0:50],hue=df['type'])
sns.set(style='darkgrid')

plt.figure(figsize=(25,10))

sns.countplot(x='country',data=df,hue='type',order=df['country'].value_counts().index[0:10])

plt.xticks(rotation=90)

plt.show()
top = df['country'].value_counts()[0:8]

fig=px.pie(df, values=top, names=top.index, labels=top.index)

fig.update_traces(textposition='inside',textinfo='percent+label')

fig.show()
sns.set()

plt.figure(figsize=(10,10))

sns.countplot(x="release_year",data= df,order = df['release_year'].value_counts().index[0:40])

plt.xticks(rotation=90)

plt.title('Number of Movies & Tv-shows per year')

plt.show()
top_listed=df['listed_in'].value_counts()[0:25]



fig=px.pie(df,values=top_listed,names=top_listed.index,labels=top_listed.index)

fig.update_traces(textposition='inside',textinfo='percent+label')

fig.show()
sns.set(style='white')

plt.figure(figsize=(15,15))

sns.countplot(x='listed_in',hue='rating',data = df,order =df["listed_in"].value_counts().index[0:10])

plt.xticks(rotation = 90)

plt.show()
df['season_count'] = df.apply(lambda x : x['duration'].split(" ")[0] if "Season" in x['duration'] else "", axis = 1)

df['duration'] = df.apply(lambda x : x['duration'].split(" ")[0] if "Season" not in x['duration'] else "", axis = 1)

df['season_count'].value_counts()
display(df[df['season_count'] == '15'][['title','director', 'cast','country','release_year']])
time=df.loc[df['listed_in'] == 'Documentaries',:]

time=time.loc[df['duration'] >= '99',:]

time

kids_show=df.loc[df['listed_in'] == "Kids' TV",:].reset_index()

kids_show[["title","country","release_year"]].head(10)
Country = pd.DataFrame(df["country"].value_counts().reset_index().values,columns=["country","TotalShows"])

Country.head()



fig = px.choropleth(   

    locationmode='country names',

    locations=Country.country,

    featureidkey="Country.country",

    labels=Country["TotalShows"]

)

fig.show()