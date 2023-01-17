####### importing all Liaberies
import warnings

warnings.filterwarnings('ignore')

import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt 

import seaborn  as sns
df1=pd.read_csv('/kaggle/input/netflix-shows/netflix_titles.csv')
# making copy of dataset to perform all EDA

df=df1.copy()
df.head()
### CHEKING NUMBER OF ROWS AND COLUMNS

print('Number of showa',df.shape[0])

print('Number of features',df.shape[1])
df.describe() ## 5POINT SUMMARY
df.describe(include='object')### 5 POINT SUMMARY FOR THE OBJECT COLUMN
df.info() ## DTYPES AND NULL VALUES IN DATASET
df=df.drop('show_id',axis=1)  ### DROPING SHOWID COLUMN BECAUSE ITS NOT NEEDED FOR EDA

df.head()
df['type'].unique()
df.info()
df.isna().sum()
df1=df.copy()
### treating null values
df1=df1.drop(['director','cast'],axis=1 )


df1.isnull().sum()
df1['rating'].nunique()
df1['country'].mode()
df1['rating'].mode()
##df1['date_added']=df1['date_added'].fillna(df1['date_added'].mode())
df1['date_added'].mode()
df1['rating']=df1['rating'].replace(np.nan,'TV-MA')
df1['date_added']=df1['date_added'].replace(np.nan,'January 1, 2020')
df1['country']=df1['country'].replace(np.nan,'United States')
df1[df1['date_added']=='January 1, 2020'].sum()
df1.isnull().sum()
df1.shape
df1['rating'].isna().sum()
df1['date_added']=pd.to_datetime(df1['date_added'])  ### converting the datetime yy/mm/dd
df1['year'] = pd.to_datetime(df1['date_added']).dt.year

df1['year']=df1['year'].astype('int')
df1.head()
#df['year']=df['date_added'].str[0:4]
df1.head()
df1['month'] = pd.to_datetime(df1['date_added']).dt.month

df1['month']= df1['month'].astype('int')
df1.head()
df1.year.nunique()
df1_s1=df1[df1['duration']=='1 Season']   ### storing season 1 value in variable
df1_s1.describe(include='object')
shortmovie=df1[df1['duration']>'90 min']
longmovie=df1[df1['duration']<'90 min']
sns.countplot(df1['type'])  ### dropna data

plt.show()
sns.countplot(df['type'])  ## original data

plt.show()
sns.countplot(longmovie['type'])  ## original data

plt.show()
plt.figure(figsize=(10,8))

sns.countplot(df['rating'])

plt.xticks(rotation='90')

plt.show()
plt.figure(figsize=(10,8))

sns.countplot(df1['rating'])

plt.xticks(rotation='90')

plt.show()
plt.figure(figsize=(10,8))

sns.countplot(df1_s1['rating'])

plt.xticks(rotation='90')

plt.show()
plt.figure(figsize=(10,8))

sns.countplot(longmovie['rating'])

plt.xticks(rotation='90')

plt.show()
plt.figure(figsize=(10,8))

sns.countplot(shortmovie['rating'])

plt.xticks(rotation='90')

plt.show()
plt.figure(figsize=(15,10))

df1_s1['rating'].value_counts().plot(kind='pie',autopct='%1.2f%%')

plt.show()
plt.figure(figsize=(15,10))

df1['rating'].value_counts().plot(kind='pie',autopct='%1.2f%%')

plt.show()
plt.figure(figsize=(15,10))

longmovie['rating'].value_counts().plot(kind='pie',autopct='%1.2f%%')

plt.show()
plt.figure(figsize=(15,10))

shortmovie['rating'].value_counts().plot(kind='pie',autopct='%1.2f%%')

plt.show()
df1['rating'].value_counts()
sns.countplot(df1['month'])

plt.show()
sns.countplot(df1_s1['month'])

plt.show()
sns.countplot(longmovie['month'])

plt.show()
sns.countplot(shortmovie['month'])

plt.show()
sns.countplot(df1['year'])

plt.show()
sns.countplot(df1_s1['year'])

plt.show()
sns.countplot(longmovie['year'])

plt.show()
sns.countplot(shortmovie['year'])

plt.show()
sns.countplot(df1['year'],hue=df1['type'])

plt.show()
sns.countplot(df1_s1['year'],hue=df1_s1['type'])

plt.show()
sns.countplot(longmovie['year'],hue=longmovie['type'])

plt.show()
sns.countplot(shortmovie['year'],hue=shortmovie['type'])

plt.show()
sns.countplot(df1['month'],hue=df1['type'])

plt.show()
sns.countplot(df1_s1['month'],hue=df1_s1['type'])

plt.show()
sns.countplot(longmovie['month'],hue=longmovie['type'])

plt.show()
sns.countplot(shortmovie['month'],hue=shortmovie['type'])

plt.show()
keys = [pair for pair, df2 in df1.groupby(['month'])]



plt.plot(keys, df1.groupby(['month']).count())

plt.xticks(keys)

plt.grid()

plt.show()
keys = [pair for pair, df2 in df1_s1.groupby(['month'])]



plt.plot(keys, df1_s1.groupby(['month']).count())

plt.xticks(keys)

plt.grid()

plt.show()
keys = [pair for pair, df2 in longmovie.groupby(['month'])]



plt.plot(keys, longmovie.groupby(['month']).count())

plt.xticks(keys)

plt.grid()

plt.show()
keys = [pair for pair, df2 in shortmovie.groupby(['month'])]



plt.plot(keys, shortmovie.groupby(['month']).count())

plt.xticks(keys)

plt.grid()

plt.show()
keys = [pair for pair, df2 in df1.groupby(['year'])]



plt.plot(keys, df1.groupby(['year']).count())

plt.xticks(keys)

plt.grid()

plt.show()
keys = [pair for pair, df2 in df1_s1.groupby(['year'])]



plt.plot(keys, df1_s1.groupby(['year']).count()['rating'])

plt.xticks(keys)

plt.grid()

plt.show()
keys = [pair for pair, df2 in longmovie.groupby(['year'])]



plt.plot(keys, longmovie.groupby(['year']).count()['rating'])

plt.xticks(keys)

plt.grid()

plt.show()
keys = [pair for pair, df2 in shortmovie.groupby(['year'])]



plt.plot(keys, shortmovie.groupby(['year']).count()['rating'])

plt.xticks(keys)

plt.grid()

plt.show()