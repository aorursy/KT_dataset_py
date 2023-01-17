import os

import zipfile



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
#zipfile_path='kernel/input/google-play-store-apps.zip'

#zipfile_ref=zipfile.ZipFile(zipfile_path, 'r')

#zipfile_ref.extractall()

#zipfile_ref.close()
df = pd.read_csv('../input/google-play-store-apps/googleplaystore.csv')

df.head()
df.shape
df.info()
df['Size'].value_counts()
def clean(x):

    if (x[-1]=='M'):

        x=x[:-1]

    elif(x[-1]=='k'):

        x=float(x[:-1])/1024

    else :

        x=np.nan

    return x

df['Size']=df['Size'].apply(lambda x : clean(x))
df['Size']
df['Installs'].value_counts()
def clear(x):

    if (x[-1]=='+'):

        x=x[:-1].replace(',','')

    elif (x=='Free'):

        x=0

    else:

        x=x.replace(',','')

    return int(x)

df['Installs']=df['Installs'].apply(lambda x : clear(x))
df['Installs'].head()
df.head()
df['Type'].value_counts()
def clear(x):

    if x =='0':

        x = 'Free'

    return x

df['Type'] = df['Type'].apply(lambda x : clear(x))
df['Type'].value_counts()
df['Price'].unique()
def clear(x):

    if (x[0]=='$'):

        x=x[1:]

    elif (x=='Everyone'):

        x=0

    else:

        x=float(x)

    return float(x)

df['Price']=df['Price'].apply(lambda x : clear(x))
df['Price'].value_counts()
df['Content Rating'].value_counts()
df.isnull().sum()
rating_mean = df['Rating'].mean()
# since there are 1474 null values in rating column, we need to take care of those missing values

#df[df['Rating'].isnull()]['Rating'] = rating_mean

df['Rating'].fillna(value=df['Rating'].mean(), inplace=True)
df[df['Rating'].isnull()]
df.isnull().sum()
df['Size']=pd.DataFrame(data=df['Size'], dtype=np.float)
df['Size'].mean()
df['Size'].fillna(value=df['Size'].mean(), inplace=True)
df.isnull().sum()
df['Current Ver'].unique()
df['Current Ver'].fillna(value='1.0.0', inplace=True)
df[df['Current Ver'].isnull()]
df.isnull().sum()
df.dropna(inplace=True)
df.isnull().sum()
df.shape
df.head()
df.isnull().sum().sum()
df.describe()
df.info()
# We can convert our 'last updated' column of our dataset to a datetime object

df['Last Updated']=pd.to_datetime(df['Last Updated'])
df['Reviews']=pd.DataFrame(data=df['Reviews'], dtype=np.float)
df.info()
df.head()
# Number of apps under different category
plt.figure(figsize=(20,5))

count_category=df.sort_values(by='Category',ascending=False)

count_category['Category'].value_counts().plot(kind='bar',)

#sns.countplot(x='Category', data=count_category)

plt.xticks(rotation=90)
# Number of apps under different genre

plt.figure(figsize=(30,5))

count_genre=df.sort_values(by='Genres', ascending=False)

count_genre['Genres'].value_counts().plot(kind='bar', color='blue')
temp = df['Type'].value_counts()

plt.pie(x=temp, labels=['Free', 'Paid'], autopct='%1.2f%%',shadow=True,explode=(0,0.1))
content_rating=df['Content Rating'].value_counts()

plt.pie(x=content_rating, labels=df['Content Rating'].unique(), shadow =True)
sns.countplot(x=df['Content Rating'])

plt.xticks(rotation=90)
df.columns
top_rated = df.sort_values(by=['Rating', 'Reviews'], ascending=False)

top_rated[['App','Category', 'Rating','Reviews']].head(10)
top_reviewed = df.sort_values(by='Reviews', ascending=False)

top_reviewed[['App', 'Category', 'Reviews']].head(10)
# Large apps

large_size = df.sort_values(by='Size', ascending=False)

large_size[['App','Category', 'Size']].head(10)
# Most installed apps

most_installed = df.sort_values(by='Installs', ascending=False)

most_installed[['App','Category','Installs']].head(10)
# Most expensive apps

most_expensive = df.sort_values(by='Price', ascending=False)

most_expensive[['App', 'Category', 'Price']].head(10)
fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(20,5))

top_rated['Category'].head(50).value_counts().plot(kind='bar', title='Category', ax=ax[0])

top_rated['Content Rating'].head(50).value_counts().plot(kind='bar', title='Content Rating', ax=ax[1])

top_rated['Genres'].head(50).value_counts().plot(kind='bar', title='Genres', ax=ax[2])
fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(20,5))

large_size['Category'].head(50).value_counts().plot(kind='bar', title='Category', ax=ax[0])

large_size['Content Rating'].head(50).value_counts().plot(kind='bar', title='Content Rating', ax=ax[1])

large_size['Genres'].head(50).value_counts().plot(kind='bar', title='Genres', ax=ax[2])
fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(20,5))

most_installed['Category'].head(50).value_counts().plot(kind='bar', title='Category', ax=ax[0])

most_installed['Content Rating'].head(50).value_counts().plot(kind='bar', title='Content Rating', ax=ax[1])

most_installed['Genres'].head(50).value_counts().plot(kind='bar', title='Genres', ax=ax[2])
fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(20,5))

most_expensive['Category'].head(50).value_counts().plot(kind='bar', title='Category', ax=ax[0])

most_expensive['Content Rating'].head(50).value_counts().plot(kind='bar', title='Content Rating',ax = ax[1])

most_expensive['Genres'].head(50).value_counts().plot(kind='bar', title='Genres', ax=ax[2])
df.columns
fig, ax = plt.subplots(nrows=4, ncols=1, figsize=(20,25))

df.groupby('Category')['Rating'].mean().sort_values(ascending=False).plot(kind='bar', title='Rating', ax=ax[0])

df.groupby('Category')['Price'].mean().sort_values(ascending=False).plot(kind='bar', title='Price', ax=ax[1])

df.groupby('Category')['Installs'].mean().sort_values(ascending=False).plot(kind='bar', title='Installs', ax=ax[2])

df.groupby('Category')['Size'].mean().sort_values(ascending=False).plot(kind='bar', title='Size', ax=ax[3])

plt.tight_layout()
fig, ax = plt.subplots(nrows=4, ncols=1, figsize=(20,30))

df.groupby('Genres')['Price'].mean().sort_values(ascending=False).plot(kind='bar', title='Price', ax=ax[0])

df.groupby('Genres')['Installs'].mean().sort_values(ascending=False).plot(kind='bar', title='Installs', ax=ax[1])

df.groupby('Genres')['Size'].mean().sort_values(ascending=False).plot(kind='bar', title='Size', ax=ax[2])

df.groupby('Genres')['Rating'].mean().sort_values(ascending=False).plot(kind='bar', title='Rating', ax=ax[3])

plt.tight_layout()
fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(20,5))

df.groupby('Content Rating')['Price'].mean().sort_values(ascending=False).plot(kind='bar', title='Price', ax=ax[0])

df.groupby('Content Rating')['Size'].mean().sort_values(ascending=False).plot(kind='bar', title='Size', ax=ax[1])

df.groupby('Content Rating')['Installs'].mean().sort_values(ascending=False).plot(kind='bar', title='Installs', ax=ax[2])

df.groupby('Content Rating')['Rating'].mean().sort_values(ascending=False).plot(kind='bar', title='Ratings', ax=ax[3])

plt.tight_layout()
sns.pairplot(data=df, hue='Type')
sns.heatmap(df.corr())
sns.jointplot(x='Rating', y='Installs', data=df)
sns.jointplot(x='Reviews', y='Installs', data=df)
sns.jointplot(x='Size', y='Installs', data=df)
sns.jointplot(x='Size', y ='Rating', data=df)
sns.jointplot(x='Price', y='Rating', data=df)
sns.jointplot(x='Size', y='Price', data=df)
g=sns.FacetGrid(data=df, col='Category', col_wrap=5, sharex=True, margin_titles=True, sharey=False)

g.map(sns.countplot, 'Rating', order=[1,2,3,4,5])
plt.figure(figsize=(20,5))

#temp = df.groupby('Category')['App'].sum()

sns.countplot(x='Category', data=df, hue='Type')

plt.xticks(rotation=90)
import pandas_profiling
profiling_report=pandas_profiling.ProfileReport(df)
#profiling_report
profiling_report.to_file('Play Store Apps Report.html')