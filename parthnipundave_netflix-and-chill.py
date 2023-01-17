import pandas as pd
data = pd.read_csv('../input/netflix-shows/netflix_titles.csv')
data.head()
data.info()
data.dropna(inplace=True)
import seaborn as sns

import matplotlib.pyplot as plt
ratingsdf = data['rating'].value_counts()
ratingsdf.columns=['Rating','Count']

df = pd.DataFrame(ratingsdf)

df['Count'] = df['rating']

df['Rating'] = df.index

df.reset_index(inplace=True)

df.drop(['index','rating'],axis=1,inplace=True)

df
plt.figure(figsize=(12,7))

sns.barplot(data=df,x='Rating',y='Count')
plt.figure(figsize=(12,7))

sns.countplot(data=data,x='rating')
plt.figure(figsize=(12,7))

sns.countplot(data=data[data['type']=='Movie'],x='rating')
plt.figure(figsize=(12,7))

sns.countplot(data=data[data['type']=='TV Show'],x='rating')
data[data['type']=='TV Show']['rating'].unique()
def Rate(cat,arg):

    plt.figure(figsize=(12,7))

    sns.countplot(data=data[data[cat]==arg],x='rating')
Rate('country','India')
Rate('duration','1 Season')
listeddf = data['listed_in'].value_counts()

listeddf.columns=['listed_in','Count']

df = pd.DataFrame(listeddf)

df['Count'] = df['listed_in']

df['Listed_in'] = df.index

df.reset_index(inplace=True)

df.drop(['index','listed_in'],axis=1,inplace=True)

df
plt.figure(figsize=(15,100))

sns.barplot(data=df,y='Listed_in',x='Count')
plt.figure(figsize=(20,100))

sns.countplot(data=data,y='listed_in')
lists = []

listed = data['listed_in'].str.split(',')

for li in listed:

    for l in li:

        lists.append(l)

df= pd.DataFrame(data=lists,columns=['Genre'])

df
plt.figure(figsize=(10,35))

sns.countplot(data=df,y='Genre')
def Genre(cat='all',arg='all'):

    if cat == 'all' or arg=='all':

        plt.figure(figsize=(10,35))

        sns.countplot(data=df,y='Genre')

    else:

        lists = []

        listed = data[data[cat]==arg]['listed_in'].str.split(',')

        for li in listed:

            for l in li:

                lists.append(l)

        df= pd.DataFrame(data=lists,columns=['Genre'])

        df

        plt.figure(figsize=(35,20))

        sns.countplot(data=df,y='Genre')

Genre('duration','1 Season')
Genre('country','India')
df = data['release_year'].value_counts()

df = pd.DataFrame(df)

df['Release_year'] = df.index

df['Count'] = df['release_year']

df.reset_index(inplace=True)

df.drop(['index','release_year'],inplace=True,axis=1)

df
plt.figure(figsize=(12,15))

sns.countplot(data=data,y='release_year')
def Year(cat='all',arg='all'):

    if cat =='all' or arg=='all':

        plt.figure(figsize=(12,15))

        sns.countplot(data=data,y='release_year')

    else:

        df = data[data[cat]==arg]['release_year'].value_counts()

        df = pd.DataFrame(df)

        df['Release_year'] = df.index

        df['Count'] = df['release_year']

        df.reset_index(inplace=True)

        df.drop(['index','release_year'],inplace=True,axis=1)

        df

        plt.figure(figsize=(30,15))

        sns.barplot(data=df,x='Release_year',y='Count')
Year('country','India')
country = data['country']
count = country.str.split(',')
country = []

for countie in count:

    for c in countie:

        country.append(c)
df = pd.DataFrame(data=country,columns=['Country'])
df = df['Country'].value_counts()

df = pd.DataFrame(df)

df['Count'] = df['Country']

df['Country'] = df.index

df.reset_index(inplace=True)

df.drop('index',axis=1,inplace=True)

df
df[df['Country']=='India']
data['country'].values
plt.figure(figsize=(15,25))

# plt.xlim(2000)

sns.barplot(data=df[df['Count']>50],x='Count',y='Country')
plt.figure(figsize=(15,25))

# plt.xlim(2000)

sns.barplot(data=df[df['Count']<50],x='Count',y='Country')
sns.countplot(data=data,x='type')
plt.figure(figsize=(12,7))

sns.countplot(data=data,x='rating',hue='type')
def Type(cat='all',arg='all'):

    if cat =='all' or arg =='all':

        sns.countplot(data=data,x='type')

    else:

        sns.countplot(data=data[data[cat]==arg],x='type')

Type('country','India')
plt.figure(figsize=(10,25))

sns.countplot(data=data,y='release_year',hue='type')


sns.countplot(data=data[data['country']=='India'],x='country',hue='type')
df = data['director'].value_counts()

df.columns=['Director','count']

df = pd.DataFrame(df)

df['Count'] = df['director']

df['Director']=df.index

df.reset_index(inplace=True)

df.drop(['index','director'],axis=1,inplace=True)

df
plt.figure(figsize=(20,7))

sns.barplot(data=df[:10],x='Director',y='Count')
data['duration'].nunique()
data.duration
plt.figure(figsize=(15,35))

sns.countplot(data=data,y='duration')
tv=data[data['type']=='TV Show']
plt.figure(figsize=(15,7))

sns.countplot(data=tv,x='duration')
data.info()
data['cast'].nunique()
casts = data['cast'].str.split(',')
actors = []

for cast in casts:

    for actor in cast:

        actors.append(actor)
df = pd.DataFrame(actors,columns=['Actor'])

df = df['Actor'].value_counts()

df = pd.DataFrame(df)

df['Count'] = df['Actor']

df['Actor'] = df.index

df.reset_index(inplace=True)

df.drop('index',axis=1,inplace=True)

df
plt.figure(figsize=(20,7))

sns.barplot(data=df[:10],x='Actor',y='Count')