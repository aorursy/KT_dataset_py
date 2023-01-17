# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

from collections import Counter

warnings.filterwarnings('ignore')

%matplotlib inline



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df=pd.read_csv('../input/movies-on-netflix-prime-video-hulu-and-disney/MoviesOnStreamingPlatforms_updated.csv')
df.head()
df.shape
df.dtypes
df.drop(['Unnamed: 0','Type'],axis=1,inplace=True)
df.isnull().sum()
len(df['Title'].unique())
print(f'''{df['Netflix'].sum()} titles are streaming in Netflix , {df['Hulu'].sum()} titles are streaming in Hulu, {df['Prime Video'].sum()} titles are streaming on Prime video , {df['Disney+'].sum()} titles are streaming on Disney+''')
df[(df['Netflix']==1) & (df['Hulu']==1) & (df['Prime Video']==1) & (df['Disney+']==1)]
df['OTT_Count']=df['Netflix']+df['Hulu']+df['Prime Video']+df['Disney+']
(df['OTT_Count'].value_counts()/df.shape[0])*100
df[df['OTT_Count']==3]
netflix=df[(df['Netflix']==1) & (df['Hulu']==0) &(df['Prime Video']==0)& (df['Disney+']==0)]

netflix.shape
prime=df[(df['Netflix']==0) & (df['Hulu']==0) &(df['Prime Video']==1)& (df['Disney+']==0)]

prime.shape
(netflix['Age'].value_counts()/netflix.shape[0])*100
(prime['Age'].value_counts()/prime.shape[0])*100
f,ax=plt.subplots(1,2,figsize=(12,5))

sns.distplot(netflix['IMDb'],ax=ax[0])

ax[0].set_title("Distribution of IMDb rating-Netflix Titles",fontsize=15)

ax[0].set_xlabel("IMDb rating",fontsize=8)

ax[0].set_ylabel("Frequency",fontsize=8)

sns.distplot(prime['IMDb'],ax=ax[1])

ax[1].set_title("Distribution of IMDb rating-Prime Titles",fontsize=15)

ax[1].set_xlabel("IMDb rating",fontsize=8)

ax[1].set_ylabel("Frequency",fontsize=8)
plt.figure(figsize=(8,8))

sns.kdeplot(netflix['IMDb'],shade=True,color='red')

sns.kdeplot(prime['IMDb'],shade=True,color='blue')

plt.legend(title='IMDB Rating',labels=['netflix','prime'])

plt.title("Distribution of Ratings",fontsize=15)

plt.xlabel("IMDB Rating",fontsize=8)

netflix['Rotten Tomatoes']=netflix['Rotten Tomatoes'].str.replace(r'%',r'').astype('float')

prime['Rotten Tomatoes']=prime['Rotten Tomatoes'].str.replace(r'%',r'').astype('float')
f,ax=plt.subplots(1,2,figsize=(12,5))

sns.distplot(netflix['Rotten Tomatoes'],ax=ax[0])

ax[0].set_title("Distribution of Rotten Tomatoes rating",fontsize=15)

ax[0].set_xlabel("Rotten Tomatoes rating",fontsize=8)

ax[0].set_ylabel("Frequency",fontsize=8)

sns.distplot(prime['Rotten Tomatoes'],ax=ax[1])

ax[1].set_title("Distribution of Rotten Tomatoes rating",fontsize=15)

ax[1].set_xlabel("Rotten Tomatoes rating",fontsize=8)

ax[1].set_ylabel("Frequency",fontsize=8)
def clean_column(df,col):

    """

    Function to extract and count the individual entries in a column separated by comma.

    param:

    df : dataframe

    col:column to be counted

    returns:

    set containing values available in the column

    counter object with count of each values in the column

    """

    df[col]=df[col].astype('str')

    empty_set=set()

    count_values=Counter()



    for gen in df[col]:

        gen=gen.split(',')

        empty_set.update([g.strip() for g in gen])

        for g in gen:

            count_values[g.strip()]+=1

            

    return empty_set,count_values
genres,count_genres=clean_column(netflix,'Genres')

genres_prime,count_genres_prime=clean_column(prime,'Genres')
def clean_data(data,dictval):

    """

    A function to create a dataframe from dictionary object for plotting

    params:

    dictval:dictionary

    data:dataframe for analysis

    returns:

    dataframe

    """

    df=pd.DataFrame.from_dict(dictval,orient='index') # keys should be rows.

    df.sort_values(0,ascending=False,inplace=True)

    df.rename(columns={0:'count'},inplace=True)

    df['perc']=(df['count']/data.shape[0])*100

    return df
netflix_genres=clean_data(netflix,count_genres)

prime_genres=clean_data(prime,count_genres_prime)
netflix_genres
prime_genres
f,ax=plt.subplots(1,2,figsize=(12,8))

sns.barplot(netflix_genres[:10].index,y='perc',data=netflix_genres[:10],ax=ax[0])

ax[0].set_xlabel("Genres",fontsize=8)

ax[0].set_ylabel("Percengage of titles(%)",fontsize=8)

ax[0].set_title("Genres in Netflix",fontsize=15)

ax[0].tick_params(labelrotation=90)

sns.barplot(prime_genres[:10].index,y='perc',data=prime_genres[:10],ax=ax[1])

ax[1].set_xlabel("Genres",fontsize=8)

ax[1].set_ylabel("Percengage of titles(%)",fontsize=8)

ax[1].set_title("Genres in Prime",fontsize=15)

ax[1].tick_params(labelrotation=90)