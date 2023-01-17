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
data=pd.read_csv('../input/movies-on-netflix-prime-video-hulu-and-disney/MoviesOnStreamingPlatforms_updated.csv')
data.head()
data.head()
data.columns
genres = data['Genres'].str.get_dummies(',')
data = pd.concat([data,genres], axis = 1, sort = False)
data.head()
data.drop(['Genres'],axis=1,inplace=True)
num_netflix = data['Netflix'].sum()
print(num_netflix)
num_hulu = data['Hulu'].sum()
print(num_hulu)
num_prime = data['Prime Video'].sum()
print(num_prime)
num_disney = data['Disney+'].sum()
print(num_disney)
num_platform = (num_netflix,num_hulu,num_prime,num_disney)
col_names = ('Netflix','Hulu','Prime Video','Disney+')
PlatformList = list(zip(col_names,num_platform))
PlatformCounts = pd.DataFrame(data=PlatformList,columns=['Platform','Number of Movie'])
PlatformCounts
import seaborn as sns

sns.barplot(x=PlatformCounts['Platform'],y=PlatformCounts['Number of Movie'])
import plotly.express as px
age_column = 'Age'
age_data = data[[age_column]].explode(age_column).groupby(age_column).size().to_frame(name='No of Movies')

fig = px.pie(
    age_data,
    values='No of Movies',
    names=age_data.index
)
fig.show()
netflix_data=data.copy()
netflix_data.drop(['Prime Video','Disney+','Hulu','Unnamed: 0'],axis=1,inplace=True)
netflix_data = netflix_data[netflix_data.Netflix==1]
netflix_data.head(10)
netflix_data['Rotten Tomatoes'] = netflix_data['Rotten Tomatoes'][netflix_data['Rotten Tomatoes'].notnull()].str.replace('%', '').astype(float)
netflix_data.fillna({'IMDb':netflix_data['IMDb'].mean()}, inplace=True)
netflix_data.fillna({'Age':"all"}, inplace=True)
netflix_data.fillna({'Rotten Tomatoes':netflix_data['Rotten Tomatoes'].mean()}, inplace=True)
netflix_data.fillna({'Directors':"NA"}, inplace=True)
netflix_data.fillna({'Country':"NA"}, inplace=True)
netflix_data['IMDb'].mean()
netflix_data['Rotten Tomatoes'].mean()
netflix_data.sort_values('IMDb',ascending=False,inplace=True)
netflix_data
prime_data=data.copy()
prime_data.drop(['Netflix','Disney+','Hulu','Unnamed: 0'],axis=1,inplace=True)
prime_data = prime_data[data['Prime Video']==1]
prime_data.head()
prime_data['Rotten Tomatoes'] = prime_data['Rotten Tomatoes'][prime_data['Rotten Tomatoes'].notnull()].str.replace('%', '').astype(float)
prime_data.fillna({'IMDb':netflix_data['IMDb'].mean()}, inplace=True)
prime_data.fillna({'Age':"all"}, inplace=True)
prime_data.fillna({'Rotten Tomatoes':netflix_data['Rotten Tomatoes'].mean()}, inplace=True)
prime_data.fillna({'Directors':"NA"}, inplace=True)
prime_data.fillna({'Country':"NA"}, inplace=True)
prime_data['IMDb'].mean()
prime_data['Rotten Tomatoes'].mean()
prime_data.sort_values('IMDb',ascending=False,inplace=True)
prime_data
hulu_data=data.copy()
hulu_data.drop(['Netflix','Disney+','Prime Video','Unnamed: 0'],axis=1,inplace=True)
hulu_data = hulu_data[data['Hulu']==1]
hulu_data.head()
hulu_data['Rotten Tomatoes'] = hulu_data['Rotten Tomatoes'][hulu_data['Rotten Tomatoes'].notnull()].str.replace('%', '').astype(float)
hulu_data.fillna({'IMDb':netflix_data['IMDb'].mean()}, inplace=True)
hulu_data.fillna({'Age':"NA"}, inplace=True)
hulu_data.fillna({'Rotten Tomatoes':netflix_data['Rotten Tomatoes'].mean()}, inplace=True)
hulu_data.fillna({'Directors':"NA"}, inplace=True)
hulu_data.fillna({'Country':"NA"}, inplace=True)
hulu_data['IMDb'].mean()
hulu_data['Rotten Tomatoes'].mean()
hulu_data.sort_values('IMDb',ascending=False,inplace=True)
hulu_data
disney_data=data.copy()
disney_data.drop(['Netflix','Prime Video','Hulu','Unnamed: 0'],axis=1,inplace=True)
disney_data = disney_data[data['Disney+']==1]
disney_data.head()

disney_data['Rotten Tomatoes'] = disney_data['Rotten Tomatoes'].str.replace('%', '').astype(float)
disney_data.fillna({'IMDb':netflix_data['IMDb'].mean()}, inplace=True)
disney_data.fillna({'Age':"NA"}, inplace=True)
disney_data.fillna({'Rotten Tomatoes':netflix_data['Rotten Tomatoes'].mean()}, inplace=True)
disney_data.fillna({'Directors':"NA"}, inplace=True)
disney_data.fillna({'Country':"NA"}, inplace=True)

disney_data.sort_values('IMDb',ascending=False,inplace=True)
disney_data
disney_data['IMDb'].mean()
disney_data['Rotten Tomatoes'].mean()
