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
nf=pd.read_csv('/kaggle/input/netflix-shows/netflix_titles.csv')
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
nf.head()
nf.describe()
nf.info()
#relation between release year and ratings of netflix shows

plt.figure(figsize=(20,6))
plt.title('Release Year vs Ratings',fontsize=15)
sns.set_style('darkgrid')
sns.countplot(nf[nf['release_year']>=2010]['release_year'],hue=nf['rating'],palette='coolwarm',color='purple')
plt.xticks(rotation=45)
plt.show()
nf.isnull().sum()
#Data Cleaning
#To remove null values apply rating of shows/films the maximum rating of that year released
df=nf.copy()
s=nf[(nf['rating'].isnull()) & (nf['release_year']>=2016)].index
lis=nf['rating'].tolist()
for i in s:
    lis[i]='TV-MA'
nf['rating_new']=lis
s=nf[(nf['rating_new'].isnull()) & (nf['release_year']<2016)].index
lis=nf['rating_new'].tolist()
for i in s:
    lis[i]='TV-14'
nf['rating_new']=lis
nf.drop('rating',axis=1)
plt.figure(figsize=(6,4))
sns.countplot(x='type',data=nf,palette='coolwarm')
plt.xticks(rotation=90)
plt.show()

plt.title(label='Top 10 countries releasing movies in netflix.',fontsize=15)
nf[nf['type']=='Movie']['country'].value_counts()[:10].plot(kind='bar',color='orange')

plt.figure(figsize=(10,5))
plt.title(label='Top 10 tv series gerres in netflix.',fontsize=15)

nf[nf['type']=='TV Show']['listed_in'].value_counts()[:10].plot(kind='barh',color='purple',alpha=0.5)

plt.figure(figsize=(10,5))
plt.title(label='Shows Produced by netflix',fontsize=18)
nf.groupby('release_year')['type'].count()[-22:-1].plot(kind='barh',color='pink')
nf[nf['type']=='TV Show']['duration'].value_counts()
plt.title('Duration of TV Shoes on Netflix')
nf[nf['type']=='TV Show']['duration'].value_counts().plot(kind='bar')
nf.drop('director',inplace=True,axis=1)
nf.drop('cast',inplace=True,axis=1)
nf['country'].fillna(value='United States',inplace=True)
nf.drop('rating',axis=1,inplace=True)
l=[]
def fun(x):
    l.append(x.split()[0])
nf['date_added'].dropna().apply(fun)
nf.dropna(inplace=True)
nf['month']=l
nf.head()
plt.title('Releases in 2019(month-wise)',fontsize=15)
plt.xlabel('Count')
plt.ylabel('Month')
nf[nf['release_year']==2019]['month'].value_counts().plot(kind='barh',color='yellow')
#in 2019 november has more releases
