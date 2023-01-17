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
# importing the library which is useful for us:

import matplotlib.pyplot as plt # for performing data visulization through graph and plotting data

import seaborn as sns # it is another useful library to plot graphs and data visualization, but it provides better and beautiful graphs
# Reading data

data=pd.read_csv('/kaggle/input/windows-store/msft.csv')

data.head()
# info of data

data.info()
print(data.isnull().sum()) # printing the null values within data

data.dropna(how='any',inplace=True)# dropping rows having null value

data.isnull().sum()# checking again
#lets desribe about the data

print(data.describe(include='all').T)

fig,ax=plt.subplots(1,2,figsize=(10,4))

plt.style.use('ggplot')

sns.boxplot(y=data['Rating'],ax=ax[0])

sns.boxplot(y=data['No of people Rated'],ax=ax[1])

ax[0].set_title("What people rated")

ax[1].set_title("How many people Rated")
# To check unique value within various categorical variable

print('Category has Unique values\n',data['Category'].unique())

print('Price has Unique values\n',data['Price'].unique())

data['Price'].replace({'Free':0},inplace=True)

data.head()
data['Price']=data['Price'].str.extract('(\d+)')

data['Price'].fillna(0,inplace=True)# we have to fill Nan value with 0 as previously our 0 is converted to nan value

data.head()
plt.figure(figsize=(10,5))

plt.style.use('ggplot')

sns.countplot(data['Category'])

plt.xticks(rotation=70)
data['Price']=data['Price'].astype(int)

data.info()
book_data=data[data['Category']=='Books']

print('Avg,Max,Min Rating of Book columns')

print(round(book_data['Rating'].mean(),2),book_data['Rating'].max(),book_data['Rating'].min())

book_data['Value']=book_data['Price'].apply(lambda x :'Free' if x ==0 else 'SomeCost')

fig,ax=plt.subplots(1,2,figsize=(10,5))

sns.countplot(book_data['Value'],ax=ax[0])

sns.distplot(book_data['Rating'],ax=ax[1])



music_data=data[data['Category']=='Music']

print('Avg,Max,Min Rating of Music columns')

print(round(music_data['Rating'].mean(),2),music_data['Rating'].max(),music_data['Rating'].min())

music_data['Value']=music_data['Price'].apply(lambda x :'Free' if x ==0 else 'SomeCost')

fig,ax=plt.subplots(1,2,figsize=(10,5))

sns.countplot(music_data['Value'],ax=ax[0])

sns.distplot(music_data['Rating'],ax=ax[1])
book_data['Popularity']=book_data['Rating'] * book_data['No of people Rated']

top5_book=book_data.loc[book_data['Popularity'].nlargest(5).index]

print("Top 5 books are:\n",top5_book['Name'].to_string(index=False))
music_data['Popularity']=music_data['Rating'] * music_data['No of people Rated']

top5_music=music_data.loc[music_data['Popularity'].nlargest(5).index]

print("Top 5 books are:\n",top5_music['Name'].to_string(index=False))