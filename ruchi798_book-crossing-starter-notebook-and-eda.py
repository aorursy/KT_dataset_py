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
import pandas_profiling

import seaborn as sns

import matplotlib.pyplot as plt

import requests



from IPython.core.display import Image

from colorama import Fore, Back, Style

y_ = Fore.YELLOW

r_ = Fore.RED

g_ = Fore.GREEN

b_ = Fore.BLUE

m_ = Fore.MAGENTA

sr_ = Style.RESET_ALL
custom_colors = ['#48bfe3','#56cfe1','#64dfdf','#72efdd','#80ffdb']

customPalette = sns.color_palette(custom_colors)

sns.palplot(sns.color_palette(custom_colors),size=1)
#Users

u_cols = ['user_id', 'location', 'age']

users = pd.read_csv('../input/bookcrossing-dataset/Book reviews/BX-Users.csv', sep=';', names=u_cols, encoding='latin-1',low_memory=False)



#Books

i_cols = ['isbn', 'book_title' ,'book_author','year_of_publication', 'publisher', 'img_s', 'img_m', 'img_l']

items = pd.read_csv('../input/bookcrossing-dataset/Book reviews/BX_Books.csv', sep=';', names=i_cols, encoding='latin-1',low_memory=False)



#Ratings

r_cols = ['user_id', 'isbn', 'rating']

ratings = pd.read_csv('../input/bookcrossing-dataset/Book reviews/BX-Book-Ratings.csv', sep=';', names=r_cols, encoding='latin-1',low_memory=False)
users.head(5)
users.describe()
print(f"{y_}{users.dtypes}\n") 
items.head(5)
items.describe()
print(f"{y_}{items.dtypes}\n") 
ratings.head(5)
ratings.describe()
print(f"{y_}{ratings.dtypes}\n") 
users = users.drop(users.index[0])

items = items.drop(items.index[0])

ratings = ratings.drop(ratings.index[0])
users['age'] = users['age'].astype(float)

users['user_id'] = users['user_id'].astype(int)

ratings['user_id'] = ratings['user_id'].astype(int)

ratings['rating'] = ratings['rating'].astype(int)
users['age'].describe()
users.loc[(users.age>99) | (users.age<5),'age'] = np.nan
df = pd.merge(users, ratings, on='user_id')

df = pd.merge(df, items, on='isbn')

df.head(5)
df.shape
location = df.location.str.split(', ', n=2, expand=True)

location.columns=['city', 'state', 'country']



df['city'] = location['city']

df['state'] = location['state']

df['country'] = location['country']
def images(col,i):

    url = df[col][i]

    response = requests.get(url)

    img = Image(url)

    return img
images('img_s',0)
images('img_m',0)
images('img_l',0)
df = df.drop(['location','img_s','img_m','img_l'], axis = 1) 
df.dtypes
# profile = pandas_profiling.ProfileReport(df)

# profile
plt.figure(figsize=(12,10))

sns.countplot(x='rating',data=df,palette=customPalette)

plt.title('Rating Distribution',size=20)

plt.show()
plt.figure(figsize=(12,8))

sns.distplot(df['age'],kde=False)

plt.xlabel('Age')

plt.ylabel('count')

plt.title('Age Distribution',size=20)

plt.show()
df_v=df['year_of_publication'].value_counts().head(25).reset_index()

df_v.columns=['year','count']

df_v['year']='Year '+df_v['year']



plt.figure(figsize=(12,10))

sns.barplot(x='count',y='year',data=df_v,palette=customPalette)

plt.ylabel('Year Of Publication')

plt.title('Years of Publication',size=20)

plt.show()
def barplot(df,col,l):

    df_v=df[col].value_counts().head(25).reset_index()

    df_v.columns=[col,'count']



    plt.figure(figsize=(12,10))

    sns.barplot(x='count',y=col,data=df_v,palette=customPalette)

    plt.ylabel(l)

    plt.title(l,size=20)

    plt.show()
barplot(df,'book_title','Book Title')
barplot(df,'book_author','Book Author')
barplot(df,'publisher','Book publisher')