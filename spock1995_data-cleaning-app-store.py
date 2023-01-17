import pandas as pd

import numpy as np

import os

import matplotlib.pyplot as plt



%matplotlib inline

#Data_Cleaning_App_Store

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df = pd.read_csv('/kaggle/input/google-play-store-apps/googleplaystore.csv')

reviews = pd.read_csv('/kaggle/input/google-play-store-apps/googleplaystore_user_reviews.csv')
df.head()
df.info()
df.describe()
df.shape
df.rename(columns={'Content Rating':'Content_Rating', 'Last Updated':'Last_Updated', 'Current Ver':'Current_Ver','Android Ver':'Android_Ver'}, inplace=True)
df.head(10)
df[df.Rating.isnull()].head()
df.Category.unique()
df[df.Category == '1.9']
df.drop(10472,axis=0,inplace=True)
df[df.Category == '1.9']
df.Reviews = df.Reviews.astype(int)
Applications = df[df.Rating.isnull()].App

reviews_applications = reviews.App.unique()

for i in Applications:

        if i in reviews_applications:

            print(i)
df[df.App == 'Blood Pressure']
df.iloc[2513,2]=df[df.App == 'Blood Pressure'].Rating.mean()
df[df.App == 'Blood Pressure']
drp = df[df.Rating.isnull()].index
drp = list(drp)

df.drop(drp, inplace = True)
df.sample(5)
lds = np.array(df.Size.str.find('k'))
ls = np.where(lds>0) #list of indexes where size is in kb
df.Size = df.Size.str.replace('M','');

df.Size = df.Size.str.replace('k','');
df_temp = df.copy()
df_temp.drop(df[df.Size == 'Varies with device'].index,axis=0, inplace=True)
df_temp.Size = df_temp.Size.astype('float') 
df.reset_index(False,inplace=True)
df.drop('index',axis=1, inplace=True)
for x,y in df.iterrows():

    if y.Size == 'Varies with device':

        val = y.Category

        df.iloc[x,4]=df_temp.groupby('Category').Size.median()[val]

df.Size = df.Size.astype('float')
df.sample()
for x in ls:

    df.iloc[x,4]=df.iloc[x,4]*0.001

    

#Size column is ready
df.Installs.unique()
df.Installs=df.Installs.str.replace('+','')

df.Installs=df.Installs.str.replace(',','')
df.Installs=df.Installs.astype('int')
df.sample()
df.Price.unique()
df.Price = df.Price.str.replace('$', '')
df.Price = df.Price.astype('float')
df.Price.unique()
df.sample(5)
#8535, 2017
temp = np.array(df.Genres.str.find(';'))

ls = np.where(temp>0) #Finding out the Indexes where there are multiple genres
df2 = df.iloc[np.r_[ls],:]  #Copying only those indexes to another dataframe to split them
df.shape
df2.shape
df.iloc[np.r_[1,4,9,25],:]
df.Genres = df.Genres.apply(lambda x: x.split(';')[0])
df2.head()
df2.Genres = df2.Genres.apply(lambda x: x.split(';')[1])
df2.head()
df = df.append(df2, ignore_index=True)
df.shape
del df2
df.info()
df.head()
df.Last_Updated = pd.to_datetime(df['Last_Updated']) 
df.info()
ind = df[df['Current_Ver']=='Varies with device'].index
df.Current_Ver.mode()
df_t = df.copy()
df_t.drop(ind, axis=0, inplace=True)
df_t.Current_Ver.mode()
df.Current_Ver = df.Current_Ver.replace('Varies with device', '1.0')
df.head()
df.drop('Android_Ver', axis=1, inplace=True)
df[df.duplicated()]
df.drop_duplicates(keep='first', inplace=True)
df.reset_index(inplace=True)

df.drop('index',axis=1, inplace=True)

df.head()