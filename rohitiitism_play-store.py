# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df=pd.read_csv('../input/google-play-store-apps/googleplaystore.csv')

review=pd.read_csv('../input/google-play-store-apps/googleplaystore_user_reviews.csv')

df.head()
df=df.drop([10472])

df
df['Installs']=df['Installs'].astype(str)

df["Installs"]=df['Installs'].str.replace('+','')

df["Installs"]=df['Installs'].str.replace(',','')

df['Installs']=df['Installs'].astype(float)
df=df.sort_index()

df.head()

type(df['Installs'])
df['Category'].value_counts()
grp=df.groupby('Category')
grp.get_group('FAMILY')
grp['Installs'].value_counts().loc['FAMILY']
df['Installs'] = df['Installs'].astype(int)

res = df.groupby('App')['Installs'].sum().reset_index()

final_result = res.sort_values(by = 'Installs', ascending = False)

final_result(10)
res_c = df.groupby('Category')['Installs'].sum().reset_index()

final_c=res_c.sort_values(by='Installs',ascending=False)

final_c.head(10)
df.loc[df['Size'] == 'Varies with device'].shape

unwanted = df.loc[df['Size'] == 'Varies with device']

unwanted.shape

df.drop(unwanted.index,inplace = True)
df["Size"]=df["Size"].astype(str)

df["Size"]=df["Size"].str.replace('M','')

df["Size"]=df["Size"].str.replace('k','')

df["Size"]=df["Size"].str.replace('+','')                                   

df["Size"]=df["Size"].astype(float)
res = df.groupby('Category')['Size'].sum().reset_index()

final=res.sort_values(by="Size",ascending=False)

final.head(10)
res= df.groupby('Category')['Installs'].sum().reset_index()

final=res.sort_values(by="Installs",ascending=False)

final.head(10)
reset = df.groupby('Category')['Rating'].sum().reset_index()

final_reset=reset.sort_values(by="Rating",ascending=False)

final_reset.head(10)
sns.lineplot(x=final['Installs'],y=final_reset['Rating'])

plt.show()
data=final.merge(final_reset,on='Category')
data.head(10)
data.describe()
sns.relplot(x='Rating',y='Installs',data=data,hue="Category")

plt.show()
data.loc[data['Rating']==data['Rating'].max()]

data.loc[data['Installs']==data['Installs'].max()]
data1=data.drop([0,1])
sns.relplot(x='Rating',y='Installs',data=data1,hue="Category")

plt.show()