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
import pandas as pd 

df= pd.read_csv('../input/google-play-store-apps/googleplaystore.csv')

df.head(10)
df.info()
df.Rating.mean()

# mean of the apps in the platform.
df.Rating.count()

# out of 9367 apps listed on the platform.

df.drop(df.Rating.idxmax(),inplace=True)

# removing the irrelevant values from the dataframe. ie unusual values.

# cleaning the dataset.

df.Rating.max()
df.Rating.min()

# what is minimum rating in the Rating column
# how many apps are present between 0 and 1.

k=[x for x in df['Rating'] if x>0 and x<=1]

print(len(k))
# how many apps are present between 1 and 2.

l=[x for x in df['Rating'] if x>1 and x<2]

print(len(l))
# how many apps are present between 2 and 3.

four=[x for x in df['Rating'] if x >=2 and x<3]

print(len(four))
## how many apps are present between 3 and 4.

four=[x for x in df['Rating'] if x >=3 and x<4]

print(len(four))
## how many apps are present between 4 and 5.

four=[x for x in df['Rating'] if x >=4 and x<5]

print(len(four))
# how many categories are present.

[k for k in df.groupby('Category').nunique()]     

print(len(k))
# how many apps are present in different categories.

df.groupby('Category').App.nunique()
# print the category with apps count.

for temp in df.groupby('Category').App.nunique():

    print(temp)
# find the highest categorical apps that are being released into the app store.

category = dict(df.groupby('Category').App.nunique())

print(max(category,key=category.get))
# find the least categorical apps that are being released into the app store.

print(min(category,key=category.get))
# how many apps released in to the app store in category of medical.

print(category['MEDICAL'])
# how many apps are being released in to the app store in category of DATING

print(category['DATING'])
# as we can see that the category and genres are same we can remove the genres section as it is repeated.

df.drop(['Genres'],axis=1,inplace=True)
df.head()
# print the average rating of the apps for each category ?

df.groupby('Category').mean()
# print the highest average rating app category ?

df.groupby('Category').mean().idxmax()
# print the lowest average rating app category ?

df.groupby('Category').mean().idxmin()
df['Installs'].replace(regex=True,inplace=True,to_replace=r'\D',value=r'')

df['Installs']
df.head()
df.info()
df['Installs']=pd.to_numeric(df['Installs'])

df.groupby('Category').Installs.mean()
mean_installs=dict(round(df.groupby('Category').Installs.mean(),5))

print(mean_installs)
# sort the mean installs of the apps in the app store.

sorted(mean_installs.items(),key = lambda item:item[1])
# which category has the highest number of average installs

df.groupby('Category').Installs.mean().idxmax()
# which category has the lowest number of average installs

df.groupby('Category').Installs.mean().idxmin()
# which category has the highest number of installs

df.groupby('Category').Installs.sum().idxmax()
# which category has the lowest number of installs

df.groupby('Category').Installs.sum().idxmin()