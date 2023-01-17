# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
#Cleaning the wrong data



df=pd.read_csv('../input/googleplaystore.csv')

df['Size'][df['App']=='Command & Conquer: Rivals']=77

df['Rating'][df['App']=='Command & Conquer: Rivals']=4

df['Reviews'][df['App']=='Command & Conquer: Rivals']=111091

df['Type'][df['App']=='Command & Conquer: Rivals']='Free'

df['Installs'][df['App']=='Command & Conquer: Rivals']='1,000,000+'

df['Rating'][df['App']=='Life Made WI-Fi Touchscreen Photo Frame']=2.1

df['Reviews'][df['App']=='Life Made WI-Fi Touchscreen Photo Frame']=25

df['Size'][df['App']=='Life Made WI-Fi Touchscreen Photo Frame']=2.3

df['Installs'][df['App']=='Life Made WI-Fi Touchscreen Photo Frame']='1,000+'

df['Content Rating'][df['App']=='Life Made WI-Fi Touchscreen Photo Frame']='Unrated'

df.info()

df.head()
print("No. of categories=" +str(df['Category'].nunique()))

print("No of genres="+str(df['Genres'].nunique()))
print(df['Category'].value_counts())
import matplotlib.pyplot as plt

import seaborn as sns

sns.set()

cols_to_plot=['Category','Installs','Content Rating']

for col in cols_to_plot:

    s=df[col].value_counts()

    sns.countplot(df[col],palette='Set1',saturation=1)

    plt.xticks(rotation=90)

    plt.title('Countplot of '+str(col))

    plt.show()

plt.tight_layout()
sns.countplot(df['Content Rating'],hue=df['Type'])

plt.xticks(rotation=90)

plt.show()

#Paid apps in categories other than 'Everyone' are almost zero
df['Rating'][df['Rating']>5]=np.NaN

plt.hist(df['Rating'])

plt.xlabel('Rating')

plt.ylabel('Count')

plt.title('Histogram depicting variations of rating')

plt.show()
plt.subplot(1,3,1)

sns.distplot(df['Rating'].dropna())

plt.title('Raw data')

#df['Rating'].fillna(df['Rating'].median(),inplace=True)

plt.subplot(1,3,2)

sns.distplot(df['Rating'].fillna(df['Rating'].median()))

plt.title('Filling method: Median')

plt.subplot(1,3,3)

sns.distplot(df['Rating'].fillna(df['Rating'].mean()))

plt.title('Filling method: Mean')

plt.tight_layout()
df['Type'][df['Price']=='Everyone']='Everyone'

df['Price'][df['Price']=='Everyone']=0

df['Price']=df['Price'].str[1:]

df['Price'][df['Price']=='']='0'

df['Price'].unique()
df['Price']=df['Price'].astype(float)
plt.scatter(df['Rating'],df['Price'])

plt.xlabel('Rating')

plt.ylabel('Price')

plt.yscale('symlog')

plt.title('Price vs Rating')