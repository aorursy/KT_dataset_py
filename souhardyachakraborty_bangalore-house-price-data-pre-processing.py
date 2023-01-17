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
df=pd.read_csv('/kaggle/input/bengaluru-house-price-data/Bengaluru_House_Data.csv')
df
df.isnull().sum()
df.info()
df.describe().T
df['availability'].replace('Ready To Move',1)
df
def change(df):

    if df['availability']=='Ready To Move':

        return 1

    else:

        return 0

df['availability']=df.apply(change,axis=1)
df
df['location'].value_counts()
df[df['location'].isnull()]
df.dropna(subset=['location'],inplace=True)

df.isnull().sum()
import seaborn as sns

import matplotlib.pyplot as plt
sns.scatterplot(df['bath'],df['price'])
sns.scatterplot(df['balcony'],df['price'])
df


df[df['size'].isnull()]
df.dropna(subset=['size'],inplace=True)
df['new_size']=df['size'].str.replace('\D+','').astype(int)
df
df.drop(['size'],axis=1,inplace=True)
df
df.shape
df.info()
df['sqft']=df['total_sqft'].str.replace('\D+','').astype(int)
df
df.info()
df.drop(['total_sqft'],axis=1,inplace=True)
df
df['area_type'].value_counts()
df['area_type']=df['area_type'].astype('category')
def change(df):

    if df['area_type']=='Super built-up  Area':

        return 1

    elif df['area_type']=='Built-up  Area':

        return 2

    elif df['area_type']=='Plot  Area':

        return 3

    elif df['area_type']=='Carpet  Area':

        return 4

df['area_type']=df.apply(change,axis=1)
df
df.isnull().sum()
#fill bath and balcony nans with corr in size



#do location check as per instruction on wp

df[df['bath'].isnull()]
df['new_size'].value_counts()
import seaborn as sns

sns.boxplot(df['bath'])
sns.boxplot(df['balcony'])
sns.boxplot(df['new_size'])
df[df['new_size']==43]
df[df['new_size']==27]
new=df[df['new_size']<20]

new['bath']
new
mean=new['bath'].mean()
new['bath'].fillna(int(mean)+1,inplace=True)

    

    
df.isnull().sum()
mean=new['balcony'].mean()
new['balcony'].fillna(int(mean)+1,inplace=True)
new.info()
new['balcony']=new['balcony'].astype('int')

new['bath']=new['bath'].astype('int')
new.info()
new['society'].value_counts()
new[new['society']=='Sryalan']['price'].mean()
new[new['society']=='PrarePa']['price'].mean()
new[new['society']=='GrrvaGr']['price'].mean()
new[new['society']=='Prtates']['price'].mean()
new['society'].fillna('No Society',inplace=True)
new.isnull().sum()
new
#Conclusions:

#1:  Changed the area type column from objects to numerical values(Onehot encoding have to done later)

#2:  Changed the availablity column from objects to numerical values(categorical)(0=Not ready,1=Ready to move)

#3:  Dropped the size column which was object data type and added a new column 'new_size' with int data type

#4:  As per the prediction society column doesnot affect the prices so changed the NaN value to 'No society'

#5.  Total sqft was in range sometimes So, dropped it and made a new column sqft with exact values

#6:  bath and balcony changed from float to int

#7:  Tried to replace the balcony and bath NaN values with respect to area_type but NOT SUCCESSFUL

#8:  Instead filled the NaN values with mean of bath and balcony respectively

#9:  There was 1 NaN in location column dropped it

#10: There was two outliers as shown in the the graph with 27 bath and 40 bath dropped those two rows

#11: BATH AND BALCONY IS HIGHLY  LINEAR(except the two outlier in bath) and BATH IS ALSO LINEAR WITH new_size

#12: There is no NaN value in dataset





##  Data is ready to go only if one hot encoding is done
