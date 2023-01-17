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
df = pd.read_csv('/kaggle/input/bengaluru-house-price-data/Bengaluru_House_Data.csv')
df_copy = df.copy()
df.shape
df.sample(10)
df.shape
df.info()
df.isnull().sum()
# Let's see what percentage of value in society column are missing



((13320-7818)/13320)*100
df['bath']=df['bath'].fillna(0)

df['balcony']=df['balcony'].fillna(0)
df['bath']=df['bath'].astype('int32')

df['balcony'] = df['balcony'].astype('int32')
df.isnull().sum()
df.info()
df['location'].value_counts().head(20)



# Here I found that 540 location named as "Whitefield" so I keep 1 null value of location as "Whitefield".



df['location']=df['location'].fillna('Whitefield')
df.isnull().sum()
mask = df['size'].str.split(" ",expand=True)

df['size']=mask.drop([1], axis='columns')



# So here I remove BHK,Bedroom etc. from size column and just keep the actual value
df['size']=df['size'].fillna(0)



# Fill nan value of size column with 0.
df['society']=df['society'].fillna('No society')



# Fill nan value of society column with No society
mask1=[i for i in df['availability'] if i not in ['Ready To Move','Immediate Possession']]

df['availability']=df['availability'].replace(mask1,'Available Soon')



# Fill Dates of availability column with Available Soon.
df['bath'] = df['bath'].replace([i for i in df['bath'] if i not in [0,1,2,3,4]],4)



df['bath'].value_counts()



# Deal with bath column
df.info()
# change datatypes



df['availability']=df['availability'].astype('category')

df['size']=df['size'].astype('category')

df['area_type']=df['area_type'].astype('category')

df['price']=df['price'].astype('float32')
df.info()
df['total_sqft'].value_counts().sample(30)
df['total_sqft'].str.findall('[0-9]-[0-9]')
# So after all prossesing we found our Dataset like this



df.sample(30)