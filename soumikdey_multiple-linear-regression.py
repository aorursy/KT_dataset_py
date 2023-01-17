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
df=pd.read_csv('/kaggle/input/zomato-restaurants-data/zomato.csv',encoding='latin-1')
df.sample(2)
df.info()
df.shape
df.isnull().sum()
df.drop(columns=['Restaurant ID','Restaurant Name','Is delivering now','Switch to order menu','Price range','Rating color'],axis=1,inplace=True)

df.sample(2)
df.drop(columns=['Address','Locality','Locality Verbose'],inplace=True,axis=1)
df.sample(2)
df.info()
df['Country Code'].value_counts().shape
df=df[df['Country Code']==1]
df.drop(columns=['Country Code','Currency'],inplace=True,axis=1)
df['City'].value_counts().shape
df=df[df['City'].isin(['New Delhi','Gurgaon','Noida','Faridabad'])]
df.sample(2)
#Cuisine m lafra h
from sklearn.preprocessing import LabelEncoder
LR=LabelEncoder()
df['City']=LR.fit_transform(df['City'])
df['City'].value_counts()
df['Has Table booking']=LR.fit_transform(df['Has Table booking'])

df['Rating text']=LR.fit_transform(df['Rating text'])
df.drop(columns=['Has Online delivery'],inplace=True,axis=1)

df=pd.get_dummies(df,columns=['City','Rating text'],drop_first=True)
df.sample(2)
cuisine=df.groupby('Cuisines').mean()['Average Cost for two'].reset_index()
df=df.merge(cuisine,on='Cuisines')
df.sample(2)
df.drop(columns=['Cuisines'],axis=1,inplace=True)
df.sample(2)
df.rename(columns={'Average Cost for two_x':'Cuisines'},inplace=True)
df.sample(2)
df.corr()['Average Cost for two_y']
#Has Table booking dekhna 