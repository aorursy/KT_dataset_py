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
zomato = pd.read_csv('/kaggle/input/zomato-restaurants-data/zomato.csv',encoding='latin-1')
zomato.info()
zomato.head()
zomato['Rating text'].value_counts()
zomato.drop(columns=['Restaurant ID','Restaurant Name','Is delivering now','Switch to order menu','Price range','Rating color'],axis=1,inplace=True)

zomato.head()
zomato['Country Code'].value_counts()
zomato_new = zomato[zomato['Country Code']==1]
zomato_new.drop(columns=['Country Code','Currency'],axis=1,inplace=True)
zomato_new
zomato_new = zomato_new[zomato_new['City'].isin(['New Delhi','Gurgaon','Noida','Faridabad'])]
zomato_new
zomato_new['Cuisines'].value_counts()
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
zomato_new['City']=encoder.fit_transform(zomato_new['City'])
zomato_new['City'].value_counts()
zomato_new.info()
zomato_new['Has Table booking']=encoder.fit_transform(zomato_new['Has Table booking'])
zomato_new['Has Online delivery']=encoder.fit_transform(zomato_new['Has Online delivery'])
zomato_new['Rating text']=encoder.fit_transform(zomato_new['Rating text'])
zomato_new.head()
zomato_new = pd.get_dummies(zomato_new,columns=['City','Rating text'],drop_first=True)
zomato_new.head()
cuisine = zomato_new.groupby('Cuisines').mean()['Average Cost for two'].reset_index()
cuisine 
zomato_new = zomato_new.merge(cuisine,on='Cuisines')
zomato_new.head()
zomato_new.drop(columns=['Cuisines'],axis=1,inplace=True)
zomato_new.head()
zomato_new.rename(columns={'Average Cost for two_y' : 'Cuisines'},inplace=True)
zomato_new.head()
zomato_new.corr()['Average Cost for two_x']
