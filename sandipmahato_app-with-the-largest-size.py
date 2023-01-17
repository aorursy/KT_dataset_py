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
df=pd.read_csv('/kaggle/input/google-play-store-apps/googleplaystore.csv')
df.head()
df['Size']=df['Size'].apply(lambda x:str(x).replace('M',''))

df['Size']=df['Size'].apply(lambda x:str(x).replace('K',''))

df['Size']=df['Size'].apply(lambda x:str(x).replace('B',''))

df['Size']=df['Size'].apply(lambda x:str(x).replace('G',''))
df.Size = pd.to_numeric(df.Size, errors='coerce')
df['Size']
df['Size'].max()
df[df['Size']==100.0]
df['Size'].isnull().sum()
df['Size'].fillna(df['Size'].mean(),inplace=True)
df['Size']
df['Size'].isnull().sum()
ans=df[['Size','App']][df['Size']==df['Size'].max()]
ans