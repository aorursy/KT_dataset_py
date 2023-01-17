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
df = pd.read_csv("/kaggle/input/google-play-store-apps/googleplaystore.csv")
df
df.dtypes
df.isna().any()
df.isna().sum()
df['Installs'].unique()
instll=[]
for i in range(len(df)):
    instll.append(df['Installs'][i].split('+')[0])
instll
df['Installs']=instll
df
install=[]
for i in range(len(df)):
    install.append(df['Installs'][i].replace(',',''))
df['Installs']=install
df

df[df['Installs']=='Free']
df=df.drop(10472)
df
df['Installs'].unique()
df['Installs']=df['Installs'].apply(int)
df.dtypes
df['Installs'].max()
new_df=df[df['Installs']==df['Installs'].max()]
new_df
new=new_df.groupby('App')['Installs'].sum()
new
new=new.reset_index()
new
new=new.sort_values(by=['Installs'],ascending=False)
new
new_=new.head(10)
new_
import matplotlib.pyplot as plt
plt.bar(new_['App'],new_['Installs'])
plt.xticks(rotation=90)
plt.title('Top 10 Apps with highest no of installs')
plt.xlabel('Apps')
plt.ylabel('No of installs')
