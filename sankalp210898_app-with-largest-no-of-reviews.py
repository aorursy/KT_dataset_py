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
df['Reviews'].isna().any()
df['Reviews'].apply(int)
for i in range(len(df['Reviews'])):
    if 'M' in df['Reviews'][i]:
       df['Reviews'][i]= float(df['Reviews'][i].split('M')[0])*1e6
df['Reviews']=df['Reviews'].apply(int)
new_df=df.groupby('App')['Reviews'].sum()
new_df=new_df.reset_index()
new_df
new_df=new_df.sort_values(by='Reviews',ascending=False)
new_df
new_=new_df.head(10)
new_
import matplotlib.pyplot as plt
plt.bar(new_['App'],new_['Reviews'],color='green')
plt.xlabel('Apps')
plt.ylabel('No of Reviews')
plt.title('Top 10 apps with largest no of reviews')
plt.xticks(rotation=90)
