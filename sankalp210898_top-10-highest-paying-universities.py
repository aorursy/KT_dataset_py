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
import matplotlib.pyplot as plt
df=pd.read_csv('/kaggle/input/phd-stipends/csv')
df
df.dtypes
df.isna().sum()
df1=df[pd.notnull(df['Overall Pay'])]
df1
df1['Overall Pay'].isna().any()
df1['Overall Pay'] = df1['Overall Pay'].str.replace('$','').str.replace(',','').astype(float)
df1['Overall Pay']
df1
df2=df1[pd.notnull(df1['University'])]

h_u=df2.sort_values(by = ['Overall Pay'],ascending=False)
h_u
h_u=h_u.head(10)
h_u
plt.bar(h_u['University'],h_u['Overall Pay'],color='green')
plt.xlabel('Universities')
plt.ylabel('Overall Pay')
plt.xticks(rotation=90)
plt.title('Top 10 Highest Paying Universities')
plt.bar(h_u['Department'],h_u['Overall Pay'],color='yellow')
plt.xlabel('Departments')
plt.ylabel('Overall Pay')
plt.xticks(rotation=90)
plt.title('Top 10 Highest Paying Departments')
l_u=df2.sort_values(by=['Overall Pay'])
l_u=l_u.head(10)
l_u
l_u=l_u[['University','Department','Overall Pay']]
l_u['University']
l_u['Department']
h_u['University'].reset_index(drop=True,inplace=True)
h_u['University'].name='Highest_Paying_Uni'
h_u['University']
h_u['Department'].reset_index(drop=True,inplace=True)
h_u['Department'].name='Highest_Paying_Dept'
h_u['Department']
l_u['University'].reset_index(drop=True,inplace=True)
l_u['University'].name='Lowest_Paying_Uni'
l_u['University']
l_u['Department'].reset_index(drop=True,inplace=True)
l_u['Department'].name='Lowest_Paying_Dept'
l_u['Department']
final_df=pd.concat([h_u['University'],h_u['Department'],l_u['University'],l_u['Department']],axis=1)
final_df
