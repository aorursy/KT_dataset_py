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
df = pd.read_csv('/kaggle/input/data-police-shootings/fatal-police-shootings-data.csv')
df
df.dtypes
year=[]
for i in range(len(df)):
    year.append(df['date'][i].split('-')[0])
year
df['year']=year
df
df1=df.groupby('year')['id'].count()
df1
df1=df1.reset_index()
df1
plt.plot(df1['year'],df1['id'],'g+-')
plt.ylim(ymin=0)

df2=df.groupby('race')['id'].count()
df2
df2=df2.reset_index()
df2
plt.bar(df2['race'],df2['id'])

df3=df.groupby('age')['id'].count()
df3
df3=df3.reset_index()
plt.bar(df3['age'],df3['id'])

df4=df.groupby('state')['id'].count()
df4
df4=df4.reset_index()

df_=df4.sort_values(by=['id'],ascending=False)
df_
df_.head(10)
