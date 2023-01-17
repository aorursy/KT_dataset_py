# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('/kaggle/input/madrid-10k-new-years-eve-popular-race/madrid_10k_20191231.csv')
df.head()
df['2.5km_seconds'].isna().sum()
df['5km_seconds'].isna().sum()
df['7.5km_seconds'].isna().sum()
df['total_seconds'].isna().sum()
import seaborn as sns
df['age_category'].unique()
age_compare_times = df[(df['age_category']=='16-19') | (df['age_category']=='20-22') | (df['age_category']=='55+')]
import matplotlib.pyplot as plt
def age_color(ls, color):

    c =[]

    for i in ls:

        if(i=='16-19' or i=='20-22'):

            c.append(color[0])

        elif(i=='55+'):

            c.append(color[1])

        else:

            c.append(color[2])

    return c

        

            



plt.scatter(age_compare_times['id_number'],age_compare_times['total_seconds'], color = age_color(age_compare_times['age_category'],['g','r','w']))
total = len(age_compare_times)
old = len(age_compare_times[age_compare_times['age_category']=='55+'])
young = total - old

young
age_compare_times[age_compare_times['age_category']=='55+']['total_seconds'].mean()
age_compare_times[(age_compare_times['age_category']=='16-19') | (age_compare_times['age_category']=='20-22')]['total_seconds'].mean()
age_compare_times.head()
df[df['2.5km_seconds'].isna()].head()
import numpy as np
df['2550_diff']=df['5km_seconds']-df['2.5km_seconds']

df['5075_diff']=df['7.5km_seconds']-df['5km_seconds']

df['7510_diff']=df['total_seconds']-df['7.5km_seconds']
df[df['7510_diff']<0]
df[(df['7510_diff']<0) & ((df['2.5km_seconds']).isna() | (df['5km_seconds']).isna() | (df['7.5km_seconds'].isna()))]