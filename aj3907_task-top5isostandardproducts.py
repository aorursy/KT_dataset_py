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
df = pd.read_csv('/kaggle/input/isotc213/ISO_TC 213.csv')
df.head()
df.drop(columns=['corrected_version'])
df['Status'].unique()
import matplotlib.pyplot as plt
count_df = df.groupby('Status')['price_CF'].count().reset_index()
plt.bar(count_df['Status'],count_df['price_CF'])
n_pages_max = df['Number of pages'].max()

df['Number of pages'] /= n_pages_max
status_coded = df['Status'].factorize()

status_coded[1]
s = [200*(n+0.00000000000001) for n in df['Number of pages']]

colors = ['b', 'g', 'y' ,'r'] #corresponding to 4 status

c = [colors[n] for n in status_coded[0]]
plt.figure(figsize=[10,8])

plt.scatter(df.index, df['price_CF'], s=s, color= c)

plt.xticks([])

plt.xlabel('ISO Products')

plt.ylabel('price_CF')

plt.show()
task_cols = ['title', 'Number of pages', 'price_CF']
df['Number of pages'] *= n_pages_max
task_df = df[df['Status']=='Published'][task_cols]
task_df.head()
task_df['price_CF_to_npages_ratio'] = task_df['price_CF']/task_df['Number of pages']
mean_price_CF_to_npages_ratio = task_df['price_CF_to_npages_ratio'].mean()

mean_price_CF_to_npages_ratio
task_df[task_df['price_CF_to_npages_ratio']>mean_price_CF_to_npages_ratio][task_cols].sort_values(by=['price_CF'], ascending = False).head()