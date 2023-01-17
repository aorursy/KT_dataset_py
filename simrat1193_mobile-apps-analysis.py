# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/AppleStore.csv")
df.head()
desc_df = pd.read_csv("../input/appleStore_description.csv")
desc_df.shape
merged_df = pd.merge(df, desc_df[['id', 'app_desc']], on='id')
merged_df.head()
# fig, ax = plt.subplots(figsize=(12, 7))
# sns.heatmap(merged_df.select_dtypes(exclude=['object']).corr(), square=True, cmap="YlGnBu", annot=True);
# # plt.show()
merged_df.head(5)
print("Maxiumum price of the app", max(merged_df['price']))
paid_apps = merged_df[merged_df['price'] > 0]

plt.figure(figsize=(10,10))
plt.subplot(2,1,1)
plt.style.use('fivethirtyeight')
plt.hist(paid_apps['price'], log=True)
plt.title('Price distribution of the apps')
plt.ylabel('Frequency (in log scale)')
plt.xlabel('Price')

plt.subplot(2,1,2)
sns.stripplot(data=paid_apps, y='price', size=6, orient='h', jitter=True)
plt.show()
merged_df['prime_genre'].unique()
# Top Five Paid Apps
top_five_paid_apps = paid_apps.groupby(['prime_genre'])['prime_genre'].count() \
.sort_values(ascending=False)[:5].index.tolist()
plt.figure(figsize=(25,15))
plt.subplot(1,1,1)
plt.style.use('fast')
sns.violinplot(data=paid_apps[(paid_apps.price < 50) & (paid_apps.prime_genre.isin(top_five_paid_apps))],
               y='price', x='prime_genre', scale='count', linewidth=2, vertical=True)
plt.show()
pd.options.mode.chained_assignment = None
paid_apps["size_mb"] = np.log(paid_apps["size_bytes"]/1000000)
sns.set(style="white", color_codes=True)
sns.jointplot("size_mb", "price", data=paid_apps[paid_apps.price < 50], kind='kde');