# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as pl


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
from scipy.stats import pointbiserialr
from scipy.stats import pearsonr
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

data = pd.read_csv("/kaggle/input/online-news-popularity-dataset/OnlineNewsPopularityReduced.csv")

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data.info()
#1
shares = data['shares']
Weekdays = data.columns.values[31:38]
Day=data[data['shares']>=shares]
Day_shares = Day[Weekdays].sum().values

fig = plt.figure(figsize = (15,5))
plt.bar(np.arange(len(Weekdays)), Day_shares, width = 0.4)

plt.xticks(np.arange(len(Weekdays)), Weekdays)
plt.ylabel("Count of shares", fontsize = 15)
    
plt.tight_layout()
plt.show()
#2
a = pearsonr(data['shares'], data['n_tokens_title'])
print(a)

shares = data['shares']
temp_data = data[data['shares'] <= shares]
sns.scatterplot(x='n_tokens_title',y='shares', data=temp_data)
#3
a = pearsonr(data['num_imgs'], data['shares'])
b = pearsonr(data['num_videos'], data['shares']) 
print(a,b)
#4
fig = plt.subplots(figsize=(10,10))
a = sns.countplot(x='is_weekend',data=data)
#5
a=pearsonr(data['shares'], data['n_tokens_content'])
print(a)
sns.scatterplot(x='n_tokens_content',y='shares', data=data)
a = ['n_tokens_content', 'num_keywords', 'num_videos', 'num_imgs',  'shares']
sns.heatmap(data[a].corr(method='spearman'));
#3.1
shares = data['shares']
temp_data = data[data['shares'] <= shares]
sns.lmplot(x='num_imgs', y='shares', data=temp_data)
#3.2  num_videos
shares = data['shares']
temp_data = data[data['shares'] <= shares]
sns.lmplot(x='num_videos', y='shares', data=temp_data)
pearsonr(data['shares'], data['num_keywords'])

temp_data = data[data['shares'] <= shares]
sns.lmplot(x='num_keywords', y='shares', data=temp_data)
pointbiserialr(data['n_unique_tokens'], data['data_channel_is_bus'])
columns_category=data.columns.values[13:19]
shares=data[data['shares']>=100]
shares_category = shares[columns_category].sum().values

fig = pl.figure(figsize = (15,5))
pl.bar(np.arange(len(columns_category)), shares_category, width = 0.4)

pl.xticks(np.arange(len(columns_category)), columns_category)

pl.ylabel("Count", fontsize = 12)
pl.xlabel("Category", fontsize = 12)
    
pl.tight_layout()
pl.show()