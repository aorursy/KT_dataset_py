# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import math

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

plt.style.use("ggplot")
plt.rcParams['figure.figsize'] = [8,8]

# Any results you write to the current directory are saved as output.
df_reviews = pd.read_csv("../input/googleplaystore_user_reviews.csv",dtype={'Price':'float'})
df_store = pd.read_csv("../input/googleplaystore.csv",dtype={'Price':'str'})
df_reviews.head(5)
df_store.head()
df_store.Rating.describe()
df_store.Rating.value_counts()
def rate_cate(x):
    if x >= 1 and x < 2:
        return '1~'
    elif x>=2 and x < 3:
        return '2~'
    elif x >= 3 and x < 4:
        return '3~'
    elif x >= 4 and x < 5:
        return '4~'
    else:
        return '5+'
df_store['rate_cate'] = df_store['Rating'].apply(rate_cate)
Price = df_store['Price']
sns.stripplot('rate_cate',"Price",data = df_store)
for i in range(len(Price)):
    if Price[i] != '0':
        print(Price[i])
var_list = df_store.columns
Price = df_store['Price']
for i in range(len(Price)):
    if '$' in Price[i]:
        Price[i] = Price[i].strip('$')
        
for i in range(len(Price)):
    if Price[i] == 'Everyone':
        Price[i] = '0'
    if Price[i] != '0':
        print(Price[i])
Price = Price.astype('float')
Price.describe()
Price.plot()
df_store[('fl_Price')] = Price
df_store.head(3)
sns.stripplot("rate_cate",'fl_Price',data=df_store,order=['1~','2~','3~','4~','5+'])


df_store.Price.value_counts()
sns.stripplot('rate_cate',df_store.Price.values,data=df_store,jitter=True)


