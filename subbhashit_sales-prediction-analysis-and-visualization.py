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
sample=pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/sample_submission.csv')
sample.head()
data=pd.read_csv("/kaggle/input/competitive-data-science-predict-future-sales/sales_train.csv",index_col='date',parse_dates=['date'])
data.head()
data["item_price"][:'2014-01-01'].plot(figsize=(16,10),legend=True,color='r')
data["item_price"]['2014-01-01':'2015-01-01'].plot(figsize=(16,10),legend=True,color='b')
data["item_price"]['2015-01-01':].plot(figsize=(16,10),legend=True,color='g')
plt.xlabel("Dates")
plt.ylabel("Item price rise")
plt.title("Item pices vs Date")
import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(10,10))
sns.distplot(data['item_price'])
sns.scatterplot(data['item_cnt_day'],data['item_price'])
shops=pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/shops.csv')
shops.head()
items=pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/items.csv')
items.head()
cat=pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/item_categories.csv')
cat.head()
def r(x):
    if 'PC' in x:
        return 'PC'
    elif 'PS2' in x:
        return 'PS2'
    elif 'PS3' in x:
        return 'PS3'
    elif 'PSP' in x:
        return 'PSP'
    elif 'PS4' in x:
        return 'PS4'
    elif 'PSVita' in x:
        return 'PSVita'
    elif 'XBOX 360' in x:
        return 'XBOX 360'
    elif 'XBOX ONE' in x:
        return 'XBOX ONE'
    elif 'Blu-Ray 3D' in x:
        return 'Blu-Ray 3D'
    elif 'Blu-Ray 4K' in x:
        return 'Blu-Ray 4K'
    
    else:
        return 'Others'
cat['item_category_name']=cat['item_category_name'].apply(r)
cat['item_category_name'].value_counts()
plt.figure(figsize=(15,10))
sns.countplot(cat['item_category_name'])
sns.heatmap(data.corr(),annot=True,cmap='rainbow')
data.date_block_num.value_counts()
plt.plot(data.date_block_num.value_counts().values,linestyle="--")
data.item_cnt_day.value_counts()