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
import pandas as pd

import glob

import os

df = pd.read_csv('../input/monthly-sales-2019/Sales_January_2019.csv')

df.head()



path =r'../input/monthly-sales-2019'                    

all_files = glob.glob(os.path.join(path, "*.csv"))     # advisable to use os.path.join as this makes concatenation OS independent



df_from_each_file = (pd.read_csv(f) for f in all_files)

concatenated_df   = pd.concat(df_from_each_file, ignore_index=True)

concatenated_df.to_csv('all_data.csv')
df = pd.read_csv('./all_data.csv')

#df.head()

new_data = df.dropna(axis = 0, how ='any') 
new_data.head()

#print(new_data.columns)

Or = new_data[new_data['Order Date'].str [0:2] != 'Or']

new_data['month'] = Or['Order Date'].str[0:2].astype('int')

#new_date['Sale'] = pd.to_numeric(new_data['Price Each']) * pd.to_numeric(new_data['Quantity Ordered'])

new_data_nan = new_data[new_data.isna().any(axis=1)]

new_data = new_data.dropna()



new_data['Price Each'] = pd.to_numeric(new_data['Price Each'])

new_data['Quantity Ordered'] = pd.to_numeric(new_data['Quantity Ordered'])

new_data['Sale'] = new_data['Price Each']* new_data['Quantity Ordered']

new_data.head()
import seaborn as sns

import matplotlib.pyplot as plt

pd = new_data.groupby('month',as_index=False).sum()

sns.barplot(x='month',y='Sale',data=pd)
new_data['city']= new_data['Purchase Address'].apply(lambda x:x.split(",")[1]+' '+(x.split(",")[2]).split()[0])

new_data.head()

pd1 = new_data.groupby('city',as_index=False).sum()

plt.figure(figsize=(15,6))

sns.barplot(x='city',y='Sale',data=pd1)
new_data.head()

new_data['time'] = new_data['Order Date'].apply(lambda x: x.split()[1])

new_data['hour'] = new_data['Order Date'].apply(lambda x: (x.split()[1]).split(":")[0])

pd2 = new_data.groupby('hour',as_index=False).sum()



plt.figure(figsize=(15,6))

sns.barplot(x='hour',y='Sale',data=pd2)

plt.grid()

plt.xlabel('hour')
new_data.head()

#pd3 = new_data.groupby(['Order ID'],as_index=False)

pd3=new_data[new_data['Order ID'].duplicated(keep=False)] 

pd3.head()
pd3['group'] = pd3.groupby('Order ID')['Product'].transform(lambda x:','.join(x))

pd3.head()

pd4= pd3.drop_duplicates(['group','Order ID'], keep='first')

pd4.head()
from itertools import combinations

from collections import Counter

count = Counter()

for row in pd4['group']:

    row_list = row.split(",")

    count.update(Counter(combinations(row_list,3)))

count.most_common(10)

for key,value in count.most_common(10):

    print(key,",",value)
new_data.head()

new_data['Quantity Ordered'] = new_data['Quantity Ordered'].astype('int')

new_data['Price Each'] = new_data['Price Each'].astype('int')

pd5 = new_data.groupby('Product',as_index=False)['Quantity Ordered'].sum()

pd6 = new_data.groupby('Product',as_index=False)['Price Each'].mean()

plt.figure(figsize=(15,6))

plt.xticks(rotation=45)

sns.barplot(x='Product',y='Quantity Ordered',data=pd5)

ax2 = plt.twinx()

sns.lineplot(x='Product',y='Price Each',data=pd6, color="b", ax=ax2)