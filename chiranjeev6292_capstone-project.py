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
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random as r
import sklearn
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth
import matplotlib.pyplot as plt
#from tabulate import tabulate
#from prettytable import PrettyTable
from astropy.table import Table
import re
df= pd.read_csv("../input/train.csv", encoding='latin1',sep=',')
df.head()
df.price.describe()
plt.subplot(1, 2, 1)
(df['price']).plot.hist(bins=50, figsize=(12, 6), edgecolor = 'white', range = [0, 250])
plt.xlabel('price', fontsize=12)
plt.title('Price Distribution', fontsize=12)
plt.subplot(1, 2, 2)
np.log(df['price']+1).plot.hist(bins=50, figsize=(12,6), edgecolor='white')
plt.xlabel('log(price+1)', fontsize=12)
plt.title('Price Distribution', fontsize=12)
df.groupby(['shipping'])['shipping'].count()
shipping_fee_by_buyer = df.loc[df['shipping'] == 0, 'price']
shipping_fee_by_seller = df.loc[df['shipping'] == 1, 'price']
fig, ax = plt.subplots(figsize=(18,8))
ax.hist(shipping_fee_by_seller, color='r', alpha=1.0, bins=50, range = [0, 100],
       label='Price when Seller pays Shipping')
ax.hist(shipping_fee_by_buyer, color='b', alpha=0.7, bins=50, range = [0, 100],
       label='Price when Buyer pays Shipping')
plt.xlabel('price', fontsize=12)
plt.ylabel('frequency', fontsize=12)
plt.title('Price Distribution by Shipping Type', fontsize=15)
plt.tick_params(labelsize=12)
plt.legend()
plt.show()
print('The average price is {}'.format(round(shipping_fee_by_seller.mean(), 2)), 'if seller pays shipping');
print('The average price is {}'.format(round(shipping_fee_by_buyer.mean(), 2)), 'if buyer pays shipping')
fig, ax = plt.subplots(figsize=(18,8))
ax.hist(np.log(shipping_fee_by_seller+1), color='r', alpha=1.0, bins=50,
       label='Price when Seller pays Shipping')
ax.hist(np.log(shipping_fee_by_buyer+1), color='b', alpha=0.7, bins=50,
       label='Price when Buyer pays Shipping')
plt.xlabel('log(price+1)', fontsize=12)
plt.ylabel('frequency', fontsize=12)
plt.title('Price Distribution by Shipping Type', fontsize=15)
plt.tick_params(labelsize=12)
plt.legend()
plt.show()
print('There are', df['category_name'].nunique(), 'unique values in category name column')
df['category_name'].value_counts()[:10]
a=set(df['Transaction_ID'])
a
a = [a for a in a if str(a) != 'nan']
a
temp=pd.DataFrame(a)
temp['Transaction_ID']=pd.DataFrame(a)
temp
unique_name=set(df['name'])
unique_name
len(unique_name)
temp_name=pd.DataFrame(unique_name)
temp_name['name']=pd.DataFrame(unique_name)
temp_name
temp_name['name_id']=df.loc[0:len(unique_name),['train_id']]
temp_name
temp_final=pd.DataFrame()
temp_final['name']=temp_name['name']
temp_final['name_id']=temp_name['name_id']
temp_final
df3=df.merge(temp_final,on='name',how='left')
df3['name_id'] = df3['name_id'].astype(str)
df3
df1_temp=df3.groupby('Transaction_ID').name_id.agg([('name_id', ', '.join)])
df1_temp
df1_temp= df1_temp.name_id.str.split(",",expand=True) 
df1_temp.replace(to_replace=[None], value=' ', inplace=True)
df1_temp
sample=np.array(df1_temp)
sample.shape
sample1=sample.reshape(104,14999)
sample1.shape
sample1
sample1.shape
te1 = TransactionEncoder()
te_ary1 = te1.fit(sample1).transform(sample1)
fp_tree1 = pd.DataFrame(te_ary1, columns=te1.columns_)
fp_tree1
fp1=fpgrowth(fp_tree1, min_support=0.96)
fp1


t=np.array(fp1)
t1=Table(t)
b=str(t1)
b=list(b)
sample=b.str.contains(pat=',')
q=[]
regex = re.compile(r'\d*\, \d*[,}]')
#'\d*\, \d*[,][ ][\d*]'
matches = regex.finditer(b)
#matches = regex.sub(r' ', b)
for match in matches:
    #q.append(match)
    print(match)

fp2=np.array(matches)
fp2
#fp1=np.delete(fp1,1,axis=1)
sample=','
t=np.array(fp1)
t1=Table(t)

#for i in t1:
#    if sample in i:
#        print("hello")
#        #t1[i][1]=t1[i][1]
#    else:
#        print("byebye")
#        #t1[i][1]=0

     
b=str(t)
b=b.replace('[','')
b=b.replace(']','')
b
#t1
q=[]
#regex = re.compile(r'\([01]\.\d*\, frozenset\(\{')
regex = re.compile(r'[01]\.\d* frozenset\(\{')
 #[][01] frozenset\(\{   
#regex = re.compile(r'frozenset\(\{\d*\, \d*')
#matches = regex.finditer(b)
matches = regex.sub(r' ', b)
#for match in matches:
    #q.append(match)
matches
#type(matches)
matches=matches.replace(')','')
matches=matches.replace('\n','')
matches=matches.replace('}','')
matches
#t2 = Table(np.delete(t1, 1, axis=0))
#t2
#b=str(t1)
#b
#b
matches=list(matches)
matches=np.array(matches)
type(matches)
matches
print(matches.shape)
