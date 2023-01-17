# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt
import seaborn as sns
#plt.rcParams['font.sans-serif'] = ['SimHei'] #用来正常显示中文标签
#plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号
%matplotlib inline
df = pd.read_csv('../input/BlackFriday.csv')
df.info()
df.head(4)
df.isnull().sum()
user = df[['User_ID', 'Gender', 'Age', 'Occupation', 'City_Category',
       'Stay_In_Current_City_Years', 'Marital_Status']].drop_duplicates()
tmp = df.groupby('User_ID')['Purchase'].agg({'count','sum'}).rename(columns={'count':'buycount','sum':'buysum'})
user = pd.merge(user,tmp,on='User_ID')
product = df[['Product_ID','Product_Category_1','Product_Category_2','Product_Category_3']].drop_duplicates()
tmp = df.groupby('Product_ID')['Purchase'].agg({'count','sum'}).rename(columns={'count':'productcount','sum':'productsum'})
product = pd.merge(product,tmp,on='Product_ID')
product['price'] = product['productsum']/product['productcount']
buylist = df[['User_ID','Product_ID','Purchase']]
df.groupby('Gender')['User_ID'].count().plot.bar(title='Gender')
df.groupby('Gender')['User_ID'].count().plot.bar()
user['Age'].value_counts().sort_index(ascending=False).plot.barh(title='Age')
user.head(4)
user.pivot_table(index='Age',columns='Gender',values='buysum',aggfunc='mean').plot.bar(title='Age & Gender')
user['Occupation'].value_counts().plot.bar(title='Occupation')
user['City_Category'].value_counts().plot.bar(title='City_Category')
user['Stay_In_Current_City_Years'].value_counts().plot.bar(title='Stay_In_Current_City_Years')
user['Marital_Status'].value_counts().plot.bar(title='Marital_Status')
product.groupby('Product_Category_1')['productsum'].sum().sort_values(ascending=False).plot.bar(title='产品销售额')
product.groupby('Product_Category_1')['productcount'].sum().sort_values(ascending=False).plot.bar(title='产品销量')
product.groupby('Product_Category_2')['productsum'].sum().sort_values(ascending=False).plot.bar(title='产品销售额')
product.groupby('Product_Category_2')['productcount'].sum().sort_values(ascending=False).plot.bar(title='产品销量')
product.groupby('Product_Category_3')['productsum'].sum().sort_values(ascending=False).plot.bar(title='产品销售额')
product.groupby('Product_Category_3')['productcount'].sum().sort_values(ascending=False).plot.bar(title='产品销量')
df.groupby('Product_ID',as_index=False)['Purchase'].sum().sort_values('Purchase').reset_index()['Purchase'].cumsum().plot()
plt.plot([3500*0.8,3500*0.8],[0,5017668378],c='r')
plt.plot([0,3500],[5017668378*0.2,5017668378*0.2],c='g')

top10=df.groupby('Product_ID')['Purchase'].sum().nlargest(10).index
df[df.Product_ID.isin(top10)].groupby('Age')['Purchase'].sum().plot.bar()
fig,(ax1,ax2) = plt.subplots(1,2,figsize=(10,4))
df.pivot_table(index='Age',columns='Gender',values='Purchase',aggfunc='sum').plot.bar(ax=ax1)
df.pivot_table(index='Age',columns='Gender',values='Purchase',aggfunc='mean').plot.bar(ax=ax2)
#ax1.xticks(rotation=45)
ax1.set_title('不同年龄不同性别累计消费情况')
ax2.set_title('不同年龄不同性别人均消费情况')
tmp = df.groupby(['Age','Product_ID'],as_index=False).sum().sort_values(['Age','Purchase'],ascending=[1,0])
tmp2 = tmp.groupby('Age')[['Age','Product_ID','Purchase']].head(5)
tmp3=tmp2.pivot_table(index='Age',columns='Product_ID',values='Purchase',aggfunc='sum')

f,ax = plt.subplots(figsize=(10,4))
cmap = sns.cubehelix_palette(start = 1, rot = 3, gamma=0.8, as_cmap = True)
sns.heatmap(tmp3, cmap = cmap, linewidths = 5, ax = ax)
ax.set_yticklabels(tmp3.index,rotation=0)
plt.show()
tmp = df.groupby(['Age','Product_Category_1'],as_index=False).sum().sort_values(['Age','Purchase'],ascending=[1,0])
tmp2 = tmp.groupby('Age')[['Age','Product_Category_1','Purchase']].head(5)
tmp3=tmp2.pivot_table(index='Age',columns='Product_Category_1',values='Purchase',aggfunc='sum')

f,ax = plt.subplots(figsize=(10,4))
cmap = sns.cubehelix_palette(start = 1, rot = 3, gamma=0.8, as_cmap = True)
sns.heatmap(tmp3, cmap = cmap, linewidths = 5, ax = ax)
ax.set_yticklabels(tmp3.index,rotation=0)
plt.show()
tmp = df.groupby(['City_Category','Product_ID'],as_index=False).sum().sort_values(['City_Category','Purchase'],ascending=[1,0])
tmp2 = tmp.groupby('City_Category')[['City_Category','Product_ID','Purchase']].head(5)
tmp3=tmp2.pivot_table(index='City_Category',columns='Product_ID',values='Purchase',aggfunc='sum')

f,ax = plt.subplots(figsize=(10,4))
cmap = sns.cubehelix_palette(start = 1, rot = 3, gamma=0.8, as_cmap = True)
sns.heatmap(tmp3, cmap = cmap, linewidths = 5, ax = ax)
ax.set_yticklabels(tmp3.index,rotation=0)
plt.show()
tmp = df.groupby(['Stay_In_Current_City_Years','Product_ID'],as_index=False).sum().sort_values(['Stay_In_Current_City_Years','Purchase'],ascending=[1,0])
tmp2 = tmp.groupby('Stay_In_Current_City_Years')[['Stay_In_Current_City_Years','Product_ID','Purchase']].head(5)
tmp3=tmp2.pivot_table(index='Stay_In_Current_City_Years',columns='Product_ID',values='Purchase',aggfunc='sum')

f,ax = plt.subplots(figsize=(10,4))
cmap = sns.cubehelix_palette(start = 1, rot = 3, gamma=0.8, as_cmap = True)
sns.heatmap(tmp3, cmap = cmap, linewidths = 5, ax = ax)
ax.set_yticklabels(tmp3.index,rotation=0)
plt.show()
tmp = df.groupby(['Marital_Status','Product_ID'],as_index=False).sum().sort_values(['Marital_Status','Purchase'],ascending=[1,0])
tmp2 = tmp.groupby('Marital_Status')[['Marital_Status','Product_ID','Purchase']].head(5)
tmp3=tmp2.pivot_table(index='Marital_Status',columns='Product_ID',values='Purchase',aggfunc='sum')

f,ax = plt.subplots(figsize=(10,4))
cmap = sns.cubehelix_palette(start = 1, rot = 3, gamma=0.8, as_cmap = True)
sns.heatmap(tmp3, cmap = cmap, linewidths = 5, ax = ax)
ax.set_yticklabels(tmp3.index,rotation=0)
plt.show()
df.head(3)
top20user = user[user.buysum>user['buysum'].quantile(q=0.8)]['User_ID']
user[user.User_ID.isin(top20user)].groupby('City_Category')['User_ID'].count().sort_values(ascending=False).head(10).plot.bar()
df[df.User_ID.isin(top20user)].groupby('Product_ID')['Purchase'].sum().sort_values(ascending=False).head(10).plot.bar()
user[user.User_ID.isin(top20user)].groupby('Age')['User_ID'].count().sort_values(ascending=False).head(10).plot.bar()
user[user.User_ID.isin(top20user)].groupby('Occupation')['User_ID'].count().sort_values(ascending=False).head(10).plot.bar()
df.head(3)