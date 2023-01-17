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
import pandas as pd 
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt

Bak=pd.read_csv('../input/BreadBasket_DMS.csv')
Bak.head()
Bak.loc[Bak['Item']=='NONE',:].head()
Bak.loc[Bak['Item']=='NONE',:].count()
Bak=Bak.drop(Bak.loc[Bak['Item']=='NONE'].index)
Bak.loc[Bak['Item']=='NONE',:].count()
Bak['Year'] = Bak.Date.apply(lambda x:x.split('-')[0])
Bak['Month'] = Bak.Date.apply(lambda x:x.split('-')[1])
Bak['Day'] = Bak.Date.apply(lambda x:x.split('-')[2])
Bak['Hour'] =Bak.Time.apply(lambda x:int(x.split(':')[0]))
#df = df.drop(columns='Time')
Bak.head()
print('Total number of Items sold at the bakery is:',Bak['Item'].nunique())
print('List of Items sold at the bakery:')
Bak['Item'].unique()
print('List of Items sold at the Bakery:\n')
for item in set(Bak['Item']):
    print(item)
print('Ten Most Sold Items At The Bakery')
Bak['Item'].value_counts().head(10)
fig, ax=plt.subplots(figsize=(16,7))
Bak['Item'].value_counts().sort_values(ascending=False).head(20).plot.bar(width=0.5,edgecolor='k',align='center',linewidth=2)
plt.xlabel('Food Item',fontsize=20)
plt.ylabel('Number of transactions',fontsize=20)
ax.tick_params(labelsize=20)
plt.title('20 Most Sold Items at the Bakery',fontsize=25)
plt.grid()
plt.ioff()
Bak.loc[Bak['Time']<'12:00:00','Daytime']='Morning'
Bak.loc[(Bak['Time']>='12:00:00')&(Bak['Time']<'17:00:00'),'Daytime']='Afternoon'
Bak.loc[(Bak['Time']>='17:00:00')&(Bak['Time']<'20:00:00'),'Daytime']='Evening'
Bak.loc[(Bak['Time']>='20:00:00')&(Bak['Time']<'23:50:00'),'Daytime']='Night'

fig, ax=plt.subplots(figsize=(16,7))
Bak['Daytime'].value_counts().plot.bar(width=0.5,edgecolor='k',align='center',linewidth=2)
plt.xlabel('Time of the Day',fontsize=20)
plt.ylabel('Number of transactions',fontsize=20)
ax.tick_params(labelsize=20)
plt.title('Sales During Different Period of the Day',fontsize=25)
plt.grid()
plt.ioff()
Bak1 = Bak.groupby(['Date']).size().reset_index(name='counts')
Bak1['Day'] = pd.to_datetime(Bak1['Date']).dt.day_name()
#Bak1
plt.figure(figsize=(20,15))
plt.subplot(2,2,1)
ax=sns.boxplot(x='Day',y='counts',data=Bak1,width=0.8,linewidth=2)
plt.xlabel('Day of the Week',fontsize=15)
plt.ylabel('Total Sales',fontsize=15)
plt.title('Sales on Different Days of Week',fontsize=20)
ax.tick_params(labelsize=10)
plt.grid()
plt.ioff()
Bak['Year'] = Bak.Date.apply(lambda x:x.split('-')[0])
Bak['Month'] = Bak.Date.apply(lambda x:x.split('-')[1])
Bak['Day'] = Bak.Date.apply(lambda x:x.split('-')[2])
Bak['Hour'] =Bak.Time.apply(lambda x:int(x.split(':')[0]))
#df = df.drop(columns='Time')
Bak.head()
Bak.loc[Bak.Month == '10', 'Monthly'] = 'Oct'  
Bak.loc[Bak.Month == '11', 'Monthly'] = 'Nov' 
Bak.loc[Bak.Month == '12', 'Monthly'] = 'Dec' 
Bak.loc[Bak.Month == '01', 'Monthly'] = 'Jan' 
Bak.loc[Bak.Month == '02', 'Monthly'] = 'Feb' 
Bak.loc[Bak.Month == '03', 'Monthly'] = 'Mar' 
Bak.loc[Bak.Month == '04', 'Monthly'] = 'Apr' 
#Bak.loc[Bak.Month == '05', 'Monthly'] = 'May' 
#df.loc[df.First_name != 'Bill', 'name_match'] = 'Mis-Match' 
Bak.tail()
fig, ax=plt.subplots(figsize=(16,7))
ax=Bak.groupby('Monthly')['Item'].count().sort_values().plot.bar(width=0.5,edgecolor='k',align='center',linewidth=2)
plt.xlabel('Month',fontsize=20)
plt.ylabel('Number of transactions',fontsize=20)
ax.tick_params(labelsize=20)
plt.title('Sales During Different Months of a Year',fontsize=25)
plt.grid()
plt.ioff()
