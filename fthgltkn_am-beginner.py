# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
avocado=pd.read_csv('../input/avocado.csv')

avocado=pd.DataFrame(avocado)
kolonlar=['Unnamed: 0','Date','AveragePrice','Total_Volume','4046','4225','4770', 'Total_Bags', 'Small_Bags', 'Large_Bags', 'XLarge_Bags', 'type',

       'year', 'region']

avocado.columns=kolonlar

avocado.columns
avocado.head()

avocado.tail()
#avocado.Date.value_counts()

#avocado.AveragePrice.value_counts()
#Date_list=list(avocado['Date'])

#Average_Price_list=list(avocado['AveragePrice'])

#avocado.info()
year_list=list(avocado['year'])

Average_Price_ratio=[]

for i in year_list:

    x=avocado[avocado['year']==i]

    Average_Price_rate=sum(x.AveragePrice)/len(x)

    Average_Price_ratio.append(Average_Price_rate)

data=pd.DataFrame({'year_list':year_list,'Average_Price_ratio':Average_Price_ratio})

new_index=(data['Average_Price_ratio'].sort_values(ascending=False)).index.values

sorted_data1=data.reindex(new_index)



plt.figure(figsize=(7,7))

sns.barplot(x=sorted_data1['year_list'],y=sorted_data1['Average_Price_ratio'])

plt.xlabel('year_list')

plt.ylabel('Average_Price_ratio')

plt.title('Average_Price_ratio')

plt.show()
year_list=list(avocado['year'])

Total_Volume_ratio=[]

for i in year_list:

    x=avocado[avocado['year']==i]

    Total_Volume_rate=sum(x.Total_Volume)/len(x)

    Total_Volume_ratio.append(Total_Volume_rate)

data=pd.DataFrame({'year_list':year_list,'Total_Volume_ratio':Total_Volume_ratio})   

new_index=(data['Total_Volume_ratio'].sort_values(ascending=False)).index.values

sorted_data2=data.reindex(new_index)  



plt.figure(figsize=(7,7))

sns.barplot(x=sorted_data2['year_list'],y=sorted_data2['Total_Volume_ratio'])

plt.xlabel('year_list')

plt.ylabel('Total_Volume_ratio')

plt.title('Total_Volume_ratio')

plt.show()
sorted_data1['Average_Price_ratio']=sorted_data1['Average_Price_ratio']/max(sorted_data1['Average_Price_ratio'])

sorted_data2['Total_Volume_ratio']=sorted_data2['Total_Volume_ratio']/max(sorted_data2['Total_Volume_ratio'])

data=pd.concat([sorted_data1,sorted_data2['Total_Volume_ratio']],axis=1)

data.sort_values('Average_Price_ratio',inplace=True)





f,ax1 = plt.subplots(figsize =(5,5))

sns.pointplot(x='year_list',y='Average_Price_ratio',data=data,color='red')

sns.pointplot(x='year_list',y='Total_Volume_ratio',data=data,color='blue')

plt.text(40,0.6,'avocado.AveragePrice',color='red',fontsize = 17,style = 'italic')

plt.text(40,0.55,'avocado.Total_Volume',color='lime',fontsize = 18,style = 'italic')

plt.xlabel('date list')

plt.ylabel('values')

plt.grid()





plt.figure(figsize=(12,5))

plt.title("Distribution Price")

ax = sns.distplot(avocado["AveragePrice"], color = 'r')

plt.show()
avocado.boxplot(column="AveragePrice", by="type")

avocado.boxplot(column="4046", by="type")

avocado.boxplot(column="year", by="type")

plt.show()