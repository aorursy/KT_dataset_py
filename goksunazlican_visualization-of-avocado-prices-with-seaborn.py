# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

from collections import Counter

%matplotlib inline

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data=pd.read_csv('../input/avocado.csv')

data2=pd.read_csv('../input/avocado.csv')
new_index = (data['AveragePrice'].sort_values(ascending=False)).index.values

sorted_data = data.reindex(new_index)

sorted_data
data.head()
data.info()
data2 = data2[data2.region!='TotalUS']

data2['region'].value_counts()
#data['Total Bags'].value_counts()
area_list=list(data2.region.unique())

area_averageprice_ratio=[]

for i in area_list:

    x=data2[data2['region']==i]

    area_averageprice_rate=sum(x['AveragePrice'])/len(x)

    area_averageprice_ratio.append(area_averageprice_rate)

    data2['AveragePrice']=data2['AveragePrice'].astype(float)

df=pd.DataFrame({'area_list':area_list,'area_averageprice_ratio':area_averageprice_ratio})

new_index=df['area_averageprice_ratio'].sort_values(ascending=False).index.values

sorted_data=df.reindex(new_index)



plt.figure(figsize=(15,10))

sns.barplot(x=sorted_data['area_list'], y=sorted_data['area_averageprice_ratio'])

plt.xticks(rotation= 90)

plt.xlabel('States')

plt.ylabel('Average Price Rate')

plt.title('Average Price Given States')

#data['AveragePrice']=data['AveragePrice'].astype(int)



avgprice_count=Counter(data2.AveragePrice)

most_common_type=avgprice_count.most_common(15)

x,y = zip(*most_common_type)

x,y = list(x),list(y)

plt.figure(figsize=(15,10))

ax= sns.barplot(x=x, y=y,palette = sns.cubehelix_palette(len(x)))

plt.xlabel('Average Price')

plt.ylabel('Frequency')

plt.title('Most Common Average Price')
area_list=list(data2.region.unique())

area_totalvolume_ratio=[]

for i in area_list:

    x=data2[data2.region==i]

    area_totalvolume_rate=sum(x['Total Volume'])/len(x)

    area_totalvolume_ratio.append(area_totalvolume_rate)

df=pd.DataFrame({'area_list':area_list, 'area_totalvolume_ratio':area_totalvolume_ratio})

new_index=df['area_totalvolume_ratio'].sort_values(ascending=True).index.values

sorted_data2=df.reindex(new_index)



plt.figure(figsize=(15,10))

sns.barplot(x=sorted_data2['area_list'], y=sorted_data2['area_totalvolume_ratio'])

plt.xticks(rotation= 90)

plt.xlabel('States')

plt.ylabel('Average Total Volume Rate')

plt.title('Average Total Price Given States')



area_list=list(data2.region.unique())

SmallBags=[]

LargeBags=[]

XLargeBags=[]



for i in area_list:

    x=data2[data2.region==i]

    SmallBags.append(sum(x['Small Bags'])/len(x))

    LargeBags.append(sum(x['Large Bags'])/len(x))

    XLargeBags.append(sum(x['XLarge Bags'])/len(x))

    

f,ax = plt.subplots(figsize = (9,15))

sns.barplot(x=SmallBags,y=area_list,color='green',alpha = 0.5,label='SmallBags' )

sns.barplot(x=LargeBags,y=area_list,color='blue',alpha = 0.7,label='LargeBags')

sns.barplot(x=XLargeBags,y=area_list,color='cyan',alpha = 0.6,label='XLargeBags')

ax.legend(loc='upper right',frameon = True)     # legendlarin gorunurlugu

ax.set(xlabel='Percentage of Races', ylabel='States',title = "Percentage of State's Size of Bags")


area_list=list(data2.region.unique())

sorted_data['area_averageprice_ratio']=sorted_data['area_averageprice_ratio']/max(sorted_data['area_averageprice_ratio'])

sorted_data2['area_totalvolume_ratio']=sorted_data2['area_totalvolume_ratio']/max(sorted_data2['area_totalvolume_ratio'])

new_data=pd.concat([sorted_data,sorted_data2['area_totalvolume_ratio']],axis=1)

new_data.sort_values('area_averageprice_ratio',inplace=True)



f,ax1=plt.subplots(figsize=(20,10))

sns.pointplot(x='area_list',y='area_averageprice_ratio',data=new_data,color='lime',alpha=0.5)

sns.pointplot(x='area_list',y='area_totalvolume_ratio',data=new_data,color='green',alpha=0.5)

plt.text(40,0.3,'average price ratio',color='green',fontsize = 17,style = 'italic')

plt.text(30,0.7,'total volume ratio',color='lime',fontsize = 17,style = 'italic')

plt.xlabel('States',fontsize = 15,color='blue')

plt.ylabel('Values',fontsize = 15,color='blue')

plt.xticks(rotation= 90)



plt.title('Average Price  VS  Total Volume',fontsize = 20,color='blue')

new_data.head()
g = sns.jointplot(new_data.area_averageprice_ratio,new_data.area_totalvolume_ratio,kind="kde",height=7)

plt.savefig("graph.png")

plt.show()
g = sns.jointplot("area_averageprice_ratio","area_totalvolume_ratio",data=new_data,height=5,ratio=5,color="r")
data_SD=data2[(data2.region=='LosAngeles') & (data2.AveragePrice<2)]

data_SD.type.value_counts()






sizes=data_SD['type'].value_counts().values



plt.figure(figsize=(15,15))

plt.pie(sizes,autopct='%1.1f%%',rotatelabels = True)

plt.title('Types ',color='blue',fontsize=15)

sns.lmplot(x="area_averageprice_ratio",y="area_totalvolume_ratio",data=new_data)

plt.show()
sns.kdeplot(new_data.area_averageprice_ratio,new_data.area_totalvolume_ratio,color='magenta',shade=True,cut=3)

plt.show()
data2.head()
pal = sns.cubehelix_palette(2, rot= -.5, dark= .3)

sns.violinplot(data=new_data,palette=pal,inner="points")

plt.show()
new_data.corr()
f,ax=plt.subplots(figsize=(10,10))

sns.heatmap(new_data.corr(),annot=True,linewidths=0.5,linecolor="red",fmt='.1f',ax=ax)

plt.show()
data.head()
sns.boxplot(x="type", y="year" ,data=data, palette="PRGn")

plt.show()
#sns.swarmplot(x="type", y="year" ,data=data, palette="PRGn")

#plt.show()
new_data.head()
sns.pairplot(new_data)

plt.show()
sns.countplot(data.type)

plt.title("type",color = 'blue',fontsize=15)

plt.show()
organic = ['organic' if i=='organic'  else 'conventional' for i in data2.type]

df = pd.DataFrame({'type':organic})

sns.countplot(x=df.type)

plt.ylabel('AveragePrice')

plt.title('type of avocado', color='blue',fontsize=10)