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
data=pd.read_csv("../input/2015.csv")
data.head()
data.info()
print('for happiness score')
data['Happiness Score'].describe()

region_list=list(data.Region.unique())
average_score=[]

for i in region_list:
    x=data[data.Region==i]
    average_score.append(sum(x['Happiness Score'])/len(x))
df1=pd.DataFrame({'region_list':region_list,'average_score':average_score})

new_index1=(df1.average_score.sort_values(ascending=False)).index.values
sorted_data1=df1.reindex(new_index1)

plt.figure(figsize=(15,10))
sns.barplot(x='region_list',y='average_score',data=df1,color='pink',alpha=1)
plt.xticks(rotation=90)
plt.xlabel('Region',fontsize=14)
plt.ylabel('Happiness Score',fontsize=14)
plt.title('Average of Happiness Score According to Region,2015',fontsize=16)
plt.show()
no_hrank=data.drop(['Happiness Rank'],axis=1)
no_hrank.corr()

f,ax=plt.subplots(figsize=(15,10))
sns.heatmap(no_hrank.corr(),annot=True,linewidths=.5,fmt='.1f',ax=ax)
plt.show()
plt.figure(figsize=(20,15))
sns.lmplot('Happiness Score','Health (Life Expectancy)',data=data)
plt.xlabel('Happiness Score',color='green',fontsize=14)
plt.ylabel('Health(Lİfe Expectancy)',color='green',fontsize=14)
plt.title('Happiness Score VS Health(Life Expectancy)',color='green',fontsize=14)
plt.show()
print(data.Region.unique())
data.Region.value_counts()
labels=data.Region.value_counts().index
sizes=data.Region.value_counts().values
pal=sns.color_palette("GnBu_d")
explode=[0,0,0,0,0,0,0,0,0,0]

plt.figure(figsize=(20,20))
plt.pie(sizes,explode=explode,labels=labels,colors=pal,autopct='%1.1f%%')
plt.title('Percentage of Region',fontsize=18,color='black')
plt.figure(figsize=(15,10))
plt.boxplot(data['Happiness Score'])
plt.ylabel('Happiness Score')
plt.title('Box Plot for Happiness Score',color='blue',fontsize=16)

filter1=data['Happiness Score']>data['Happiness Score'].mean()
data[filter1]
filter2=data['Happiness Score']<3.
data['Happiness Score'][filter2]