# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

from collections import Counter

%matplotlib inline



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data2016=pd.read_csv("../input/world-happiness/2016.csv")
# In 2016 World Happiness Rank and others...
data2016.info()

data2016.head()
data2016.Region.value_counts()
# Happiness Scores According Region

ListofRegion=list(data2016.Region.unique())

Regionscore=[]

for i in ListofRegion:

    x=data2016[data2016['Region']==i]

    happiness=sum(x['Happiness Score'])/len(x)

    Regionscore.append(happiness)

data=pd.DataFrame({'ListofRegion':ListofRegion,

                  'Regionscore':Regionscore})

newdata=(data["Regionscore"].sort_values(ascending=False)).index.values

sorted_data=data.reindex(newdata)



plt.figure(figsize=(15,10))

sns.barplot(x=sorted_data['ListofRegion'],y=sorted_data['Regionscore'])

plt.xticks(rotation=45)

plt.xlabel('List Of Region',fontsize=20)

plt.ylabel('Region Scores',fontsize=20)

plt.show()

# Most happy 20 Country 

sortdata=data2016.iloc[:20,:]

plt.figure(figsize=(15,15))

sns.barplot(x=sortdata["Country"],y=sortdata["Happiness Score"],palette=sns.cubehelix_palette(20))

plt.xticks(rotation=60)

plt.xlabel("List of Country",fontsize=15)

plt.ylabel("Happiness Score",fontsize=15)

plt.show()
df16=data2016.iloc[:20,:]

df16.head()

f,ax=plt.subplots(figsize=(15,25))

sns.barplot(x=df16["Lower Confidence Interval"],y=df16.Country,color='blue',alpha=0.8,label='Lower Confidence Interval')

sns.barplot(x=df16["Upper Confidence Interval"],y=df16.Country,color='grey',alpha=0.8,label='Upper Confidence Interval')

sns.barplot(x=df16["Family"],y=df16.Country,color='purple',alpha=0.8,label='Family')

ax.legend(loc='upper right',frameon=True)

plt.ylabel('Countries')

plt.show()

data2016.head()
# Freedom Top 40 Country

top10country=data2016.loc[:40,:]

plt.figure(figsize=(30,10))

sns.barplot(x=top10country.Country,y=top10country.Freedom)

plt.xticks(rotation=60)

plt.xlabel('Countries',fontsize=17)

plt.ylabel('Freedom Rates',fontsize=17)

plt.show()
data2016.head()
data2019=pd.read_csv("../input/world-happiness/2019.csv")

data2019.head()
# 2016 Generosity and 2019 Social Support

truncateddata=data2019.iloc[:40,:]

f,ax1=plt.subplots(figsize=(15,15))

sns.pointplot(x='Country or region',y='Social support',data=truncateddata,color='green',alpha=0.8)

sns.pointplot(x='Country or region',y='Generosity',data=truncateddata,color='brown',alpha=0.8)

plt.xticks(rotation=60)

plt.grid()
data2016.head()


# Happiness Score and Generosity

sns.jointplot(x=data2016["Happiness Score"],y=data2016["Generosity"],data=data,kind='kde',size=7)

plt.show()
data2016.Region.value_counts()
data2016.Region.dropna(inplace=True)

labels=data2016.Region.value_counts().index

sizes=data2016.Region.value_counts().values

explode=[0,0,0,0,0,0,0,0,0,0]

colors=["grey",'blue','brown','yellow',"purple","green",'black','orange','red','pink']

plt.figure(figsize=(10,10))

plt.pie(sizes,explode=explode,colors=colors,labels=labels,autopct='%1.1f%%')

plt.show()
data2016.head()
sns.lmplot(x='Economy (GDP per Capita)',y='Family',data=data2016)

plt.show()
data2016.corr()
plt.figure(figsize=(10,10))

sns.heatmap(data2016.corr(),annot=True,linewidth=0.5,linecolor='grey',fmt='0.1f')

plt.show()
data2016.head(15)
Happiness_Scores=['Good Happiness Score' if i>=7.0 else 'Bad Happiness Score' for i in data2016["Happiness Score"]]

newdata=pd.DataFrame({'Good Happiness Score':Happiness_Scores})

sns.countplot(newdata['Good Happiness Score'])

plt.show()
sns.countplot(y=data2016.Region)

plt.xlabel('Number of Countries')
Freedom_situations=['High Freedom' if i>=0.5 else 'Low Freedom' for i in data2016.Freedom]

newdata=pd.DataFrame({'High Freedom':Freedom_situations})

sns.countplot(newdata['High Freedom'])

plt.show()
# Which continent is best ??

bestcont=data2016.iloc[:20,:]

sns.countplot(y=bestcont.Region)

plt.show()