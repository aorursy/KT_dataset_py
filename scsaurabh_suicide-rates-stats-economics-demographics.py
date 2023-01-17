import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns
data = pd.read_csv("../input/master.csv")

data.head()
data.isnull().sum()
data.info()
data.describe()
data.drop(["HDI for year",'country-year'],inplace=True,axis=1)

data.head(3)
sns.set(style="whitegrid")

f, ax = plt.subplots(figsize=(8, 20))

ax = sns.barplot(data.population.groupby(data.country).count(),data.population.groupby(data.country).count().index)

plt.show()
f,ax = plt.subplots(1,1,figsize=(15,4))

ax = sns.countplot(data.generation,palette='rainbow')

plt.show()
f,ax = plt.subplots(1,1,figsize=(15,4))

ax = sns.barplot(x = data.age.sort_values(),y = 'suicides_no',hue='sex',data=data,palette='rainbow')

plt.show()
f,ax = plt.subplots(1,1,figsize=(15,4))

ax = sns.barplot(x = data[data.year > 2000]['year'],y = 'suicides_no',data=data,palette='rainbow')

plt.show()
f,ax = plt.subplots(1,1,figsize=(15,4))

ax = sns.kdeplot(data['suicides/100k pop'])

plt.show()
data_suicide_mean = data['suicides/100k pop'].groupby(data.country).mean().sort_values(ascending=False)

f,ax = plt.subplots(1,1,figsize=(15,4))

ax = sns.barplot(data_suicide_mean.head(10).index,data_suicide_mean.head(10),palette='coolwarm')
data_time = data['suicides_no'].groupby(data.year).count()

data_time.plot(figsize=(20,10), linewidth=2, fontsize=15,color='purple')

plt.xlabel('Year', fontsize=15)

plt.ylabel('No of suicides',fontsize=15)

plt.show()
data_gdp = (data['gdp_per_capita ($)'].groupby(data.year)).sum()

data_gdp.plot(figsize=(20,10), linewidth=2, fontsize=15,color='red')

plt.xlabel('Year', fontsize=15)

plt.ylabel(' Total gdp_per_capita ($)',fontsize=15)

plt.show()
data_suicide = data['suicides_no'].groupby(data.country).sum().sort_values(ascending=False)

f,ax = plt.subplots(1,1,figsize=(6,15))

ax = sns.barplot(data_suicide.head(10),data_suicide.head(10).index,palette='coolwarm')
data_suicide = data['suicides_no'].groupby(data.country).sum().sort_values(ascending=False)

f,ax = plt.subplots(1,1,figsize=(6,15))

ax = sns.barplot(data_suicide.tail(10),data_suicide.tail(10).index,palette='coolwarm')
f, ax = plt.subplots(figsize=(15, 10))

ax = sns.regplot(x='gdp_per_capita ($)', y='suicides/100k pop',data=data)

plt.show()


from mpl_toolkits.mplot3d import axes3d

f, ax = plt.subplots(figsize=(12, 4))

ax = f.add_subplot(111, projection='3d')

ax.scatter(data['gdp_per_capita ($)'], data.year, data['suicides/100k pop'], alpha=0.2, c="blue", edgecolors='none', s=30, label="people") 

plt.title('gdp_per_capita ($), year, suicides/100k pop')

plt.legend(loc=1)

plt.show()
f,ax = plt.subplots(1,1,figsize=(10,10))

ax = sns.heatmap(data.corr(),annot=True)

plt.show()