import pandas as pd

import numpy as np

import pandas as pd

data = pd.read_csv("../input/Suicides in India 2001-2012.csv",sep=",")

data.count()
data_all_india = data.loc[data['State']=='Total (All India)',:]

data_ut = data.loc[data['State']=='Total (Uts)',:]

data_state = data.loc[data['State']=='Total (States)',:]
data.drop(data[data['State']=='Total (All India)'].index,inplace =True)

data.drop(data[data['State']=='Total (States)'].index,inplace =True)

data.drop(data[data['State']=='Total (Uts)'].index,inplace =True)

data.reset_index(drop=True,inplace=True)

data.head()
pivot_state = pd.pivot_table(data,values='Total',index='State',aggfunc=sum)

pivot_type = pd.pivot_table(data,values='Total',index='Type',aggfunc=sum)

pivot_state['percent'] = pivot_state['Total']/sum(pivot_state['Total'])*100

pivot_state.reset_index(inplace=True)

pivot_type.reset_index(inplace=True,)

pivot_state.sort_values(by='Total',inplace=True,ascending=False)

pivot_type.sort_values(by='Total',inplace=True,ascending=False)
pivot_type[-20:]
import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

fig,ax = plt.subplots(figsize=(8,8))

sns.barplot(data=pivot_state,y='State',x='Total',palette="Blues_d")
fig,ax = plt.subplots(figsize=(18,18))

sns.barplot(data=pivot_type,y='Type',x='Total',palette="Blues_d")
data
k=data.groupby(by=['Gender','Age_group'],as_index=False)['Total'].sum()
k
sns.barplot(x="Age_group", y="Total",hue='Gender',data=k)
top_20 = list(pivot_type[:20]['Type'])
l=[]

for i in top_20:

   # print('**************************************')

  #  print(i)

   # print(data.loc[data['Type']==i].groupby(by=['Gender','Age_group'],as_index=False)['Total'].sum())

    l.append(data.loc[data['Type']==i].groupby(by=['Gender','Age_group'],as_index=False)['Total'].sum())
l[1]
for i in range(20):

    plt.figure()

    plt.title(top_20[i])

    sns.barplot(x="Age_group", y="Total",hue='Gender',data=l[i],)

    
data.head()
data.Year.value_counts()


la=[]

for i in top_20:

   # print('**************************************')

  #  print(i)

  #  print(data.loc[data['Type']==i].groupby(by=['Type','Year'],as_index=False)['Total'].sum())

    la.append(data.loc[data['Type']==i].groupby(by=['Type','Year'],as_index=False)['Total'].sum())
l = data.groupby(by=['Type','Year'],as_index=False)['Total'].sum()
for i in range(20):

    plt.figure()

    plt.title(top_20[i])

    sns.barplot(x="Year", y="Total",data=la[i])