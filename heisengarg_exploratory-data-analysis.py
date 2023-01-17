import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
df=pd.read_hdf("../input/train.h5")
df.shape
df.head()
#df.describe()
df.isnull().any()
#Super slow method to find all the rows with no NaN values in any of the columns



#count=0

#for x in range(len(df)-1710750):

#    if not any(df.isnull().values[x]):

#        count=count+1
df_clean=df.dropna(axis='index',how='any')

df_clean.shape
df_clean.shape[0]/df.shape[0]*100
def count_missing(df):

    stats={}

    for x in range(len(df.columns)):        

        stats[df.columns[x]]=df[df.columns[x]].isnull().sum()/len(df[df.columns[x]])*100

    return stats
res=count_missing(df)
plt.figure(figsize=(10,25))

plt.barh(range(0,111),res.values(),align='center')

plt.yticks(range(0,111),(res.keys()))

plt.autoscale()

plt.show()
low_err={k:v for (k,v) in res.items() if v<0.5}

del low_err['id']

del low_err['timestamp']

del low_err['y']

print((low_err),len(low_err))
l=list(low_err.keys())

for i in range(len(low_err)):

    plt.figure(figsize=(15,5))

    plt.scatter(y=df[l[i]],x=df['timestamp'])

    plt.title(l[i],fontsize=15)

    plt.xlabel('Timestamp Value')

    plt.ylabel('Value')

    plt.xlim(0,1900)
for i in range(len(low_err)):

    plt.figure(figsize=(15,5))

    #plt.scatter(y=df[l[i]],x=df['timestamp'])

    sns.distplot(df[l[i]].dropna().values)  #Removed NaN values for now

    plt.title(l[i],fontsize=15)

    plt.xlabel('Value')

    plt.ylabel('Frequency')

    
plt.figure(figsize=(30,20))

sns.heatmap(df_clean.corr(method='pearson', min_periods=1),cmap='RdYlGn')
plt.figure(figsize=(20,20))

sns.heatmap(df_clean.corr(method='spearman', min_periods=1),cmap='RdYlGn')

plt.figure(figsize=(10,25))

uni_cor=df_clean.apply(lambda x: x.corr(df["y"],method='spearman'),axis=0)

uni_cor=uni_cor.drop('y')

uni_cor.sort_values(inplace=True)

plt.barh(range(0,len(uni_cor)),uni_cor,align='center')

plt.yticks(range(0,len(uni_cor)),list(df_clean.columns.values))

plt.autoscale()

plt.show()