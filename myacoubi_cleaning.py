import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import random



color = sns.color_palette()



%matplotlib inline

train = pd.read_csv("../input/train_2016_v2.csv")

properties=pd.read_csv("../input/properties_2016.csv")

train.columns.values

properties.columns.values
train_df.head()

properties.shape
train.info()
properties.info()
missing = properties.isnull().sum(axis=0).reset_index()

missing.columns = ['name', 'count']

missing = missing.ix[missing['count']>0]

tot=properties.shape[0]

missing['ratio']=missing['count'].apply(lambda x:x/tot)

missing = missing.sort_values(by='count')

missing
train_df = pd.merge(train, properties, on='parcelid', how='left')

train_df.head(10)
train_df['transaction_year'] = train_df["transactiondate"].dt.year

cnt_srs = train_df['transaction_year'].value_counts()

plt.figure(figsize=(12,6))

sns.barplot(cnt_srs.index, cnt_srs.values, alpha=0.8, color=color[3])

#plt.xticks(rotation='vertical')

#plt.xlabel('year of transaction', fontsize=12)

#plt.ylabel('Number of Occurrences', fontsize=12)

#plt.show()

train_df.shape
missing = train_df.isnull().sum(axis=0).reset_index()

missing.columns = ['name', 'count']

missing = missing.ix[missing['count']>0]

tot=train_df.shape[0]

missing['ratio']=missing['count'].apply(lambda x:x/tot)

missing = missing.sort_values(by='count')

missing
dropcolumns=missing.ix[missing["ratio"]>0.80]["name"].tolist()

dropcolumns
train_df=train_df.drop(dropcolumns,axis=1)
train_df.shape
train_df.shape[1]+len(dropcolumns),train.shape[1]+properties.shape[1]-1 #parcelid est dans les deux tableaux
missing = train_df.isnull().sum(axis=0).reset_index()

missing.columns = ['name', 'count']

missing = missing.ix[missing['count']>0]

tot=train_df.shape[0]

missing['ratio']=missing['count'].apply(lambda x:x/tot)

missing = missing.sort_values(by='count')

missing
train_df.info()
train_df.columns.values
def fill(data):

    for col in data.columns.values:

        if data[col].dtype!='O' and data[col].count()<data.shape[0]:

            dt=data[col].value_counts().to_frame()

            dt.columns=['count']

            dt["percentage"]=dt['count'].apply(lambda x: x/data[col].count())

            liste=[]

            for i in range(dt.shape[0]):

                if dt.iloc[i]['percentage']>0.4:

                    liste=liste+[dt.index[i] for j in range(10*int(dt.iloc[i]['percentage']))]

            if liste:

                data[col]=data[col].fillna(random.choice(liste))

fill(train_df) 

print("ok")
train_df.info()