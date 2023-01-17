import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # visualization

import re

%matplotlib inline
df=pd.read_csv('../input/Hp_laptops.csv',encoding='unicode-escape')
df.head()
df.info()
df.shape
df.drop(['Unnamed: 0'],axis=1,inplace=True)
df.isnull().sum()
df.drop(['display'],axis=1,inplace=True)
df.head()
df.actual_price.dtype
def strtoint(column):

    return column.apply(lambda x:int(x.replace(',','')))
df['actual_price']=strtoint(df['actual_price'])

df['discout']=strtoint(df['discout'])

df['final_price']=strtoint(df['final_price'])
df.head()
df.generation.value_counts()
df.generation.fillna('1.0',inplace=True)

df.generation.isnull().sum()
df.graphic_card.value_counts()[:5]
df.graphic_card.fillna(df.graphic_card.mode()[0],inplace=True)

df.graphic_card.isnull().sum()
df.hard_disk.value_counts()[:5]
df.hard_disk.fillna(df.hard_disk.mode()[0],inplace=True)

df.hard_disk.isnull().sum()
df.included_items.fillna('Not provided',inplace=True)

df.included_items.isnull().sum()
df.ram.fillna(df.ram.mode()[0],inplace=True)

df.included_items.isnull().sum()
df.rating.fillna(df.rating.mean(),inplace=True)

df.rating.isnull().sum()
df[df.processor_company.isnull()]
print(df.processor_type.mode()[0])

print(df.processor.mode()[0])

print(df.processor_company.mode()[0])
df.at[38, 'processor']=df.processor.mode()[0]

df.at[120,'processor']=df.processor.mode()[0]

df.at[147,'processor']=df.processor.mode()[0]

df.processor.fillna(df.processor.mode()[0],inplace=True)

df.processor_company.fillna(df.processor_company.mode()[0],inplace=True)

df.processor_type.fillna(df.processor_type.mode()[0],inplace=True)
df.isnull().sum()
import re

def getSize(value):

    b=re.findall(r'\b\d+',str(value))

    if len(b)>0:

        return b[0]

    return None

df['ram_size']=df['ram'].apply(getSize)
df['ram_size'].value_counts()
df.ram_size.isnull().sum()
df.ram_size=df.ram_size.astype('int')
def getSize(value):

    p=re.compile(r'\b\d+\s\w+')

    val=p.findall(value)

    size=None

    if len(val)>0:

        size=val[0]

        size_val=re.findall(r'\d+',size)[0]

        if 'TB' in size:

            size_val=int(size_val)*1024

        return size_val

    return size
df['hd_size(GB)']=df.hard_disk.apply(getSize)
df['hd_size(GB)'].isnull().sum()
df['hd_size(GB)'].fillna(df['hd_size(GB)'].mode()[0],inplace=True)
df['hd_size(GB)']=df['hd_size(GB)'].astype('int')
df.head()
df.isnull().sum()
df.to_csv('Hp_laptops_new.csv')
df.actual_price.plot(kind='hist')
df[(df.final_price>30000)&(df.final_price<60000) & (df['hd_size(GB)']==1024) & (df['ram_size']==8)]
df.processor_company.value_counts().plot(kind='bar')
df.processor.value_counts().plot(kind='bar')
df.groupby(['processor_company']).mean()
df.os_installed.value_counts()
df.os_installed.value_counts().plot(kind='bar')
df.ram_size.corr(df.actual_price)
df.ram_size.value_counts().plot(kind='pie')
df.hard_disk.value_counts().plot(kind='bar')
df.corr()
df.describe()
df.describe(include=[np.object]).transpose()
print(df[df.actual_price>100000]['Name'].unique())

df[df.actual_price==df.actual_price.max()]