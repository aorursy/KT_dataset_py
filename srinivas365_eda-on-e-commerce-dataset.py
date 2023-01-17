# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input/"))

import matplotlib.pyplot as plt

%matplotlib inline



# Any results you write to the current directory are saved as output.
df=pd.read_csv('../input/data.csv',encoding='unicode_escape')
df.head()
df.columns
df.shape
df.Country.value_counts()
df.isnull().sum().sort_values(ascending=False)
df.drop(['CustomerID'],axis=1,inplace=True)
df.dropna(inplace=True)
df.isnull().sum()
df.describe()
df[(df['Quantity']<=0) | (df['UnitPrice']<0)].count()
df=df[df['Quantity']>0]
df=df[df['UnitPrice']>=0]
df.shape
df.describe(include=[np.object])
df.head()
df['total_amount']=df['Quantity']*df['UnitPrice']
df.head()
from datetime import datetime

def toDtObject(value):

    return datetime.strptime(value,'%m/%d/%Y %H:%M')
df['InvoiceDate']=df['InvoiceDate'].apply(toDtObject)
df.head()
def getMonth(value):

    return int(value.strftime('%m'))

def getDay(value):

    return int(value.strftime('%d'))

def getYear(value):

    return int(value.strftime('%Y'))

def getHour(value):

    return int(value.strftime('%H'))

df['month']=df['InvoiceDate'].apply(getMonth)

df['day']=df['InvoiceDate'].apply(getDay)

df['year']=df['InvoiceDate'].apply(getYear)

df['hour']=df['InvoiceDate'].apply(getHour)
df.head()
df['year'].value_counts()
df.InvoiceDate.describe()
df.month.value_counts().plot(kind='bar')
df.day.value_counts(ascending=True).plot(kind='bar')
df.Description.value_counts()[:10]
df.groupby(['Description']).describe().transpose()
df.Country.value_counts().plot(kind='bar')
df['hour'].value_counts().plot(kind='bar')
df[df['month']==11].groupby(['Description']).sum().sort_values(by='total_amount',ascending=False)[:5]