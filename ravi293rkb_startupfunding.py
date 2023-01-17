# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data=pd.read_csv('../input/startup_funding.csv')
#data['Date']=pd.to_datetime(data['Date'])
                 


'''def temp(v):
    try:
        return pd.to_datetime(v.replace('.','/').replace('//','/'))
    except: 
        print(v)'''
#data['Date']=data['Date'].apply(lambda v: temp(v))
data['Date']=pd.to_datetime(data['Date'].str.replace('.','/').str.replace('//','/'))
data['month_year']=data['Date'].dt.strftime('%Y-%m')
data['amount']=data['AmountInUSD'].str.replace(',','').astype(float)
print(data[['Date','month_year','amount']].head())
#data['Date'].value_counts()
data.groupby(['month_year']).size().plot.bar(figsize=(15,5))
data.describe(include='all')
x=data['IndustryVertical'].value_counts()/ data.shape[0]*100
x.head(10).plot.bar()

x1=data['CityLocation'].value_counts()/ data.shape[0]*100
x1.head(10).plot.bar()
data.groupby(['CityLocation'])['amount'].mean().head(10).plot.bar()