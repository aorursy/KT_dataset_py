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
os.listdir("../")
df=pd.read_csv('../input/startup_funding.csv',encoding='utf8')
df.shape
df.columns
#companies in chennai
pd.options.display.max_columns = None
chennai_startups=df[df['CityLocation']=='Chennai']#['InvestorsName'].value_counts()
#changing 'Amount column'to numerical
df['AmountInUSD']=pd.to_numeric(df['AmountInUSD'].str.replace(',',''))
#top 5 cities with more startup
import matplotlib.pyplot as plt
import seaborn as sb
sb.set()
%matplotlib inline
df['CityLocation'].value_counts().head().plot(kind='barh',x='index',y='CityLocation')
#top 5 investors, invested more money
top_inv=df.groupby('InvestorsName')['AmountInUSD'].sum().sort_values(ascending=False).head()
top_inv.plot(kind='barh')
#top 5 investors invested in more companies
top_inv=df.loc[(~df['AmountInUSD'].isnull()),'InvestorsName'].dropna().value_counts().head().index.tolist()
temp=df.loc[df['InvestorsName'].isin(top_inv),['InvestorsName','AmountInUSD']]

no_of_cmp=temp.groupby('InvestorsName')['AmountInUSD'].agg({'Amount':'sum','No of companies':'count'}).reset_index().sort_values(by='No of companies',ascending=False)
no_of_cmp
no_of_cmp.plot(x='InvestorsName',y='No of companies',kind='barh')
df
df[df['StartupName'].str.contains('ticketnew',case=False)]
