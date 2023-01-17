# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from plotly.offline import iplot
import plotly as py
import plotly.tools as tls
import cufflinks as cf
py.offline.init_notebook_mode(connected=True)
cf.go_offline()


df1=pd.read_csv('../input/usa-cers-dataset/USA_cars_datasets.csv')
df1.head()
df1.drop(columns=['Unnamed: 0'],inplace=True,axis=1)
df1.columns
df1.isnull().sum()
df1.dtypes
df1.shape
df1.describe()
df1['Average_price']=df1.groupby('brand')['price'].transform('median')
df1.head()
df2=df1[df1['price']==0]
l1_delete=df1[df1['price']==0].index
df1.drop(l1_delete,inplace=True) 
df2['price']=df2['Average_price']

df3=pd.concat([df1,df2],ignore_index=True)
df3.head() 
df3.groupby('year')['brand'].count()
fig1=plt.figure(figsize=(12,5))
ax1=fig1.add_axes([0,0,1,1])
sns.countplot(x='year',data=df3,ax=ax1) 
#the mean data points that we have for each year(Highy influenced by outliers)
df3.groupby('year')['brand'].count().mean() 
df3.groupby('year')['brand'].count().median() 
df3['Occurence']=df3.groupby('year')['brand'].transform('count')
df4=df3[np.logical_or.reduce([df3['year']==2015,df3['year']==2016,df3['year']==2017,df3['year']==2018,df3['year']==2019,df3['year']==2020])]
df4.groupby('year')['brand'].count()
fig1=plt.figure(figsize=(12,5))
ax1=fig1.add_axes([0,0,1,1])
sns.countplot(x='year',data=df4,ax=ax1)
df4.head()
df5=df4.drop(columns=['Average_price','Occurence'],axis=1)
e1=df4.groupby(['brand'])['model'].count().reset_index()
a1=list(e1[e1['model']>3]['brand'])
a1
for i in a1:
    df4[df4['brand']==i].groupby('model')[['price','Average_price']].mean().iplot(kind='spread',xTitle='Average_price for BRAND',yTitle='Price of each model under the brand',size=5,title='FOR'+' '+i)
q2=df4.groupby(['year','color'])['brand'].count().reset_index()
l1=[2015,2016,2017,2018,2019,2020]

for i in l1:
    q2[q2['year']==i][['color','brand']].iplot(x='color',y='brand',color='red',kind='bar',title='Colors sold in year'+ ' '+str(i))
q3=df4.groupby(['state','color'])['brand'].count().reset_index()
q31=q3['state'].unique()
for i in q31:
    q3[q3['state']==i][['color','brand']].iplot(x='color',y='brand',color='green',kind='barh',title='Colors sold in'+' '+i)
df4['price'].iplot(kind='hist',title="OVERALL DISTRIBUTION OF PRICE")
yr=df4['year'].unique()

g1=df4.groupby('year')['price']
for i,j in g1:
    j.iplot(kind='hist',title='Distribution of price in year'+' '+str(i))
df4['title_status'].value_counts()

df_salvage=df4[df4['title_status']=='salvage insurance']
df_salvage['brand'].value_counts().sort_values().iplot(kind='barh',xTitle='Count',yTitle='Car_Brands',title='CAR BRANDS HAVING SALVAGE INSURANCE')
