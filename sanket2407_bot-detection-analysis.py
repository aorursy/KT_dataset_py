import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

import plotly.graph_objs as go

import warnings

warnings.filterwarnings('ignore')

from IPython.core.display import display, HTML

display(HTML("<style>.container {width:100% !important;}</style>"))
df=pd.read_csv('../input/botdetection/ibm_data.csv',index_col=0)
df.head(10)
# Let's bring some statistical insights fro data

df.describe()
# Let's check the inforamtion and type

df.info()
# check fro null values

df.isnull().sum()
import datetime

df['page_vw_ts']=pd.to_datetime(df['page_vw_ts'])
df.page_vw_ts.dt.dayofyear.head()
df['ip_addr']= df['ip_addr'].astype(str)

df['VISIT']= df['VISIT'].astype(int)

df['ENGD_VISIT']= df['ENGD_VISIT'].astype(int)

df['VIEWS']= df['VIEWS'].astype(int)

df['wk']= df['wk'].astype(int)
df['day']=df['page_vw_ts'].dt.weekday
df.head()
#Let's see the trend of year,month and day

df['year']=df.page_vw_ts.dt.year
df.head()
df['month']=df.page_vw_ts.dt.month
df.head()
df.year.value_counts().sort_index().plot()
df.month.value_counts().sort_index().plot()
df.day.value_counts().sort_index().plot()
df.head()
df.corr()
# lets's drop wk,mth,yr as it contins most of the null values and does not play any important role

df.drop('wk',axis=1,inplace=True)
df.drop('mth',axis=1,inplace=True)

df.drop('yr',axis=1,inplace=True)
df.head()
# lets's drop some more columns that are irrelevant to our data means if they dont be in data set it will not affect our analysis.

df.drop('intgrtd_mngmt_name',axis=1,inplace=True)

df.drop('intgrtd_operating_team_name',axis=1,inplace=True)

df.drop('st',axis=1,inplace=True)

df.drop('sec_lvl_domn',axis=1,inplace=True)

df.drop('device_type',axis=1,inplace=True)
df.head()
df.isnull().sum()
df['city'].fillna((df['city'].mode()[0]),inplace=True)
df=df.dropna()
df.isnull().sum()
sns.pairplot(df)
plt.figure(figsize=[10,5])

plt.subplot(1,2,1)

sns.scatterplot(df.VIEWS,df.day,color='r')

sns.scatterplot(df.VIEWS,df.VISIT,color='g')

sns.scatterplot(df.VISIT,df.ENGD_VISIT,color='b')



plt.subplot(1,2,2)

sns.scatterplot(df.VIEWS,df.month,color='r')

sns.scatterplot(df.VIEWS,df.VISIT,color='g')

sns.scatterplot(df.VISIT,df.ENGD_VISIT,color='b')
df.pivot_table(['VISIT','VIEWS','ENGD_VISIT'],('day')).plot(kind='bar')
df.pivot_table(['VISIT','VIEWS','ENGD_VISIT'],('month')).plot(kind='bar')
data=df.pivot_table(['VISIT','VIEWS','ENGD_VISIT','day','month'],('ip_addr'),aggfunc='sum')
data
columns=['VISIT','VIEWS','ENGD_VISIT','day','month']

df1=pd.DataFrame(data[columns])

df1.dropna(inplace=True)
df1
# Let's scale the data fisrt

from sklearn.preprocessing import MinMaxScaler

scaler=MinMaxScaler()

df1.VIEWS=scaler.fit_transform(df1[['VIEWS']])

df1.VISIT=scaler.fit_transform(df1[['VISIT']])

df1.ENGD_VISIT=scaler.fit_transform(df1[['ENGD_VISIT']])

df1.day=scaler.fit_transform(df1[['day']])

df1.month=scaler.fit_transform(df1[['month']])
# using k-means to make 2 cluster groups

from sklearn.cluster import KMeans

km=KMeans(n_clusters=2)
y_pred=km.fit_predict(df1)
df1['cluster']=y_pred
# For plotting the graph of cluster

p=df1[df1.cluster==0]

q=df1[df1.cluster==1]
plt.scatter(p.VISIT,p.day,color='g')

plt.scatter(p.VISIT,p.VIEWS,color='g')

plt.scatter(p.VISIT,p.ENGD_VISIT,color='g')



plt.scatter(q.VISIT,q.day,color='b')

plt.scatter(q.VISIT,q.VIEWS,color='b')

plt.scatter(q.VISIT,q.ENGD_VISIT,color='b')



plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],color='red')
plt.scatter(p.VISIT,p.month,color='g')

plt.scatter(p.VISIT,p.VIEWS,color='g')

plt.scatter(p.VISIT,p.ENGD_VISIT,color='g')



plt.scatter(q.VISIT,q.month,color='b')

plt.scatter(q.VISIT,q.VIEWS,color='b')

plt.scatter(q.VISIT,q.ENGD_VISIT,color='b')



plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],color='red')
data['BOT']=y_pred
data[data.BOT==1]