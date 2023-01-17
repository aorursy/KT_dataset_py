# import modules

import numpy as np

import pandas as pd

from datetime import datetime

from datetime import timedelta

import os



import random

from scipy import stats

from sklearn.preprocessing import StandardScaler

from sklearn.cluster import KMeans



import matplotlib as mpl

import matplotlib.pyplot as plt

import seaborn as sns

from mpl_toolkits.mplot3d import Axes3D

sns.set()
# read data

files_csv=[]

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        files_csv.append(os.path.join(dirname, filename))

frame=[]

for i in range(len(files_csv)):

    df_i=pd.read_csv(files_csv[i])

    df_i['month']=files_csv[i][-7:-4]

    frame.append(df_i)

df=pd.concat(frame,ignore_index=True,sort=False)

print("The dataframe has {} rows and {} columns.\n".format(df.shape[0],df.shape[1]))

print("Shown below are the first 3 rows of the dataframe:\n")

pd.set_option('display.max_columns', 100)

display(df.head(3))
# data preparation



# step 1: select customers who purchased

df_sales=df.loc[df.event_type=='purchase',:]



# step 2: drop "category_code", "brand", "product_id", "category_id", and "user_session"

df_sales=df_sales.drop(columns=['category_code','brand','product_id','category_id','user_session'])



# step 3: drop duplicates

df_sales=df_sales.drop_duplicates()



# step 4: convert "event_time" to DateTime format

df_sales['event_time']=pd.to_datetime(df_sales['event_time'],infer_datetime_format=True)



nullcolumns=df_sales.isnull().sum()

nullnumbers=len(nullcolumns[nullcolumns!=0])

print("After data selection and cleansing, the dataframe has {} rows, {} columns, and {} null value.\n".format(df_sales.shape[0],df_sales.shape[1],nullnumbers))

print("Shown below are the first 3 rows of the cleaned dataframe:\n")

display(df_sales.head(3))
# initial data exploration



plt.figure(figsize=(10,8))



# plot the number of customers each day 

plt.axes([0.08, 0.4, 0.87, 0.4])

df_sales_n_user=df_sales.resample("D",on='event_time')['user_id'].size()

df_sales_n_user.plot(kind='line')

plt.xlabel('')

plt.ylabel('customer #')



# plot total sales/month 

plt.axes([0.08,0,0.4,0.32])

a=df_sales.resample('M',on='event_time')['price'].sum().to_frame()

a['month']=['Oct','Nov','Dec',"Jan\n2020", "Feb"]

a['price']=a['price']/1000000

sns.barplot(x='month',y='price',data=a,color="lightsteelblue")

plt.xlabel('month')

plt.ylabel('total sales (million $)')



# plot average spend/customer

plt.axes([0.55,0,0.4,0.32])

df_sales_p_day=df_sales.resample('D',on='event_time')['price'].sum()

df_sales_spent=df_sales_p_day/df_sales_n_user

df_sales_spent.plot(kind='area',color="lightsteelblue")

plt.xlabel('date')

plt.ylabel('average spend/customer ($)');
# group the data by "user_id", and calcualte each customer's recency, frequency, and monetary value



# step 1: calculate "Recency", set Feb 2020 as the reference month, and use "month" as the unit

d=con={"Oct":4,"Nov":3,"Dec":2,"Jan":1,"Feb":0}

df_sales.loc[:,'Recency']=df_sales['month'].map(d)

df_R=df_sales.groupby('user_id')['Recency'].min().reset_index().rename(columns={"0":"Recency"})



# step 2: calculate "Frequency"

df_F=df_sales.groupby('user_id')['event_type'].count().reset_index().rename(columns={"event_type":"Frequency"})



# step 3: calculate "Monetary"

df_M=df_sales.groupby('user_id')['price'].sum().reset_index().rename(columns={"price":"Monetary"})



# step 4: merge "Recency", "Frequency", and "Monetary"

df_RF=pd.merge(df_R,df_F,on='user_id')

df_RFM=pd.merge(df_RF,df_M,on='user_id')



# step 5: remove outliers before K-Means clustering

conditions=np.abs(stats.zscore(df_RFM.loc[:,['Recency','Frequency','Monetary']]) < 3).all(axis=1)

df_RFM2=df_RFM.loc[conditions,:]



df_RFM2.head(3)
# visualize the distribution of "Recency", "Frequency", and "Monetary"

fig,(ax1,ax2,ax3)=plt.subplots(1,3,figsize=(10,4))



# plot "Recency"

ax1.hist(df_RFM2['Recency'],bins=5,color='lightsteelblue')

ax1.set_xticks(np.arange(0,5,1))

ax1.set_xlabel('recency (month)')

ax1.set_ylabel('customer #')



# plot "Frequency"

ax2.hist(df_RFM2['Frequency'],bins=5,color='lightsteelblue')

ax2.set_xlabel('frequency')

ax2.set_ylabel('customer#')



# plot "Monetary"

ax3.hist(df_RFM2['Monetary'],bins=5,color='lightsteelblue')

ax3.set_xlabel('monetary value ($)')

ax3.set_ylabel('customer#')



plt.tight_layout()
# k-means clustering: using recency, frequency, and monetary as clustering varaibles



# step 1: standardize data

df_RFM3=df_RFM2.drop(columns=['user_id'])

X = StandardScaler().fit_transform(df_RFM3)



# step 2: find the optimal number of clusters

SSE=[]

for i in range(1,8,1):

    kmeans=KMeans(n_clusters=i)

    kmeans.fit(X)

    SSE.append(kmeans.inertia_)

sns.set()

plt.plot(range(1,8,1),SSE,marker='o')

plt.xlabel('number of clusters')

plt.ylabel('squared error') 

plt.title('Optimal number of clusters');
# k-means clustering: using recency, frequency, and monetary as clustering varaibles



# step 3: group customers into 4 clusters

random.seed(8)

km=KMeans(n_clusters=4,random_state=0)

km.fit(X)

random.seed(8)

pred=km.predict(X)

df_RFM2=df_RFM2.assign(clusters=pred)



# step 4: visualize the 4 clusters



# step 4_1: data preparation

R=[]

F=[]

M=[]

mycolors=['navajowhite','lightsteelblue','mediumaquamarine','thistle']

cluster_orders=[3,2,0,1]

for i in [0,1,2,3]:

    R.append(df_RFM2.loc[df_RFM2.clusters==cluster_orders[i],'Recency'].values.tolist())

    F.append(df_RFM2.loc[df_RFM2.clusters==cluster_orders[i],'Frequency'].values.tolist())

    M.append(df_RFM2.loc[df_RFM2.clusters==cluster_orders[i],'Monetary'].values.tolist())

    

# step 4_2: 3D scatter plot

fig=plt.figure(figsize=(8,5))

ax=Axes3D(fig)

for i in [0,1,2,3]:

    ax.scatter(R[i], F[i], M[i], c=mycolors[i], marker='o',alpha=0.5,label='cluster '+str(cluster_orders[i]))

ax.set_xlabel('Recency')

ax.set_ylabel('Frequency')

ax.set_zlabel('Monetary Value($)')

ax.set_xlim(0,4)

ax.set_xticks(list(range(5)))

ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

plt.show()
# replace k-means cluster names with more meaningful names

d1={0:"New Customers", 2:"Potential Loyalist", 1: "At-Risk", 3:"Loyal Customers"}

df_RFM2.loc[:,"segments"]=df_RFM2.loc[:,"clusters"].map(d1)



# calculate the number of customers, median recency, median frequency, 

# and average customer spend in each customer segment

df_RFM3=df_RFM2.groupby('segments').agg(Recency=('Recency',np.median),Frequency=('Frequency',np.median),MonetarySum=('Monetary',np.sum),size=("clusters",'size'))

df_RFM3.loc[:,'Sales/Customer']=round(df_RFM3.loc[:,'MonetarySum']/df_RFM3.loc[:,'size'])

df_RFM3=df_RFM3.astype({'Sales/Customer':int}).reset_index()



# visualize

plt.figure(figsize=(10,4))

seg_names=['Loyal Customers','Potential Loyalist','New Customers','At-Risk']



# plot the number of customers in each segment

sns.set_style("white")

plt.axes([0, 0, 0.38, 0.9])

seg=df_RFM2.groupby('segments').size().to_frame().rename(columns={0:'number of customers'}).reset_index()

sns.barplot(x='number of customers',y='segments',data=seg,order=seg_names,palette=mycolors)

for i in [0,1,2,3]:

    number=int(seg.loc[seg.segments==seg_names[i],'number of customers'])

    x_pos=round(number,-2)

    plt.text(x_pos,i,number)

plt.ylabel("")

sns.despine()



# plot recency, frequency, and average spend/customer of the 4 segments

plt.axes([0.5,0,0.42,0.9])

sns.scatterplot(x='Recency',y='Frequency',hue='segments',hue_order=seg_names,palette=mycolors,size='Sales/Customer',sizes=(200,1000),legend=False,data=df_RFM3)

plt.ylim(0,35)

plt.xticks(list(range(5)))

plt.text(1,29,'average "Loyal Customer": $146')

plt.text(2,16,'average "Potential Loyalist": $72')

plt.text(0,6,'average "New Customer": $24')

plt.text(3,6,'average "At-Risk": $24')

plt.xlabel('Median Recency (month)')

plt.ylabel('Median Frequency')

sns.despine()
# explore the relationship between customers' purchase probability in Feb 2020 and their Recency,Frequency,

# and Monetary in previous months



# step 1: calculate recency, Frequency, and Monetary in Oct 2019-Jan 2020

df_sales1=df_sales.loc[df_sales.month!='Feb',:].copy()

d={"Oct":3,"Nov":2,"Dec":1,"Jan":0}

df_sales1.loc[:,'Recency']=df_sales1.loc[:,'month'].map(d)

df_sales1_R=df_sales1.groupby('user_id')['Recency'].min().reset_index()

df_sales1_F=df_sales1.groupby('user_id')['event_type'].count().reset_index().rename(columns={'event_type':'Frequency'})

df_sales1_RF=pd.merge(df_sales1_R,df_sales1_F,on='user_id')

df_sales1_M=df_sales1.groupby('user_id')['price'].sum().reset_index().rename(columns={'price':"Monetary"})

df_sales2=pd.merge(df_sales1_RF,df_sales1_M,on='user_id')

                   

# step 2_1: find out customers who made purchases in Feb 2020

df_sales_feb_buyers=df_sales.loc[df_sales.month=='Feb','user_id'].unique().tolist()



# step 2_2: combine step 1 and step 2 results and remove outliers

df_sales2.loc[:,'Buy']=np.where(df_sales2['user_id'].isin(df_sales_feb_buyers),1,0)

conditions=np.abs(stats.zscore(df_sales2[['Recency','Frequency','Monetary']]) < 3).all(axis=1)

df_sales2=df_sales2.loc[conditions,:]

print("Shown below are the first 3 rows of the cleaned dataframe:\n")

display(df_sales2.head(3))
# Step 3 and 4: calculate and visualize the relationship between the probability of purchasing and RFM 

sns.set()

plt.figure(figsize=(12,4))



# plot purchase probability and Recency 

plt.axes([0,0,0.25,0.8])

df_Buy_R=df_sales2.groupby('Recency').agg(Number=('Buy','count'),Buy=('Buy','sum'))

df_Buy_R['Probability']=df_Buy_R['Buy']/df_Buy_R['Number']

plt.scatter(x=df_Buy_R.index,y=df_Buy_R.Probability)

plt.xlim(-0.1,4)

plt.xticks(np.arange(0,4,1))

plt.xlabel('Recency(month)')

plt.ylabel('probability of purchase')



# plot purchase probability and Frequency

plt.axes([0.32,0,0.25,0.8])

df_Buy_F=df_sales2.groupby('Frequency').agg(Number=('Buy','count'),Buy=('Buy','sum'))

df_Buy_F['Probability']=df_Buy_F['Buy']/df_Buy_F['Number']

plt.scatter(x=df_Buy_F.index,y=df_Buy_F.Probability,alpha=0.5)

plt.xlabel('Frequency')

plt.ylabel('probability of purchase')



# plot purchase probability and Monetary

plt.axes([0.63,0,0.25,0.8])

df_Buy_M=df_sales2.groupby('Monetary').agg(Number=('Buy','count'),Buy=('Buy','sum'))

df_Buy_M['Probability']=df_Buy_M['Buy']/df_Buy_M['Number']

plt.scatter(x=df_Buy_M.index,y=df_Buy_M.Probability,alpha=0.5)

plt.xlabel('Monetary Value ($)')

plt.ylabel("probability of purchase");