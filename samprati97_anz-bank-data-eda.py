import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data = pd.read_excel("/kaggle/input/ANZ synthesised transaction dataset.xlsx")
pd.options.display.max_columns = None
data.head()
data.info()
data.describe()
data.isnull().sum()
data['account'].value_counts()
data['status'].value_counts()
data['first_name'].value_counts()
import seaborn as sns
sns.countplot(data['status'])
data['txn_description'].value_counts()
plt.figure(figsize=(8,5))
sns.countplot(data['txn_description'])
plt.figure(figsize=(8,5))
sns.countplot(x=data['txn_description'], hue='movement',data=data)
sns.countplot(data['card_present_flag'])
plt.figure(figsize=(15,3))
plt.title('Amount Boxplot')
sns.boxplot(data['amount'])
plt.title('Amount plot with bins')
sns.countplot(pd.cut(data['amount'],[0,30,100,1000,4000,9000]), hue='movement',data=data)
plt.figure(figsize=(15,3))
sns.distplot(data['amount'])
sns.jointplot(x='age',y='balance',data=data)
data['balance'].sort_values()
data.sort_values(by='balance')

sns.heatmap(data.corr())
sns.pairplot(data)
data['country'].value_counts()
print(data['status'].value_counts())
print(data['movement'].value_counts())
data.head()
new=data['long_lat'].str.split(" ", n = 1, expand = True)
data['long']=new[0].str.rstrip()
data['lat']=new[1]
data['long'].astype('float64')
data['lat'].astype('float64')
sns.scatterplot(x='long',y='lat',data=data)
data.head()
data[['date','age']]
data['month'] = pd.DatetimeIndex(data['date']).month
data['day'] = pd.DatetimeIndex(data['date']).day
data.head()
plt.title('No. of transations for that month')
sns.countplot(data['month'])
data['month'].value_counts()
plt.figure(figsize=(10,5))
plt.title('Daily Trnsaction count for Debit and Credit')
sns.countplot(x=data['day'],hue='movement',data=data)
group_by_month=data.groupby('month')
group_by_month.get_group(8)
plt.figure(figsize=(25,5))
plt.subplot(1,3,1)
plt.title('October')
sns.countplot(x='day', data=group_by_month.get_group(8))
plt.subplot(1,3,2)
plt.title('November')
sns.countplot(x='day', data=group_by_month.get_group(9))
plt.yticks([])
plt.ylabel("")
plt.subplot(1,3,3)
plt.title('December')
sns.countplot(x='day', data=group_by_month.get_group(10))
plt.yticks([])
plt.ylabel("")
plt.subplots_adjust(wspace=0, hspace=0)
cust = data.groupby('account')
cust=cust.mean().sort_values('balance',ascending=False)
plt.figure(figsize=(25,5))
sns.barplot(x=cust.index, y=cust['balance'])
plt.xticks(rotation=90)
plt.title('Account with highest balance')
sns.countplot(x='month',hue='movement',data=data)
plt.title('no. of Debit and Credit transactions in each month')
sns.countplot(x='status',hue='movement',data=data)
plt.figure(figsize=(10,5))
sns.heatmap(data.isnull())
plt.title("Null Values Heatmap")
data.drop((['bpay_biller_code','merchant_code']),axis=1,inplace=True)
plt.figure(figsize=(10,5))
sns.heatmap(data.isnull())
plt.title("Null Values Heatmap")
data['merchant_state'].value_counts()
data['card_present_flag'].value_counts()
from mpl_toolkits.basemap import Basemap
plt.figure(figsize=(8, 8))
m = Basemap(projection='ortho', resolution=None, lat_0=-20, lon_0=130)
m.bluemarble(scale=0.5);
# fig = plt.figure(figsize=(8, 8))
# m = Basemap(projection='lcc', resolution=None,
#             width=8E6, height=8E6, 
#             lat_0=-20, lon_0=130,)
# m.etopo(scale=0.5, alpha=0.5)

# # Map (long, lat) to (x, y) for plotting
# x, y = m(data.long.values, data.lat.values)
# plt.plot(x, y, 'ok', markersize=5)
# plt.text(x, y, ' Seattle', fontsize=12);

data['lat']=data['lat'].astype('float64')
data['long']=data['long'].astype('float64')
data.head()
data['long'].sort_values()
data['x'] = np.nan
data['y'] = np.nan
data.loc[:, ['x', 'y']] = data[['long', 'lat']].apply(lambda row: list(m(row['long'], row['lat'])), axis=1, result_type='expand').rename({0: 'x', 1: 'y'}).values
data[['lat', 'long', 'x', 'y']]
fig = plt.figure(figsize=(8, 10))
m = Basemap(projection='lcc', resolution=None,
            width=11E6, height=9E6, 
            lat_0=-38, lon_0=121,)
m.etopo(scale=0.5, alpha=1)
plt.scatter(data['x'].values, data['y'].values, marker='v', color='k', label='0');
plt.title('Locations of customers in Autralia')
df1 =data[['age','balance']]
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

scaler.fit(df1[['balance']])
df1['balance'] = scaler.transform(df1[['balance']])

scaler.fit(df1[['age']])
df1['age'] = scaler.transform(df1[['age']])
sns.scatterplot(x='age',y='balance',data=df1)
plt.title('Age vs Balance Scatterplot')
from sklearn.cluster import KMeans
sse = []
k_rng = range(1,10)
for k in k_rng:
    km = KMeans(n_clusters=k)
    km.fit(df1[['age','balance']])
    sse.append(km.inertia_)
plt.xlabel('K')
plt.ylabel('Sum of squared error')
plt.plot(k_rng,sse)
from sklearn.cluster import KMeans

km = KMeans(n_clusters=4,random_state=56)
y_predicted = km.fit_predict(df1[['age','balance']])
y_predicted
df1['cluster']=y_predicted
df1.head()
plt.figure(figsize=(8,5))
df2 = df1[df1.cluster==0]
df3 = df1[df1.cluster==1]
df4 = df1[df1.cluster==2]
df5 = df1[df1.cluster==3]
plt.scatter(df2.age,df2['balance'],color='green',label='Middle Class')
plt.scatter(df3.age,df3['balance'],color='red',label='Newly Employed')
plt.scatter(df4.age,df4['balance'],color='black',label='Rich person')
plt.scatter(df5.age,df5['balance'],color='blue',label='Old person')
plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],color='purple',marker='*',label='centroid') 
plt.title('Customer Clustering on the basis of Age and Account Balance')
plt.legend()
data['cluster']=df1['cluster']
data.head()
data[data['cluster']==3]
data['cluster'] = data['cluster'].map({0:'Middle Class',1:'Newly Employed',2:'Rich person',3:'Old person'})
data.head()
data.loc[data['cluster']=='Rich person'].sort_values(by='balance')['first_name'].value_counts()
dfx=data.loc[data['cluster']=='Rich person'].sort_values(by='balance')
dfx['first_name'].value_counts()
dfy=data.loc[data['cluster']=='Newly Employed'].sort_values(by='balance')
dfz=data.loc[data['cluster']=='Middle Class'].sort_values(by='balance')
dfw=data[data['cluster']=='Old person'].sort_values(by='balance')
dfw
plt.figure(figsize=(18,10))
plt.subplot(2,2,1)
plt.title('Rich person')
sns.countplot(dfx['first_name'],order=dfx['first_name'].value_counts().index)
plt.ylabel('No. of transactions')
plt.subplot(2,2,2)
plt.xticks(rotation=90)
plt.ylabel('No. of transactions')
plt.title('Newly Employed')
sns.countplot(dfy['first_name'],order=dfy['first_name'].value_counts().index)
plt.ylabel('No. of transactions')
plt.subplot(2,2,3)
plt.xticks(rotation=90)
plt.title('Middle Class')
sns.countplot(dfz['first_name'],order=dfz['first_name'].value_counts().index)
plt.ylabel('No. of transactions')
plt.subplot(2,2,4)
plt.title('Old person')
sns.countplot(dfw['first_name'],order=dfw['first_name'].value_counts().index)
plt.ylabel('No. of transactions')
plt.subplots_adjust(wspace=0.1, hspace=0.5)
plt.legend()
data['cluster'].value_counts()
balance_df = data[['account','first_name','balance','cluster']]
account_group=balance_df.groupby(['account','first_name','cluster'])
new_df=account_group.mean()
plt.figure(figsize=(25,4))
sns.barplot(x=new_df.index,y=new_df['balance'])
plt.xticks(rotation=90)
new_df.reset_index(inplace=True)
new_df.iloc[43:48]
new_df.drop(new_df.index[46],inplace=True)
new_df.reset_index(inplace=True)
new_df.iloc[43:48]
plt.figure(figsize=(25,4))
plt.xticks(rotation=90)
# plt.xticks(new_df.index, new_df.first_name,rotation=90)  
sns.scatterplot(x=new_df['account'],y=new_df['balance'],hue=new_df['cluster'])
plt.figure(figsize=(25,4))
plt.xticks(new_df.index, new_df.first_name,rotation=90)  
sns.scatterplot(x=new_df['account'],y=new_df['balance'],hue=new_df['cluster'])
plt.title("Each customer's balance with respect to their cluster")
df1=pd.get_dummies(data=data, columns=['gender'],drop_first = True)
data1=df1[(df1['txn_description']=='PAY/SALARY')]
data1
data2=df1[(df1['txn_description']!='PAY/SALARY')]
data2
data3 = data1[['age','balance','gender_M','amount']]
data3
X=data3[['age','balance','gender_M']]
y=data3[['amount']]
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(X,y)
reg.score(X,y)
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor()
rf.fit(X,y)
rf.score(X,y)
X_test = data2[['age','balance','gender_M']]
y_pred=rf.predict(X_test)
data2['salary']=y_pred
data2
data5=data2.groupby(['account','first_name']).mean()
data5