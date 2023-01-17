import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
%matplotlib inline
cd = pd.read_excel("KPMG_VI_New_raw_data_update_final.xlsx", sheet_name = 'CustomerDemographic', skiprows =1)
ca = pd.read_excel("KPMG_VI_New_raw_data_update_final.xlsx", sheet_name = 4, skiprows =1)
ts = pd.read_excel("KPMG_VI_New_raw_data_update_final.xlsx", sheet_name = 1, skiprows =1)
nc = pd.read_excel("KPMG_VI_New_raw_data_update_final.xlsx", sheet_name = 2, skiprows =1)
#print(cd.info(), ca.info(), ts.info())
#print(set(ca.customer_id)- set(cd.customer_id))
#print(set(cd.customer_id)- set(ca.customer_id))
temp1 = pd.merge(ca,cd, how = 'outer', on = 'customer_id')
df = pd.merge(ts,temp1, how ='outer', on = 'customer_id')
#df.info()
df.info()
df.columns
rs = df.groupby('transaction_date').transaction_id.count()
#rs.index, rs.values
sns.lineplot(x = rs.index, y = rs.values)
rs = df.groupby('DOB').transaction_id.count()
#rs.index, rs.values
sns.lineplot(x = rs.index, y = rs.values)

df.DOB.describe()
df['age'] = 2020 - df.DOB.apply(lambda x: int(x.strftime('%Y')) if x==x else None)
sns.boxplot(df.age)
df[['online_order','order_status']].apply(lambda x: x.value_counts(dropna=0))
df.iloc[:,[6,7,8,9]].apply(lambda x: x.value_counts(dropna=0))
pd.melt(df.loc[:,['list_price','standard_cost']])
sns.boxplot(x ='variable', y = 'value', data =pd.melt(df.loc[:,['list_price','standard_cost']]))
from datetime import datetime

df.product_first_sold_date[0]

df.product_first_sold_date =pd.to_timedelta(df.product_first_sold_date, unit = 'D')+ pd.to_datetime('1899-12-30')
sns.lineplot(x = df.product_first_sold_date.value_counts().index, y = df.product_first_sold_date.value_counts().values)
df[['country','state','gender']].apply(lambda x: set(x))

df.state = df.state.replace(to_replace ={'New South Wales': 'NSW', 'Victoria':"VIC"})
df.gender = df.gender.replace(to_replace ={'Female': 'F', 'Femal':"F", "Male":"M"})
sns.boxplot(df.past_3_years_bike_related_purchases)
df[['job_industry_category']].apply(lambda x: set(x))
df[['job_industry_category']].apply(lambda x: x.value_counts(dropna=0))
print(df.wealth_segment.value_counts(dropna=0),df.deceased_indicator.value_counts(dropna=0))
df.default
df.owns_car.value_counts(dropna= 0)
sns.boxplot(df.tenure)
dt = pd.DataFrame(list(zip(df.columns,df.isna().sum()/20510*100, df.isna().sum())))
dt
#df[df.transaction_id.notna()]
20000-197-30
df.dropna(subset =['transaction_id','brand','address','DOB','online_order'],axis =0,inplace = True)
df.drop(['last_name','default'], axis =1 , inplace = True)
df.info()
dt = df.drop(['first_name', 'address', 'country', 'DOB', 'postcode', 'job_title'], axis = 1)
dt.info()
df2 =dt.drop(['online_order','order_status','brand','product_line','product_class',
              'product_size','product_first_sold_date'], axis =1)
revenue=df2[['transaction_id','customer_id','list_price']].groupby('customer_id').list_price.sum()
quantity = df2[['transaction_id','customer_id','list_price']].groupby('customer_id').transaction_id.count()
revenue
data = df2.drop(['transaction_id','product_id','list_price','standard_cost','transaction_date'], axis =1)
data.drop_duplicates(inplace = True)

temp = pd.merge(data,revenue, how = 'left', on = 'customer_id')
data = pd.merge(temp,quantity, how = 'left', on = 'customer_id')
data.rename(columns ={'list_price':'revenue','transaction_id': 'quantity'}, inplace= True)
data.set_index('customer_id')
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
mapping = {"N":0,"Y":1,"No":0,"Yes":1}
map2 = {'Mass Customer':0, 'Affluent Customer':1, 'High Net Worth':2}
data.job_industry_category.fillna('unknown',inplace= True)
data.deceased_indicator=data.deceased_indicator.map(mapping)
data.owns_car=data.owns_car.map(mapping)
data.wealth_segment= data.wealth_segment.map(map2)
print(set(data.job_industry_category),set(data.wealth_segment))
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
data.job_industry_category = le.fit_transform(data.job_industry_category)
data.state = le.fit_transform(data.state)
data.gender= le.fit_transform(data.gender)
data.set_index('customer_id',inplace= True)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
%matplotlib inline
y_km = KMeans(n_clusters=5).fit(data)
sum_of_squared_distance = []
K = range(2,10)
for k in K:
    km = KMeans(n_clusters =k)
    model = km.fit(data)
    ssq= model.inertia_
    sum_of_squared_distance.append(ssq)

sns.lineplot(x = K, y = sum_of_squared_distance)
y_km.labels_
data['kmg'] = y_km.labels_
data.groupby('kmg').quantity.count()
data.groupby('kmg').revenue.mean()
data.groupby('kmg').revenue.sum()
data.groupby('kmg').revenue.mean()
data[data.kmg==3].describe()
data[data.kmg==4].describe()
plt.bar(data.kmg, data.revenue)
data
from sklearn.cluster import AgglomerativeClustering
ac = AgglomerativeClustering(n_clusters =3, affinity = 'euclidean', linkage = 'complete')
labels = ac.fit_predict(X)
X= data.drop(['revenue'],axis =1)
y = data.revenue
from sklearn.preprocessing import StandardScaler
X = StandardScaler().fit_transform(X)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.3)
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train,y_train)
y_hat = lm.predict(X_test)
from sklearn import metrics
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_hat)))

