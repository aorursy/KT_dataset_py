# import packages.



%matplotlib inline

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.cluster import KMeans

from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import silhouette_score

## data path

train_path = "../input/ashrae-energy-prediction/train.csv"

df = pd.read_csv(train_path)

df.head
## usage is in BKTU: convert meter_readings into kwh by multiplying by 0.2931 from BKTU

df.loc[:,'meter_reading'] *= 0.2931 
## sample data: subset meter == 0 (electricity)

#train = train.filter(train['meter'] == 0)

df = df[(df['meter'] == 0) & (df['timestamp'] <= "2016-01-31 00:00:00" )]

#df.head
## feature engineering

 ## extract day and month from datestamp column

df = df.assign(**{'hour': pd.to_datetime(df['timestamp']).dt.hour,

                  'day': pd.to_datetime(df['timestamp']).dt.dayofyear,

                  'day_name': pd.to_datetime(df['timestamp']).dt.day_name()

                 

                 }).drop('meter', 1)



#'day': pd.to_datetime(df['timestamp']).dt.day,
df.describe()
df[(df['meter_reading'] >=8626)]
## drop duplicates if there is any

df = df.drop_duplicates()
sns.boxplot(x=df['meter_reading'])
# let assume one building can consume 500 kwh.

high_val = df[(df['meter_reading'] >= 500)]

high_val.head
high_val['building_id'].unique()
# plot high_val: big electricity users.

high_chart = sns.lineplot(x="hour",

                         y="meter_reading",

                         data=high_val

                         ).set_title('hourly distribution for usage above 500')

plt.show()
# split df into weekdays and weekends

## weekdays: df1

df1 = df[(df['day_name'] != 'Saturday') & (df['day_name'] != 'Sunday')]



# weekends: df2

df2 = df[(df['day_name'] == 'Saturday') | (df['day_name'] == 'Sunday')]

# df2.head()

df1.head
df2.head
## transposing df1 and df2 by the column hour: i want to cluster usage per hour of the day.

## transposing hour column

df1 = pd.pivot_table(df1, values = 'meter_reading', index=["building_id", 'day'], columns = 'hour').reset_index()

df1 = df1.drop('day', 1)



## filter rows where sum of 0 > 0 and the count of non null is > 23.

df1 = df1[df1.iloc[:,1:25].ne(0).sum(1) > 23 ]



# filter rows where the sum of NaN from column 0 to 23 is less than 10

df1 = df1[df1.isnull().sum(axis=1) < 10] 



# fill NaN with row mean.

df1.iloc[:,1:25] = df1.iloc[:,1:25].T.fillna(df1.iloc[:,1:25].mean(axis=1)).T



## compute the percentage of usage per hour per day.

#df1.iloc[:,1:25] = 100 * df1.iloc[:,1:25].div(df1.iloc[:,1:25].sum(axis=1), axis=0) # 25776 * 25

#df1 = df1.dropna(0)
# df2

df2 = pd.pivot_table(df2, values = 'meter_reading', index=["building_id", "day"], columns = 'hour').reset_index()

df2 = df2.drop('day', 1)



## filter rows where sum of 0 > 0 and the count of non null is > 23.

df2 = df2[df2.iloc[:,1:25].ne(0).sum(1) > 23 ]



# filter rows where the sum of NaN from column 0 to 23 is less than 10

df2 = df2[df2.isnull().sum(axis=1) < 10] 



# fill NaN with row mean.

df2.iloc[:,1:25] = df2.iloc[:,1:25].T.fillna(df2.iloc[:,1:25].mean(axis=1)).T



## compute the percentage of usage per hour per day.

#df2.loc[:,1:25] = 100 * df2.iloc[:,1:25].div(df2.iloc[:,1:25].sum(axis=1), axis=0) # 11349  * 25
#plot df1: weekdays

plt.style.use('seaborn')

df1.iloc[:,1:25].T.plot(figsize=(16,8), legend=False, color='blue', alpha=0.01)

plt.xlabel("hour of the day")

plt.ylabel("usage (in kwh)")

plt.title("Weekdays hourly usage")

plt.show()
plt.style.use('seaborn')

df2.iloc[:,1:25].T.plot(figsize=(16,8), legend=False, color='blue', alpha = 0.01)

plt.xlabel("hour of the day")

plt.ylabel("usage (in kwh)")

plt.title("Weekends hourly usage")

plt.show()



## possible clusters are grouped into categories of users (big: could be companies with running processes, small: households; regular consumers)
## weekdays

distortions = []

K1 = range(1,10)

for k in K1:

    kmeanModel1 = KMeans(n_clusters=k)

    kmeanModel1.fit(df1.iloc[:, 1:25])

    distortions.append(kmeanModel1.inertia_)



# plot elbow line

plt.figure(figsize=(16,8))

plt.plot(K1, distortions)

plt.xlabel('k')

plt.ylabel('Distortion')

plt.title('The Elbow Method showing the optimal k (weekdays)')

plt.show()
# weekdays model

kmeans1 = KMeans(n_clusters=4).fit(df1.iloc[:, 1:25])

centroids1 = kmeans1.cluster_centers_

print(centroids1)
## plot all weekdays centroids.

plt.style.use('seaborn')

ax = sns.lineplot(x='hour', y='usage', marker="o", 

                  hue="index",palette=["C0", "C1", "C2", "C3"],

                  data=pd.melt(pd.DataFrame(centroids1).reset_index(), 

                               id_vars="index", var_name="hour", value_name="usage")).set_title('clustering weekdays')

plt.legend(title = 'Cluster', loc= 'upper right', labels = ['low', 'high', 'very high', 'medium'])  

plt.show()                
## weekends

distortions = []

K2 = range(1,10)

for k in K2:

    kmeanModel2 = KMeans(n_clusters=k)

    kmeanModel2.fit(df2.iloc[:, 1:25])

    distortions.append(kmeanModel2.inertia_)



# plot elbow line

plt.figure(figsize=(16,8))

plt.plot(K2, distortions, 'bx-')

plt.xlabel('k')

plt.ylabel('Distortion')

plt.title('The Elbow Method showing the optimal k (weekends)')

plt.show()
# weekends model with k = 4

kmeans2 = KMeans(n_clusters=4).fit(df2.iloc[:, 1:25])

centroids = kmeans2.cluster_centers_

print(centroids)
## plot all centroids.

plt.style.use('seaborn')

ax = sns.lineplot(x='hour', y='usage', marker="o", 

                  hue="index",palette=["C0", "C1", "C2", "C3"],

                  data=pd.melt(pd.DataFrame(centroids).reset_index(), 

                               id_vars="index", var_name="hour", value_name="usage")).set_title('clustering weekends')

plt.legend(title = 'Cluster', loc= 'center right', labels = ['low', 'high', 'medium', 'very high'])  

plt.show() 