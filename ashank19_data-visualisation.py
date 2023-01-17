# Importing relevant libraries

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sb



%matplotlib inline
# Reading the unzipped file into a pandas dataframe   

df=pd.read_csv('../input/1987.csv')
# First few columns of the dataset

df.head(5)
# Info of the dataset

df.info()
#Descriptive statistics for the dataset

df.describe()
# Creating a copy of dataset before cleaning

df_new=df.copy()
df_new=df_new.drop(['AirTime','TaxiIn','TaxiOut','Diverted','CarrierDelay','CancellationCode','WeatherDelay',

             'NASDelay','SecurityDelay','LateAircraftDelay','TailNum'],axis=1)
df_new.describe()
df_new.drop(df_new[df_new.ArrTime.isnull()].index,inplace=True)
sum(df_new.ArrTime.isnull())
sum(df_new.ArrDelay.isnull())
sum(df_new.DepDelay.isnull())
sum(df_new.DepTime.isnull())
df_new.info()
df_new.sample(5)
#Converting the columns into hh:mm:ss format 

s = df_new['CRSArrTime'].astype(int).astype(str).str.zfill(4)

df_new['arrtime'] = s.str[:2] + ':' + s.str[2:] + ':00'

df_new['Schduled_Arr'] = pd.to_timedelta(df_new['arrtime'])

s = df_new['ArrTime'].astype(int).astype(str).str.zfill(4)

df_new['arrtime'] = s.str[:2] + ':' + s.str[2:] + ':00'

df_new['Actual_arr'] = pd.to_timedelta(df_new['arrtime'])

s = df_new['CRSDepTime'].astype(int).astype(str).str.zfill(4)

df_new['arrtime'] = s.str[:2] + ':' + s.str[2:] + ':00'

df_new['Schduled_dep'] = pd.to_timedelta(df_new['arrtime'])

s = df_new['DepTime'].astype(int).astype(str).str.zfill(4)

df_new['arrtime'] = s.str[:2] + ':' + s.str[2:] + ':00'

df_new['Actual_dep'] = pd.to_timedelta(df_new['arrtime'])
df_new.sample(5)
# Dropping the previous columns

df_new=df_new.drop(['DepTime','arrtime','CRSDepTime','ArrTime','CRSArrTime'],axis=1)
df_new.head(5)
df_new['Month']=df_new['Month'].astype(str)

df_new['DayofMonth']=df_new['DayofMonth'].astype(str)

df_new['DayOfWeek']=df_new['DayOfWeek'].astype(str)
df_new['FlightNum']=df_new['FlightNum'].astype(str)
df_new.info()
df_new=df_new.drop(['Year'],axis=1)
df_new.info()
df_new['Total_Delay']=df_new['ArrDelay']+df_new['DepDelay']
df_new.drop(df_new[(df_new['ActualElapsedTime'] <= 0)].index,inplace=True)

# Resetting the index

df_new=df_new.reset_index(drop=True)
df_new.query('ActualElapsedTime <=0')
df_new=df_new.drop(['Cancelled'],axis=1)
df_new.columns
# Saving the cleaned data with a new csv file name.

df_new.to_csv('1987_modified.csv')
plt.figure(figsize=[8,5])

bins=np.arange(0,df_new['Total_Delay'].max()+10,100)

plt.hist(data=df_new,x='Total_Delay',bins=bins);
plt.figure(figsize=[8,5])

sb.barplot(data=df_new,x='UniqueCarrier',y='Total_Delay');
#plt.figure(figsize=[20,20])

sb.violinplot(data=df_new,x='UniqueCarrier',y='Total_Delay',color=sb.color_palette()[0],inner='quartile');

#
b=df_new['UniqueCarrier'].value_counts()

b.plot(kind='pie',figsize=(20,10),legend=True)

plt.legend(loc=6,bbox_to_anchor=(1.0,0.5));
plt.figure(figsize=[8,5])

sb.regplot(data=df_new,x='Total_Delay',y='Distance',fit_reg=False);
df_new['Distance'].corr(df_new['Total_Delay'])
sb.regplot(data=df_new,y='Distance',x='ActualElapsedTime',fit_reg=False)
df_new['Distance'].corr(df_new['ActualElapsedTime'])
# Importing releavant libraries

import statsmodels.api as sm

# Declaring the dependent and the independent variables

line_reg=df_new.drop(df_new[df_new.Distance.isnull()].index)

line_reg=line_reg.reset_index(drop=True)

x1=line_reg['Distance']

y=line_reg['ActualElapsedTime']
x=sm.add_constant(x1)

results=sm.OLS(y,x).fit()

results.summary()
sb.set()

plt.scatter(x1,y)

y1=0.1207*x1+30.8584

fig=plt.plot(x1,y1,lw=4,c='red',label='regression line')

plt.show()
sb.regplot(data=df_new,y='Distance',x='ArrDelay',fit_reg=True);
df_new['Distance'].corr(df_new['ArrDelay'])
sb.regplot(data=df_new,y='Distance',x='DepDelay');
df_new['Distance'].corr(df_new['ArrDelay'])
g=sb.FacetGrid(data=df_new,col='Month',margin_titles=True)

g.map(plt.scatter,'Distance','Total_Delay')

g.add_legend();
df_new[(df_new['Month'] == "10")]['Total_Delay'].corr(df_new[(df_new['Month'] == "10")]['Distance'])
df_new[(df_new['Month'] == "11")]['Total_Delay'].corr(df_new[(df_new['Month'] == "11")]['Distance'])
df_new[(df_new['Month'] == "12")]['Total_Delay'].corr(df_new[(df_new['Month'] == "12")]['Distance'])
g=sb.FacetGrid(data=df_new,col='Month',col_wrap=1,margin_titles=True,sharex=False,sharey=False,size=7)

g.map(sb.barplot,'UniqueCarrier','Total_Delay')

g.add_legend();
g=sb.FacetGrid(data=df_new,col='DayOfWeek',margin_titles=True,col_wrap=3,height=5,sharex=False,sharey=False)

g.map(plt.scatter,'Distance','Total_Delay')

g.add_legend();
print(df_new[(df_new['DayOfWeek'] == "1")]['Total_Delay'].corr(df_new[(df_new['DayOfWeek'] == "1")]['Distance']))

print(df_new[(df_new['DayOfWeek'] == "2")]['Total_Delay'].corr(df_new[(df_new['DayOfWeek'] == "2")]['Distance']))

print(df_new[(df_new['DayOfWeek'] == "3")]['Total_Delay'].corr(df_new[(df_new['DayOfWeek'] == "3")]['Distance']))

print(df_new[(df_new['DayOfWeek'] == "4")]['Total_Delay'].corr(df_new[(df_new['DayOfWeek'] == "4")]['Distance']))

print(df_new[(df_new['DayOfWeek'] == "5")]['Total_Delay'].corr(df_new[(df_new['DayOfWeek'] == "5")]['Distance']))

print(df_new[(df_new['DayOfWeek'] == "6")]['Total_Delay'].corr(df_new[(df_new['DayOfWeek'] == "6")]['Distance']))

print(df_new[(df_new['DayOfWeek'] == "7")]['Total_Delay'].corr(df_new[(df_new['DayOfWeek'] == "7")]['Distance']))
from sklearn.cluster import KMeans
#Creating a copy of the datset

x=df_new.copy()

# Dropping all columns with categorical values

x=x.drop(['Schduled_dep','Actual_dep','Actual_arr','Schduled_Arr','Dest','Origin','UniqueCarrier'],axis=1)

# Dropping the rows with null values of distance

x=x.drop(x[x.Distance.isnull()].index)

x=x.reset_index(drop=True)
# creating a k-means object with 2 clusters

kmeans=KMeans(2)

# fit the data

identified_clusters=kmeans.fit_predict(x)

# predicting the cluster for each observation

x['cluster']=identified_clusters
# creating a scatter plot based on two features (distance and total delay)

plt.scatter(data=x,x='Total_Delay',y='Distance',c='cluster',cmap='rainbow')

plt.show();
# WCSS is the within cluster sum of squares elbow method used for calculating the optimum number of clusters for the given scatter plot

wcss=[]

for i in range(1,8):

    kmeans=KMeans(i)

    kmeans.fit(x)

    wcss.append(kmeans.inertia_)
no=range(1,8)

plt.plot(no,wcss);
# creating a k-means object with 7 optimum clusters 

kmeans=KMeans(7)

# fit the data

identified_clusters=kmeans.fit_predict(x)

# predicting the cluster for each observation

x['cluster']=identified_clusters

# creating a scatter plot based on two features (distance and total delay)

plt.scatter(data=x,x='Total_Delay',y='Distance',c='cluster',cmap='rainbow')

plt.show();