import timeit

main_start_time = timeit.default_timer()

from IPython.display import Image

url = 'https://www.netclipart.com/pp/m/4-45364_free-cartoon-taxi-cab-clip-art-taxi-clipart.png'

Image(url,width=200, height=200,retina=True)
!pip install preprocessing

!pip install dataprep
import pandas as pd 

import seaborn as sns

import matplotlib.pyplot as plt 

import numpy as np

#import plotly_express as px

from dataprep.eda import plot,plot_correlation





sns.set_style('whitegrid')

%matplotlib inline 

import os

print(os.listdir('../input/nytaxi'))
nyc_taxi=pd.read_csv('../input/nytaxi/ny_taxi.csv')

nyc_taxi.info()
nyc_taxi.isnull().sum()
## Checking Duplicate Values 

print(nyc_taxi.shape)

temp = nyc_taxi[nyc_taxi.duplicated()]

print(temp.shape)

print(" From the Above Code we can see there are no Duplicate Values")

del temp
print("There are %d unique id's in Training dataset, which is equal to the number of records"%(nyc_taxi.id.nunique()))
nyc_taxi.describe(include=['O'])
nyc_taxi.describe()
#Distance  function to calculate distance between given longitude and latitude points.

from math import radians, cos, sin, asin, sqrt



def distance(lon1, lat1, lon2, lat2):

    """

    Calculate the great circle distance between two points 

    on the earth (specified in decimal degrees)

    """

    # convert decimal degrees to radians 

    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])



    # haversine formula 

    dlon = lon2 - lon1 

    dlat = lat2 - lat1 

    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2

    c = 2 * asin(sqrt(a)) 

    r = 6371 # Radius of earth in kilometers. Use 3956 for miles

    return c * r
start_time = timeit.default_timer()

#sns.kdeplot(nyc_taxi.trip_duration)

print(nyc_taxi.trip_duration.min())

print(nyc_taxi.trip_duration.max())

print(nyc_taxi.trip_duration.mean())

elapsed = round(timeit.default_timer() - start_time,2)

print("Time Taken by :",elapsed)
start_time = timeit.default_timer()

nyc_taxi['distance'] = nyc_taxi.apply(lambda x: distance(x['pickup_longitude'],x['pickup_latitude'],x['dropoff_longitude'],x['dropoff_latitude']), axis = 1)

nyc_taxi['speed']= (nyc_taxi.distance/(nyc_taxi.trip_duration/3600))

nyc_taxi['dropoff_datetime']= pd.to_datetime(nyc_taxi['dropoff_datetime']) 

nyc_taxi['pickup_datetime']= pd.to_datetime(nyc_taxi['pickup_datetime'])

elapsed = round(timeit.default_timer() - start_time,2)

print("Time Taken by :",elapsed)
#plot(nyc_taxi)
## Checking Correlation  Data.

#plot_correlation(nyc_taxi)
nyc_taxi_new=nyc_taxi.drop(['id','vendor_id','pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude','store_and_fwd_flag'],axis=1)

nyc_taxi_bkup=nyc_taxi_new.copy()
def date_param(date1):

    min1=day1.minute

    hours=day1.hour

    day= day1.day

    month=day1.month

    month_name=day1.month_name()

    day_name=day1.day_name()

    day_no=day1.weekday()

    return min1,hours

    #day_no,day_name,month,month_name

#nyc_taxi_new['vendor_id'] = nyc_taxi_new['vendor_id'].replace(2,0)

start_time = timeit.default_timer()

nyc_taxi_new['pickup_min'] = nyc_taxi_new['pickup_datetime'].apply(lambda x : x.minute)

nyc_taxi_new['pickup_hour'] = nyc_taxi_new['pickup_datetime'].apply(lambda x : x.hour)

nyc_taxi_new['pickup_day'] = nyc_taxi_new['pickup_datetime'].apply(lambda x : x.day)

nyc_taxi_new['pickup_month']= nyc_taxi_new['pickup_datetime'].apply(lambda x : int(x.month))

nyc_taxi_new['pickup_weekday'] = nyc_taxi_new['pickup_datetime'].dt.day_name()

nyc_taxi_new['pickup_month_name'] = nyc_taxi_new['pickup_datetime'].dt.month_name()



nyc_taxi_new['drop_hour'] = nyc_taxi_new['dropoff_datetime'].apply(lambda x : x.hour)

nyc_taxi_new['drop_month'] = nyc_taxi_new['dropoff_datetime'].apply(lambda x : int(x.month))

nyc_taxi_new['drop_day'] = nyc_taxi_new['dropoff_datetime'].apply(lambda x : x.day)

nyc_taxi_new['drop_min'] = nyc_taxi_new['dropoff_datetime'].apply(lambda x : x.minute)

nyc_taxi_new['drop_weekday'] = nyc_taxi_new['dropoff_datetime'].dt.day_name()

nyc_taxi_new['drop_month_name'] = nyc_taxi_new['dropoff_datetime'].dt.month_name()

nyc_taxi_bkup=nyc_taxi_new.copy()

elapsed = round(timeit.default_timer() - start_time,2)

print("Time Taken in Adding Columns :",elapsed)
### There is hardly any Correlation between Passenger Count and Trip Duration evident from below linear plot . 

temp1=nyc_taxi.head(10000)

sns.lmplot(x='distance' ,y='trip_duration',hue='passenger_count' ,data=temp1,palette='RdBu')
nyc_taxi_new.distance.groupby(pd.cut(nyc_taxi_new.distance, np.arange(0,100,10))).count().plot(kind='bar')

plt.show()
plt.figure(figsize = (10,5))

sns.boxplot(nyc_taxi_new.distance)

plt.show()
sns.countplot(nyc_taxi_new.passenger_count)

plt.show()
sns.violinplot(nyc_taxi_new.trip_duration)
## Trip Count and Duration in Buckets

nyc_taxi_new.trip_duration.groupby(pd.cut(nyc_taxi_new.trip_duration, np.arange(1,7200,600))).count().plot(kind='bar')

plt.xlabel('Trip Counts')

plt.ylabel('Trip Duration (seconds)')

plt.show()
sns.countplot(nyc_taxi_new.pickup_hour)

plt.show()
sns.countplot(nyc_taxi_new.pickup_month)

plt.ylabel('Trip Counts')

plt.xlabel('Months')

plt.show()
# Showing pickup and dropoff in charts



#setting up canvas

figure,ax=plt.subplots(nrows=2,ncols=1,figsize=(10,10))



# chart for pickup_day

sns.countplot(x='pickup_weekday',data=nyc_taxi_new,ax=ax[0])

ax[0].set_title('Number of Pickups done on each day of the week')



# chart for dropoff_day

sns.countplot(x='drop_weekday',data=nyc_taxi_new,ax=ax[1])



ax[1].set_title('Number of dropoffs done on each day of the week')



plt.tight_layout()
figure,ax=plt.subplots(nrows=1,ncols=2,figsize=(12,10))

group1 = nyc_taxi_new.groupby('pickup_hour').trip_duration.mean()

group2 = nyc_taxi_new.groupby('drop_hour').trip_duration.mean()



sns.pointplot(group1.index, group1.values,color='b',ax=ax[0])

sns.pointplot(group1.index, group1.values,color='r',ax=ax[1])

plt.ylabel('Trip Duration (seconds)')

plt.xlabel('Pickup Hour')

ax[0].set_title('Pickup & Drop Hourly ')

ax[1].set_title('Drop Hourly')

plt.show()

plt.tight_layout()
figure,ax=plt.subplots(nrows=1,ncols=2,figsize=(8,6))

group3 = nyc_taxi_new.groupby('pickup_month').trip_duration.mean()

group4 = nyc_taxi_new.groupby('drop_month').trip_duration.mean()



sns.pointplot(group3.index, group3.values,color='b',ax=ax[0])

sns.pointplot(group4.index, group4.values,color='r',ax=ax[1])

plt.ylabel('Trip Duration (seconds)')

plt.xlabel('Pickup & Drop Month')

plt.show()

plt.tight_layout()
group6 = nyc_taxi_new.groupby('pickup_weekday').distance.mean()

sns.pointplot(group6.index, group6.values)

plt.ylabel('Distance (km)')

plt.show()

plt.tight_layout()
group7 = nyc_taxi_new.groupby('pickup_month').distance.mean()

sns.pointplot(group7.index, group7.values)

plt.ylabel('Distance (km)')

plt.show()

plt.tight_layout()
figure,ax=plt.subplots(nrows=1,ncols=2,figsize=(12,10))

group9 = nyc_taxi_new.groupby('pickup_hour').speed.mean()

group10 = nyc_taxi_new.groupby('drop_hour').speed.mean()

sns.pointplot(group9.index, group9.values,ax=ax[0],colour='b')

sns.pointplot(group10.index, group10.values,ax=ax[1],colour='r')

plt.show()

plt.tight_layout()
figure,ax=plt.subplots(nrows=1,ncols=2,figsize=(12,10))

group11 = nyc_taxi_new.groupby('pickup_weekday').speed.mean()

group12 = nyc_taxi_new.groupby('drop_weekday').speed.mean()

sns.pointplot(group11.index, group11.values,ax=ax[0],colour='b')

sns.pointplot(group12.index, group12.values,ax=ax[1],colour='r')

plt.show()

plt.tight_layout()
figure,ax=plt.subplots(nrows=1,ncols=2,figsize=(8,6))

nyc_taxi_bkup1=nyc_taxi_new.copy()

#nyc_taxi_new.trip_duration.plot(kind='kde')

sns.distplot(np.log(nyc_taxi_new.trip_duration),ax=ax[0])

sns.distplot(nyc_taxi_new.trip_duration,ax=ax[1])

plt.show()

plt.tight_layout()
print('Longitude Bounds: {} to {}'.format(max(nyc_taxi.pickup_longitude.min(),nyc_taxi.dropoff_longitude.min()),max(nyc_taxi.pickup_longitude.max(),nyc_taxi.dropoff_longitude.max())))

print('Lattitude Bounds: {} to {}'.format(max(nyc_taxi.pickup_latitude.min(),nyc_taxi.dropoff_latitude.min()),max(nyc_taxi.pickup_latitude.max(),nyc_taxi.dropoff_latitude.max())))
#Visualizing Passenger road map for picking up

fig, ax = plt.subplots(ncols=1, nrows=1,figsize=(10,8))

plt.ylim(40.63, 40.85)

plt.xlim(-74.03,-73.75)

ax.scatter(nyc_taxi['pickup_longitude'],nyc_taxi['pickup_latitude'], s=0.02, alpha=1)
## Drop Latitude 
fig, ax = plt.subplots(ncols=1, nrows=1,figsize=(10,8))

plt.ylim(40.63, 40.85)

plt.xlim(-74.03,-73.75)

ax.scatter(nyc_taxi['dropoff_longitude'],nyc_taxi['dropoff_latitude'], s=0.02, alpha=1)
start_time = timeit.default_timer()

i=nyc_taxi_new.shape[0]

temp1=nyc_taxi_new[(nyc_taxi_new['speed']<1)&(nyc_taxi_new['distance']==0)]

nyc_taxi_new.drop(temp1.index,inplace=True)

print("No of Records Deleted is ",i-nyc_taxi_new.shape[0])
nyc_taxi_new1=nyc_taxi_new.copy()
i=nyc_taxi_new.shape[0]

temp1=nyc_taxi_new[(nyc_taxi_new['pickup_day']< nyc_taxi_new['drop_day'])& (nyc_taxi_new['trip_duration']> 10000) &(nyc_taxi_new['distance'] <5) & (nyc_taxi_new['pickup_hour']<23)]

nyc_taxi_new.drop(temp1.index,inplace=True)

print("No of Records Deleted is ",i-nyc_taxi_new.shape[0])
nyc_taxi_bkup=nyc_taxi_new.copy()

i=nyc_taxi_new.shape[0]

temp1=nyc_taxi_new[(nyc_taxi_new['speed']<1) & (nyc_taxi_new['distance']< 1) ]

nyc_taxi_new.drop(temp1.index,inplace=True)

print("No of Records Deleted is ",i-nyc_taxi_new.shape[0])
# Droping the 3 Columns where looks to be interstate

nyc_taxi_bkup=nyc_taxi_new.copy()

i=nyc_taxi_new.shape[0]

nyc_taxi_new[nyc_taxi_new['trip_duration']/60 >10000]['trip_duration'].plot(kind='bar')

nyc_taxi_new.drop([978383,680594,355003],inplace=True)

print("No of Records Deleted is ",i-nyc_taxi_new.shape[0])
nyc_taxi_bkup=nyc_taxi_new.copy()

i=nyc_taxi_new.shape[0]

temp1=nyc_taxi_new[nyc_taxi_new['distance']< .2]

nyc_taxi_new.drop(temp1.index,inplace=True)

print("No of Records Deleted is ",i-nyc_taxi_new.shape[0])
#nyc_taxi_bkup=nyc_taxi_new.copy()

i=nyc_taxi_new.shape[0]

temp1=nyc_taxi_new[nyc_taxi_new['passenger_count']==0]

nyc_taxi_new.drop(temp1.index,inplace=True)

print("No of Records Deleted is ",i-nyc_taxi_new.shape[0])
#nyc_taxi_bkup=nyc_taxi_new.copy()

i=nyc_taxi_new.shape[0]

temp1=nyc_taxi_new[nyc_taxi_new['passenger_count']==0]

nyc_taxi_new.drop(temp1.index,inplace=True)

print("No of Records Deleted is ",i-nyc_taxi_new.shape[0])
nyc_taxi_new.trip_duration.groupby(pd.cut(nyc_taxi_new.passenger_count, np.arange(1,9,1))).count().plot(kind='bar')

plt.xlabel('passenger_count')

plt.ylabel('Trip Duration (seconds)')

plt.show()

i=nyc_taxi_new.shape[0]

temp1=nyc_taxi_new[nyc_taxi_new['passenger_count']>6]

nyc_taxi_new.drop(temp1.index,inplace=True)

print("No of Records Deleted is ",i-nyc_taxi_new.shape[0])
nyc_taxi_new.describe()
import datetime as dt

print(len(nyc_taxi_new[nyc_taxi_new['dropoff_datetime'].dt.year>2016]))

print(len(nyc_taxi_new[nyc_taxi_new['dropoff_datetime'].dt.year<2016]))
#nyc_taxi_new.sort_values(by='speed',ascending=False).head(10)

###Assuming all the Trips people take is atleast more than 1 Minutes

nyc_taxi_bkup=nyc_taxi_new.copy()

i=nyc_taxi_new.shape[0]

temp1=nyc_taxi_new[nyc_taxi_new['trip_duration']<120]

nyc_taxi_new.drop(temp1.index,inplace=True)

print("No of Records Deleted is ",i-nyc_taxi_new.shape[0])
nyc_taxi_bkup=nyc_taxi_new.copy()

i=nyc_taxi_new.shape[0]

temp1=nyc_taxi_new[nyc_taxi_new['speed']>50]['speed']

#sns.hist(temp1,bins=10)

nyc_taxi_new.drop(temp1.index,inplace=True)

print("No of Records Deleted is ",i-nyc_taxi_new.shape[0])

elapsed = round(timeit.default_timer() - start_time,2)

print("Time Taken for Entire Cleaning :",elapsed)
#plot(nyc_taxi_new)

nyc_taxi_bkup=nyc_taxi_new.copy()

from sklearn import linear_model

from sklearn.model_selection import train_test_split

from sklearn import metrics

from sklearn import preprocessing

nyc_taxi_new.columns
print(nyc_taxi_new.isnull().sum())

nyc_taxi_new[nyc_taxi_new['distance'].isnull()]

nyc_taxi_new.dropna(inplace=True)

nyc_taxi_new=nyc_taxi_bkup.copy()
figure,ax=plt.subplots(nrows=1,ncols=2,figsize=(10,6))

i=nyc_taxi_new.shape[0]

ax[0].title.set_text('Trip Duration Before Removing Outliers')

sns.distplot(nyc_taxi_new.trip_duration,ax=ax[0])

temp1=nyc_taxi_new.drop({'pickup_datetime','dropoff_datetime'},axis=1)

Q1 = nyc_taxi_new.quantile(0.030)

Q3 = nyc_taxi_new.quantile(0.970)

IQR = Q3 - Q1

mds_out = temp1[~((temp1 < (Q1 - 1.5 * IQR)) |(temp1 > (Q3 + 1.5 * IQR))).any(axis=1)]

sns.distplot(mds_out.trip_duration,ax=ax[1])

ax[1].title.set_text('Trip Duration After Removing Outliers')

print("No of Records Deleted is ",i - mds_out.shape[0])

plt.show()

plt.tight_layout()



mds_out['distance'].describe()

nyc_taxi_new=mds_out.copy()

nyc_taxi_new['pickup_datetime']=nyc_taxi['pickup_datetime'].max()

nyc_taxi_new['dropoff_datetime']=nyc_taxi['dropoff_datetime'].max()
start_time = timeit.default_timer()

import category_encoders as ce 

temp_encd=nyc_taxi_new.copy()

test1=temp_encd.drop({'pickup_datetime', 'dropoff_datetime','distance', 'speed',

                     'pickup_weekday', 'pickup_month_name','drop_month_name','drop_weekday'},axis=1)

colums1=['passenger_count','pickup_min', 'pickup_hour', 'pickup_day','pickup_month','drop_hour',

 'drop_month', 'drop_day', 'drop_min']

encoder= ce.BinaryEncoder(cols=colums1)

dfbin=encoder.fit_transform(test1[colums1])

Bin_Encoded=pd.concat([test1,dfbin],axis=1)

elapsed = round(timeit.default_timer() - start_time,2)

print(Bin_Encoded.shape)

print(elapsed)

Bin_Encoded
def Train_Test(sam_size=.01,scal=False):

    sam_size=.75

    Bin_Enc1=Bin_Encoded.sample(frac=sam_size,random_state=1)  

    X = Bin_Enc1.drop('trip_duration',axis=1)

    y = Bin_Enc1['trip_duration']

    features=Bin_Enc1.drop('trip_duration',axis=1)

    if scal==True:

        X=preprocessing.scale(X) 

        X=pd.DataFrame(X)

        y = Bin_Enc1['trip_duration']

        return X,y,features

    else: return X,y,features
#del summ

col_list=['Model_Name','MenAbErr','MenSqErr','RMSE','Min_Err','Max_Err','Comments','Sample','time_sec']

summ=pd.DataFrame(columns=col_list)

summ['MenAbErr'] = summ['MenAbErr'].astype(float)

summ['MenSqErr'] = summ['MenSqErr'].astype(float)

summ['RMSE']     = summ['RMSE'].astype(float)

summ['Min_Err']  = summ['Min_Err'].astype(float)

summ['Max_Err']  = summ['Max_Err'].astype(float)

summ['Sample']   = summ['Sample'].astype(int)

summ['time_sec']  = summ['time_sec'].astype(float)

summ.info()
start_time = timeit.default_timer()

X,y,feature_columns=Train_Test(.25,False)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)



reg =linear_model.LinearRegression()

reg.fit(X_train,y_train)



#print("reg.intercept_=> %10.10f" %(reg.intercept_))

#print(list(zip(feature_columns, reg.coef_)))

y_pred=reg.predict(X_test)



#################Calculate the Error Percentages###########

Model_Name="Linear Regeression"

mabs=metrics.mean_absolute_error(y_test, y_pred)

mse= metrics.mean_squared_error(y_test, y_pred)

rmse_val=np.sqrt(metrics.mean_squared_error(y_test, y_pred))

coments='Normal'

df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred,'Error':y_test -y_pred})

Min_Error=df.Error.min()

Max_Error=df.Error.max()



print('Mean Absolute Error   :',mabs)  

print('Mean Squared Error    :',mse )  

print('Root Mean Squared     :',rmse_val )

print("Maximum Error is      :",Min_Error)

print("Minimum Error is      :",Max_Error)



elapsed = round(timeit.default_timer() - start_time,2)

summ=summ.append({'Model_Name':Model_Name, 'MenAbErr':mabs, 'MenSqErr':mse, 'RMSE':rmse_val,

                  'Comments':coments,'Sample':X.shape[0],

                  'Min_Error':Min_Error, 'Max_Error':Max_Error,

                  'time_sec':elapsed},ignore_index=True)



summ.drop({'Min_Err','Max_Err'},inplace=True,axis=1)

summ
#summ.drop(1,inplace=True)

#from sklearn.metrics import mean_squared_log_error

#feature_columns=nyc_taxi_new.drop(['drop_weekday','drop_month_name','pickup_month_name','pickup_weekday',

#                      'pickup_datetime','dropoff_datetime','trip_duration','passenger_count','speed'],axis=1)

#X2=nyc_taxi_new.drop(['drop_weekday','drop_month_name','pickup_month_name','pickup_weekday',

#                      'pickup_datetime','dropoff_datetime','passenger_count','speed'],axis=1)



#X,y,feature_columns=Train_Test(.05,False)

#print(feature_columns.shape)

#X=pd.concat([X,y],axis=1)

#print(X.columns)#

#X1=np.log(X)

#X1.replace([np.inf, -np.inf,np.inf], np.nan, inplace=True)

#X1.dropna(inplace=True)                      

#X=pd.DataFrame(X1)

#y=X['trip_duration']

#X1=X.drop('trip_duration',axis=1)



#X_train, X_test, y_train, y_test = train_test_split(X, y,train_size=.8,test_size=0.2, random_state=111)



#reg =linear_model.LinearRegression()

#reg.fit(X_train,y_train)

#print("reg.intercept_=> %10.10f" %(reg.intercept_))

#print(list(zip(feature_columns, reg.coef_)))

#y_pred=reg.predict(X_test)

#rmse_val=np.sqrt(metrics.mean_squared_error(y_test, y_pred))

##################Calculate the Error Percentages###########

#Model_Name="Linear Regeression With Log "

#mabs=metrics.mean_absolute_error(y_test, y_pred)

#mse= metrics.mean_squared_error(y_test, y_pred)

#rmse= np.sqrt(mean_squared_log_error( y_test, y_pred ))

#coments='Here RMSE is RMSLE Scale is Log'

#df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred,'Error':y_test -y_pred})

#Min_Error=df.Error.min()

#Max_Error=df.Error.max()



#print("================================================================")

#print("\n Total No of Sample Used :",X.shape[0])

#print(" Total No of Test Sample COllection :",X_test.shape[0])

#print('\nMean Absolute Error    :',mabs)  

#print('Mean Squared Error     :',mse )  

#print('Root Mean Squared Error:',rmse )

#print("Maximum Error is       :",Min_Error)

#print("Minimum Error is       :",Max_Error)



#summ=summ.append({'Model_Name':Model_Name, 'MenAbErr':mabs, 'MenSqErr':mse, 'RMSE':rmse,

#                  'Comments':coments,'Sample':X.shape[0],

#                  'Min_Error':Min_Error, 'Max_Error':Max_Error,

#                 },ignore_index=True)

#summ.drop({'Min_Err','Max_Err'},inplace=True,axis=1)

#summ
feature_columns.shape

start_time = timeit.default_timer()

np.random.seed(0)

import lime

import lime.lime_tabular

import numpy as np

#X,y,feature_columns=Train_Test(.25,True)

explainer = lime.lime_tabular.LimeTabularExplainer(np.array(X_train),feature_names=feature_columns[:30:-1], 

             verbose=True, mode='regression')
exp = explainer.explain_instance(X_test.iloc[350], reg.predict)

exp.as_pyplot_figure()

pd.DataFrame(exp.as_list())
exp = explainer.explain_instance(X_test.iloc[5000], reg.predict)

exp.show_in_notebook(show_table=True, show_all=False)
exp = explainer.explain_instance(X_test.iloc[21], reg.predict)

exp.show_in_notebook(show_table=True, show_all=False)

elapsed = round(timeit.default_timer() - start_time,2)

print("Time Taken by Lime :",elapsed)
start_time = timeit.default_timer()

y_pred_test=reg.predict(X_train)

rmse_val=np.sqrt(metrics.mean_squared_error(y_train, y_pred_test))

print('Train Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_train, y_pred_test)))



elapsed = round(timeit.default_timer() - start_time,2)

print("Time Taken to Print Null RMSE :",elapsed)
start_time = timeit.default_timer()

X,y,feature_columns=Train_Test(.25,False)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=111,test_size=.2)

y_null = np.zeros_like(y_test, dtype=int)

y_null.fill(y_test.mean())

y_pred=reg.predict(X_test)



Model_Name="NULL Value Linear Regeression"

mabs=metrics.mean_absolute_error(y_test, y_pred)

mse= metrics.mean_squared_error(y_test, y_pred)

rmse_Null=np.sqrt(metrics.mean_squared_error(y_test, y_null))

coments='NULL Value'

df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred,'Error':y_test -y_pred})

Min_Error=df.Error.min()

Max_Error=df.Error.max()



print('Mean Absolute Error             :',mabs)  

print('Mean Squared Error              :',mse )  

print('Root Mean Squared Error For NuLL:',rmse_Null )

print("Maximum Error is                :",Min_Error)

print("Minimum Error is                :",Max_Error)



elapsed = round(timeit.default_timer() - start_time,2)

summ=summ.append({'Model_Name':Model_Name, 'MenAbErr':mabs, 'MenSqErr':mse, 'RMSE':rmse_Null,

                  'Comments':coments,'Sample':X.shape[0],

                  'Min_Error':Min_Error, 'Max_Error':Max_Error,

                  'time_sec':elapsed},ignore_index=True)



#summ.drop({'Min_Err','Max_Err'},inplace=True,axis=1)



summ

#summ.drop(1,inplace=True)

start_time = timeit.default_timer()

X,y,feature_columns=Train_Test(.25,True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)



reg =linear_model.LinearRegression()



reg.fit(X_train,y_train)

#print("reg.intercept_=> %10.10f" %(reg.intercept_))

#print(list(zip(feature_columns, reg.coef_)))

y_pred=reg.predict(X_test)



#################Calculate the Error Percentages###########

Model_Name="Linear Regerr with Scaler"

mabs=metrics.mean_absolute_error(y_test, y_pred)

mse= metrics.mean_squared_error(y_test, y_pred)

rmse_val=np.sqrt(metrics.mean_squared_error(y_test, y_pred))

coments='Scale of RMSE is Different'

df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred,'Error':y_test -y_pred})

Min_Error=df.Error.min()

Max_Error=df.Error.max()





print('Mean Absolute Error    :',mabs)  

print('Mean Squared Error     :',mse )  

print('Root Mean Squared Error:',rmse_val )

print("Maximum Error is       :",Min_Error)

print("Minimum Error is       :",Max_Error)





elapsed = round(timeit.default_timer() - start_time,2)

summ=summ.append({'Model_Name':Model_Name, 'MenAbErr':mabs, 'MenSqErr':mse, 'RMSE':rmse_val,

                  'Comments':coments,'Sample':X.shape[0],

                  'Min_Error':Min_Error, 'Max_Error':Max_Error,

                  'time_sec':elapsed},ignore_index=True)



summ





pd.DataFrame(exp.as_list())
start_time = timeit.default_timer()



import xgboost as xgb

#from xgboost import plot_importance, plot_tree

#from sklearn.model_selection import RandomizedSearchCV ,cross_val_score, KFold



X,y,feature_columns=Train_Test(.01,True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=111)



model = xgb.XGBRegressor()

model.fit(X_train,y_train)

print(model)

y_pred = model.predict(data=X_test)



Model_Name="XGBoost"

mabs=metrics.mean_absolute_error(y_test, y_pred)

mse= metrics.mean_squared_error(y_test, y_pred)

rmse= np.sqrt(metrics.mean_squared_error(y_test, y_pred))

coments='Standardised Data'

df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred,'Error':y_test -y_pred})

Min_Error=df.Error.min()

Max_Error=df.Error.max()



print('\nMean Absolute Error    :',mabs)  

print('Mean Squared Error     :',mse )  

print('Root Mean Squared Error:',rmse )

print("Maximum Error is       :",Min_Error)

print("Minimum Error is       :",Max_Error)



elapsed = round(timeit.default_timer() - start_time,2)

summ=summ.append({'Model_Name':Model_Name, 'MenAbErr':mabs, 'MenSqErr':mse, 'RMSE':rmse,

                  'Comments':coments,'Sample':X.shape[0],

                  'Min_Error':Min_Error, 'Max_Error':Max_Error,

                  'time_sec':elapsed},ignore_index=True)

#summ.drop({'Min_Err','Max_Err'},inplace=True,axis=1)

summ
start_time = timeit.default_timer()



from sklearn.linear_model import Ridge

from sklearn.linear_model import RidgeCV

## training the model



X,y,feature_columns=Train_Test(.01,False)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=111)



#ridgeReg = Ridge(alpha=0.05, normalize=True)

ridgeReg = RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1])

ridgeReg.fit(X_train,y_train)



#clf = RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1]).fit(X, y)



y_pred = ridgeReg.predict(X_test)



Model_Name="Linear Ridge Regeression"

mabs=metrics.mean_absolute_error(y_test, y_pred)

mse= metrics.mean_squared_error(y_test, y_pred)

rmse= np.sqrt(metrics.mean_squared_error(y_test, y_pred))

coments='Scale is Preprocessing'

df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred,'Error':y_test -y_pred})

Min_Error=df.Error.min()

Max_Error=df.Error.max()



print('Mean Absolute Error    :',mabs)  

print('Mean Squared Error     :',mse )  

print('Root Mean Squared Error:',rmse )

print("Maximum Error is       :",Min_Error)

print("Minimum Error is       :",Max_Error)





elapsed = round(timeit.default_timer() - start_time,2)

summ=summ.append({'Model_Name':Model_Name, 'MenAbErr':mabs, 'MenSqErr':mse, 'RMSE':rmse,

                  'Comments':coments,'Sample':X.shape[0],

                  'Min_Error':Min_Error, 'Max_Error':Max_Error,

                  'time_sec':elapsed},ignore_index=True)

summ

start_time = timeit.default_timer()

from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA

X,y,feature_columns=Train_Test(.05,True)

scaler=StandardScaler()

X_norm = scaler.fit_transform(X)

pca = PCA()

pca.fit_transform(X_norm)

variance_ratio_cum_sum=np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)

print(variance_ratio_cum_sum)

plt.plot(np.cumsum(pca.explained_variance_ratio_))

plt.xlabel('Number of components')

plt.ylabel('Cumulative explained variance')

plt.annotate('30',xy=(30, .9652))
comp=30

x_pca = PCA(n_components=comp)

X_norm_final = x_pca.fit_transform(X_norm)

# correlation between the variables after transforming the data with PCA is 0



correlation = pd.DataFrame(X_norm_final).corr()

sns.heatmap(correlation, vmax=1, square=True,cmap='viridis')

plt.title('Correlation between different features')

X_norm_final=pd.DataFrame(X_norm_final)
#X2=preprocessing.scale(X_norm_final)

X=pd.DataFrame(X_norm_final)

#y=preprocessing.scale(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

reg =linear_model.LinearRegression()



reg.fit(X_train,y_train)

y_pred=reg.predict(X_test)

print("reg.intercept_=> %10.10f" %(reg.intercept_))

print(list(zip(feature_columns, reg.coef_)))



#################Calculate the Error Percentages###########

Model_Name="Principal Component Analysis"

mabs=metrics.mean_absolute_error(y_test, y_pred)

mse= metrics.mean_squared_error(y_test, y_pred)

rmse=np.sqrt(metrics.mean_squared_error(y_test, y_pred))

coments='PCA with '+str(comp)

df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred,'Error':y_test -y_pred})

Min_Error=df.Error.min()

Max_Error=df.Error.max()



elapsed = round(timeit.default_timer() - start_time,2)

summ=summ.append({'Model_Name':Model_Name, 'MenAbErr':mabs, 'MenSqErr':mse, 'RMSE':rmse,

                  'Comments':coments,'Sample':X.shape[0],

                  'Min_Error':Min_Error, 'Max_Error':Max_Error,

                  'time_sec':elapsed},ignore_index=True)



summ
start_time = timeit.default_timer()

from sklearn.tree import DecisionTreeRegressor

X,y,feature_columns=Train_Test(.01,False)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

dt=DecisionTreeRegressor()

dt.fit(X_train,y_train)

y_pred=dt.predict(X_test)





Model_Name="DecisionTreeRegressor"

mabs=metrics.mean_absolute_error(y_test, y_pred)

mse= metrics.mean_squared_error(y_test, y_pred)

rmse=np.sqrt(metrics.mean_squared_error(y_test, y_pred))

coments='Normal'

df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred,'Error':y_test -y_pred})

Min_Error=df.Error.min()

Max_Error=df.Error.max()



print('Mean Absolute Error    :',mabs)  

print('Mean Squared Error     :',mse )  

print('Root Mean Squared Error:',rmse )

print("Maximum Error is       :",Min_Error)

print("Minimum Error is       :",Max_Error)





elapsed = round(timeit.default_timer() - start_time,2)

summ=summ.append({'Model_Name':Model_Name, 'MenAbErr':mabs, 'MenSqErr':mse, 'RMSE':rmse,

                  'Comments':coments,'Sample':X.shape[0],

                  'Min_Error':Min_Error, 'Max_Error':Max_Error,

                  'time_sec':elapsed},ignore_index=True)

summ
start_time = timeit.default_timer()

from sklearn.ensemble import RandomForestRegressor

X,y,feature_columns=Train_Test(.01,False)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

rf=RandomForestRegressor()

rf.fit(X_train,y_train)

y_pred=rf.predict(X_test)



Model_Name="RandomForestRegressor"

mabs=metrics.mean_absolute_error(y_test, y_pred)

mse= metrics.mean_squared_error(y_test, y_pred)

rmse=np.sqrt(metrics.mean_squared_error(y_test, y_pred))

coments='Normal'

df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred,'Error':y_test -y_pred})

Min_Error=df.Error.min()

Max_Error=df.Error.max()



print('Mean Absolute Error    :',mabs)  

print('Mean Squared Error     :',mse )  

print('Root Mean Squared Error:',rmse )

print("Maximum Error is       :",Min_Error)

print("Minimum Error is       :",Max_Error)





elapsed = round(timeit.default_timer() - start_time,2)

summ=summ.append({'Model_Name':Model_Name, 'MenAbErr':mabs, 'MenSqErr':mse, 'RMSE':rmse,

                  'Comments':coments,'Sample':X.shape[0],

                  'Min_Error':Min_Error, 'Max_Error':Max_Error,

                  'time_sec':elapsed},ignore_index=True)

summ
start_time = timeit.default_timer()

from sklearn.ensemble import AdaBoostRegressor

ab=AdaBoostRegressor()

X,y,feature_columns=Train_Test(.01,False)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

ab.fit(X_train,y_train)

y_pred=ab.predict(X_test)



Model_Name="AdaBoostRegressor"

mabs=metrics.mean_absolute_error(y_test, y_pred)

mse= metrics.mean_squared_error(y_test, y_pred)

rmse=np.sqrt(metrics.mean_squared_error(y_test, y_pred))

coments='Normal'

df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred,'Error':y_test -y_pred})

Min_Error=df.Error.min()

Max_Error=df.Error.max()



print('Mean Absolute Error    :',mabs)  

print('Mean Squared Error     :',mse )  

print('Root Mean Squared Error:',rmse )

print("Maximum Error is       :",Min_Error)

print("Minimum Error is       :",Max_Error)





elapsed = round(timeit.default_timer() - start_time,2)

summ=summ.append({'Model_Name':Model_Name, 'MenAbErr':mabs, 'MenSqErr':mse, 'RMSE':rmse,

                  'Comments':coments,'Sample':X.shape[0],

                  'Min_Error':Min_Error, 'Max_Error':Max_Error,

                  'time_sec':elapsed},ignore_index=True)

summ

start_time = timeit.default_timer()

from sklearn.ensemble import GradientBoostingRegressor

gb = GradientBoostingRegressor()

X,y,feature_columns=Train_Test(.01,False)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

gb.fit(X_train,y_train)

y_pred = gb.predict(X_test)



Model_Name="GradientBoostingRegressor"

mabs=metrics.mean_absolute_error(y_test, y_pred)

mse= metrics.mean_squared_error(y_test, y_pred)

rmse=np.sqrt(metrics.mean_squared_error(y_test, y_pred))

coments='Normal'

df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred,'Error':y_test -y_pred})

Min_Error=df.Error.min()

Max_Error=df.Error.max()



print('Mean Absolute Error    :',mabs)  

print('Mean Squared Error     :',mse )  

print('Root Mean Squared Error:',rmse )

print("Maximum Error is       :",Min_Error)

print("Minimum Error is       :",Max_Error)





elapsed = round(timeit.default_timer() - start_time,2)

summ=summ.append({'Model_Name':Model_Name, 'MenAbErr':mabs, 'MenSqErr':mse, 'RMSE':rmse,

                  'Comments':coments,'Sample':X.shape[0],

                  'Min_Error':Min_Error, 'Max_Error':Max_Error,

                  'time_sec':elapsed},ignore_index=True)

summ
start_time = timeit.default_timer()

import lightgbm as lgb

lgbm = lgb.LGBMRegressor()

X,y,feature_columns=Train_Test(.01,False)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

lgbm.fit(X_train,y_train)

y_pred = lgbm.predict(X_test)



Model_Name="Light GBM"

mabs=metrics.mean_absolute_error(y_test, y_pred)

mse= metrics.mean_squared_error(y_test, y_pred)

rmse=np.sqrt(metrics.mean_squared_error(y_test, y_pred))

coments='Normal'

df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred,'Error':y_test -y_pred})

Min_Error=df.Error.min()

Max_Error=df.Error.max()



print('Mean Absolute Error    :',mabs)  

print('Mean Squared Error     :',mse )  

print('Root Mean Squared Error:',rmse )

print("Maximum Error is       :",Min_Error)

print("Minimum Error is       :",Max_Error)





elapsed = round(timeit.default_timer() - start_time,2)

summ=summ.append({'Model_Name':Model_Name, 'MenAbErr':mabs, 'MenSqErr':mse, 'RMSE':rmse,

                  'Comments':coments,'Sample':X.shape[0],

                  'Min_Error':Min_Error, 'Max_Error':Max_Error,

                  'time_sec':elapsed},ignore_index=True)

summ

print("                    Summary of Error is in the Report             ")

summ
plt.figure(figsize = (12,8))

#temp=summ.copy()

#temp.drop(2,inplace=True)

print(" Here is the Summary of Each of the Models ")

sns.lineplot(x='Model_Name',y='RMSE',data=summ)

plt.title("The Figure Shows RMSE and Model Name ")
elapsed = round(timeit.default_timer() - main_start_time,2)

print("Total Time Taken in Running the Notebook in Minutes :",round(elapsed/60,2))