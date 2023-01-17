import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from scipy.stats import lognorm



import matplotlib

import datetime as dt

from tabulate import tabulate             

import seaborn as sns

import matplotlib.animation as animation



from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_predict

from sklearn.linear_model import LinearRegression



import datetime as dt

from datetime import datetime

import dateutil.parser



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
green_list =pd.read_csv('../input/green_tripdata_2015-09.csv')
v = green_list.Trip_distance # create a vector to contain Trip Distance



v[~((v-v.median()).abs()>3*v.std())].hist(bins=30) # removing outliers 

plt.xlabel('Trip Distance (miles)')

plt.ylabel('Frequency')

plt.title('Histogram of Trip Distance')



scatter,loc,mean = lognorm.fit(green_list.Trip_distance.values,scale=green_list.Trip_distance.mean(), loc=0)    # applying lognorm fit

pdf_fitted = lognorm.pdf(np.arange(0,12,.1),scatter,loc,mean)

plt.plot(np.arange(0,12,0.1),600000*pdf_fitted,'r')     # limits from 0 to 12 with a step size of 0.1

plt.legend(['Lognormal Fit','Data'])



plt.show()
green_list['Pickup_dt'] = green_list.lpep_pickup_datetime.apply(lambda x:dt.datetime.strptime(x,"%Y-%m-%d %H:%M:%S"))

green_list['Pickup_hour'] = green_list.Pickup_dt.apply(lambda x:x.hour)

#green_list['Dropoff_dt'] = green_list.Lpep_dropoff_datetime.apply(lambda x:dt.datetime.strptime(x,"%Y-%m-%d %H:%M:%S"))



fig,ax = plt.subplots(1,1,figsize=(10,5)) #plotting the mean and median on a single graph

table_list =green_list.pivot_table(index='Pickup_hour', values='Trip_distance',aggfunc=('mean','median')).reset_index()     # pivot table to aggregate the trip distance by hour



table_list.columns = ['Hour','Mean Trip Distance','Median Trip Distance']

table_list[['Mean Trip Distance','Median Trip Distance']].plot(ax=ax)

print(tabulate(table_list.values.tolist(),["Hour","Mean Trip distance","Median Trip Distance"]))  #printing tabular version of graph

    

plt.title('Trip distance by pickup hour')

plt.xlabel('Hours (24 hour format)')

plt.ylabel('Distance (miles)')

plt.xlim([0,23])       # 24 hour format

plt.show()
trips =green_list[(green_list.RateCodeID==2) | (green_list.RateCodeID==3)]    # 2 represents JFK while 3 represnts Newark airports

print("Number of trips that originate or terminate at NYC are:", len(trips.index))



fare_amount =trips.Fare_amount.mean()

print("Average fare amount for tips to and fro the airports are: $",fare_amount)



amt =trips.mean()

#print(amt[16])  # mean comes out to be approximately 1.68

print("Average mode of payment while going to and fro airports is cash.")  # 2 refers to credit card mode of payment



v_air =trips.Trip_distance

v_nonair =green_list.loc[~green_list.index.isin(v_air), "Trip_distance"]  # taking the complement of v_air





trips.Pickup_hour.value_counts(normalize=True).sort_index().plot()  # plotting hourly distribution

green_list.loc[~green_list.index.isin(v_air.index),'Pickup_hour'].value_counts(normalize=True).sort_index().plot()

plt.xlabel('Hours (24 hours)')

plt.ylabel('Trip count')

plt.title('Hourly distribution of Trips')

plt.legend(['Airport trips','Non-airport trips'],bbox_to_anchor=(.05, 1), loc=2, borderaxespad=0.)

plt.show()



clean_RCID =green_list[~((green_list.RateCodeID>=1) & (green_list.RateCodeID<=6))].index                                    # cleaning RateCodeID since 99 is an outlier 

green_list.loc[clean_RCID, 'RateCodeID'] =2    # 2 was seen as the most common cash method



df = green_list.pivot_table(index='Pickup_hour', columns='RateCodeID', values='Fare_amount', aggfunc=np.median)             # plotting heat map (changing values =Distance we can

        # get another heatmap which provides extra information)

sns.heatmap(df, annot=True, fmt=".1f")

plt.title("Distribution of RateID's with Pickup Hour")

plt.show()
green_list =pd.read_csv('../input/green_tripdata_2015-09.csv')

clean_RCID =green_list[~((green_list.RateCodeID>=1) & (green_list.RateCodeID<=6))].index                                    	# cleaning RateCodeID since 99 is an outlier 

green_list.loc[clean_RCID, 'RateCodeID'] =2

green_list.Fare_amount = green_list.Fare_amount.abs()

green_list.MTA_tax = green_list.MTA_tax.abs()

green_list.Tolls_amount = green_list.Tolls_amount.abs()

green_list.improvement_surcharge = green_list.improvement_surcharge.abs()

green_list.Total_amount = green_list.Total_amount.abs()



green_list['Trip_type '] = green_list['Trip_type '].replace(np.NaN,1)



green_list =green_list.drop(['VendorID','lpep_pickup_datetime','Lpep_dropoff_datetime','Store_and_fwd_flag','Pickup_longitude','Pickup_latitude','Dropoff_longitude','Dropoff_latitude','Extra','Tolls_amount','Ehail_fee', 'Trip_type '],axis =1)



tip =green_list[(green_list['Total_amount']>=2.5)]

green_list =green_list[(green_list['Total_amount']>=2.5)]

tip['Tip_percentage'] = (tip.Tip_amount/tip.Total_amount) * 100

tips =pd.DataFrame()

tips =tip['Tip_percentage']

tips =tips.values.reshape((1490167,1))



lm =LinearRegression()

model = lm.fit(green_list,tips)

predictions = lm.predict(green_list)

print(predictions)[0:5]       #predicting the next 5 values of tip percentage

print("The accuracy is: ", lm.score(green_list,tips))
green_list =pd.read_csv('../input/green_tripdata_2015-09.csv')

d= datetime.now()



green_list['Dropoff_dt'] = green_list.Lpep_dropoff_datetime.apply(lambda x:dt.datetime.strptime(x,"%Y-%m-%d %H:%M:%S"))

green_list['Pickup_dt'] = green_list.lpep_pickup_datetime.apply(lambda x:dt.datetime.strptime(x,"%Y-%m-%d %H:%M:%S"))

green_list['Pickup_hour'] = green_list.Pickup_dt.apply(lambda x:x.hour)



green_list['Trip_duration'] =((green_list.Dropoff_dt - green_list.Pickup_dt).apply(lambda x:x.total_seconds()/60.))

green_list['Speed'] =green_list.Trip_distance / (green_list.Trip_duration/60)    #to get the speed in mph

print(green_list["Speed"].median())    #printing average of speeds



new_speed = green_list[(~(green_list.Speed.isnull())) | (green_list.Speed<100)]    #removing outliers
green_list['Date'] =[d.day for d in green_list["Pickup_dt"]]               #getting just the date

one =pd.DataFrame()            #creating empty data frames

one =green_list.loc[(green_list['Date']>=1) & (green_list["Date"]<8)]      #getting data frame for days 1-7

two =pd.DataFrame()

two =green_list.loc[(green_list['Date']>=8) & (green_list["Date"]<15)]

three =pd.DataFrame()

three =green_list.loc[(green_list['Date']>=15) & (green_list["Date"]<22)]

four =pd.DataFrame()

four =green_list.loc[(green_list['Date']>=22) & (green_list["Date"]<=30)]
s1 =new_speed.loc[0:341474]

s2 =new_speed.loc[341475:702674]

s3 =new_speed.loc[702675:1065953]

s4 =new_speed.loc[1065954:1494926]



m1 =s1["Speed"].median()

m2 =s2["Speed"].median()

m3 =s3["Speed"].median()

m4 =s4["Speed"].median()



y =[m1,m2,m3,m4]

x =[1,2,3,4]

plt.scatter(x,y, label='skitscat', color='red', s=25, marker="o")      #plotting speed as a function of hours

plt.xlabel('Weeks')

plt.ylabel('Speed (mph)')

plt.title('Speed per Week')

plt.show()
plt.plot(green_list["Speed"])            #plotting speed as a function of hours

plt.xlabel('Hours (24 hours)')

plt.ylabel('Speed (mph)')

plt.title('Hourly distribution of Speed')

plt.ylim([0,40])

plt.xlim([0,23])

plt.show()