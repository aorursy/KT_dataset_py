# To stop  displaying warning messages in output

import warnings

warnings.filterwarnings('ignore')

# To  collect garbage (delete files)

import gc

# To save dataset as pcikle file for future use

import pickle



import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

# for basic math operations like sqrt

import math

from math import sin, cos, sqrt, atan2, radians



import os

print(os.listdir("../input"))
input_data = pd.read_csv("../input/rapido-rides/ct_rr.csv")

input_data.shape
print("Data size before removing: ",input_data.shape)



# Check duplicated rows in train set

df = input_data[input_data.duplicated()]  # checks duplicate rows considering all columns

print("Number of duplicate observations: ", len(df))

del df

gc.collect();



#Dropping duplicates and keeping first occurence only

input_data.drop_duplicates(keep = 'first', inplace = True)



print("Data size after removing: ",input_data.shape)
input_data.head()
print("Number of unique customers: " ,input_data["number"].nunique()) #number of distinct customers = 1.7 lakhs
# new data frame with split value columns 

new = input_data["ts"].str.split(" ", n = 1, expand = True) 

  

# making separate first name column from new data frame 

input_data["raw_date"]= new[0] 

  

# making separate last name column from new data frame 

input_data["raw_time"]= new[1] 



input_data.head()
# new data frame with split value columns 

new = input_data["raw_date"].str.split("-", n = 2, expand = True) 

  

# making separate first name column from new data frame 

input_data["year"]= new[0] 

  

# making separate last name column from new data frame 

input_data["month"]= new[1] 



# making separate last name column from new data frame 

input_data["date"]= new[2] 



input_data.head()
# new data frame with split value columns 

new = input_data["raw_time"].str.split(":", n = 2, expand = True) 

  

# making separate first name column from new data frame 

input_data["hour"]= new[0]

#24:00 time system

  

# making separate last name column from new data frame 

input_data["minute"]= new[1] 



input_data.head()
#removing cols which are not reqd.

data = input_data.copy()

data.drop(["ts","raw_date","raw_time","number","minute"],axis=1, inplace=True)

data.head()



del input_data

gc.collect();
print("Is there any missing value? ",data.isna().sum().sum()>0)
def distance(pick_lat, pick_lng, drop_lat, drop_lng):

    

    # approximate radius of earth in km

    R = 6373.0

    

    s_lat = pick_lat*np.pi/180.0                      

    s_lng = np.deg2rad(pick_lng)     

    e_lat = np.deg2rad(drop_lat)                       

    e_lng = np.deg2rad(drop_lng)  

    

    d = np.sin((e_lat - s_lat)/2)**2 + np.cos(s_lat)*np.cos(e_lat) * np.sin((e_lng - s_lng)/2)**2

    

    return round(2 * R * np.arcsin(np.sqrt(d)),1)    #rounding off distance in km to 1 decimal place
data["distance"] = data.apply(lambda x: distance(x.pick_lat, x.pick_lng, x.drop_lat, x.drop_lng), axis=1)

data.head(3)
#assuming avg. bike speed of 35km/hrs, we can calculate time in min.

avg_speed = 35/60 #speed in km/minutes

data["ride_minutes"] = data["distance"].apply(lambda x: round(x/avg_speed,0))



print("Maximum ride distance covered in Km: ", data.distance.max())

print("Minimum ride distance covered in Km: ", data.distance.min())

print("Maximum ride time in mins: ",data.ride_minutes.max())

print("Minimum ride time in mins: ", data.ride_minutes.min())
plt.scatter(x=data['pick_lng'], y=data['pick_lat'])

plt.show()
plt.scatter(x=data['drop_lng'], y=data['drop_lat'])

plt.show()
data = data[(data.pick_lng <90) & (data.drop_lng <90) & (data.pick_lng >66) & (data.drop_lng >66) &

       (data.pick_lat <40) & (data.drop_lat <40) & (data.pick_lat >8) & (data.drop_lat >8)]



data.shape
#after removing above outliers based on lat and long



print("Maximum ride distance covered in Km: ", data.distance.max())

print("Minimum ride distance covered in Km: ", data.distance.min())

print("Maximum ride time in mins: ",data.ride_minutes.max())

print("Minimum ride time in mins: ", data.ride_minutes.min())
plt.scatter(x=data['pick_lng'], y=data['pick_lat'])

plt.show()
plt.scatter(x=data['drop_lng'], y=data['drop_lat'])

plt.show()
sns.countplot(x="hour", data=data)  #plot counts of variabe in bars form
data.boxplot('distance')
data.boxplot('ride_minutes')
def outlier_treatment(data):

    data_X = data.copy()

    for col in ['distance','ride_minutes']:

        percentiles = data_X[col].quantile([0.01,0.99]).values

        data_X[col][data_X[col] <= percentiles[0]] = percentiles[0]

        data_X[col][data_X[col] >= percentiles[1]] = percentiles[1]

    

    return data_X



data = outlier_treatment(data)

print("After outlier treatment: ", data.shape)
data.boxplot('distance')
data.boxplot('ride_minutes')
print("Before removing outliers: ", data.shape)

data = data[(data.ride_minutes<16) & (data.distance<11)]

print("After removing outliers: ", data.shape)
#after removing above outliers based on lat and long



print("Maximum ride distance covered in Km: ", data.distance.max())

print("Minimum ride distance covered in Km: ", data.distance.min())

print("Maximum ride time in mins: ",data.ride_minutes.max())

print("Minimum ride time in mins: ", data.ride_minutes.min())
print("Number of unique year present:",data.year.unique())
data['pick_lat'] = pd.to_numeric(data['pick_lat'], downcast='float')

data['pick_lng'] = pd.to_numeric(data['pick_lng'], downcast='float')

data['drop_lat'] = pd.to_numeric(data['drop_lat'], downcast='float')

data['drop_lng'] = pd.to_numeric(data['drop_lng'], downcast='float')

data['distance'] = pd.to_numeric(data['distance'], downcast='float')



data['month']= pd.to_numeric(data['month'], downcast='unsigned')

data['year']= pd.to_numeric(data['year'], downcast='unsigned')

data['hour']= pd.to_numeric(data['hour'], downcast='unsigned')



data['ride_minutes']= pd.to_numeric(data['ride_minutes'], downcast='unsigned')



data['date']= pd.to_numeric(data['date'], downcast='unsigned')

def change_month(row):

    if row['year']>2018:

        return (12 + row['month'])

    else:

        return row['month']

    
data["new_month"] = data.apply(change_month, axis=1)

data.head()
sns.countplot(x="new_month", data=data)  #plot counts of variabe in bars form
#creating price based features using price surge

peak_intensity = {0:1, 1 : 1.1, 	2 : 1.1, 	3 : 1.1, 	4 : 1.1, 	5 : 1, 	6 : 1, 	7 : 1.1, 	8 : 1.2, 	9 : 1.2, 	10 : 1.1, 	11 : 1, 

                  12 : 1, 	13 : 1, 	14 : 1, 	15 : 1, 	16 : 1, 	17 : 1.05, 	18 : 1.1, 	19 : 1.2, 	20 : 1.2, 	21 : 1.1, 	

                  22 : 1, 	23 : 1}



data['price_surge'] = data['hour'].map(peak_intensity)



#defining ride price in rs. assuming 7rs per km

def price_define(row):

    ride_price = row['distance']*7*row['price_surge']

    return ride_price



data["ride_price"] = data.apply(price_define, axis=1)

data.drop(['price_surge'],axis=1,inplace=True)

print(data.shape)

data.head(2)
model_file = "rapido_v_save"     #type name of file to be saved as

with open(model_file,mode='wb') as model_f:

    pickle.dump(data,model_f)