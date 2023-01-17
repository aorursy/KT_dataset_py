# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
#read in the raw files and merge the demographics dataset with the phone type dataset



gender_age_train=pd.read_csv("../input/gender_age_train.csv")

events=pd.read_csv("../input/events.csv")

phone_brand_device_model=pd.read_csv("../input/phone_brand_device_model.csv")

phone_brand_device_model_unique=phone_brand_device_model.drop_duplicates("device_id")
#join our events (location information) with the gender_age dataset



events_device=pd.merge(events,phone_brand_device_model,how="left")

events_device_gender_age=pd.merge(events_device,gender_age_train,on="device_id")

events_device_gender_age.head()
#the file takes in a lat,lon position and maps to a city using the a dataset contains city center points



def assignCity(lat,lon):

    for i in range(citiesLoc.shape[0]):

        targetLat= citiesLoc.iloc[i]["latitude"]

        targetLon= citiesLoc.iloc[i]["longitude"]

        if abs(lat-targetLat)<1.5 and abs(lon-targetLon)<1.5:

            return citiesLoc.iloc[i]["city"]

    return "No Match"
#data for 10 largest Chinese cities and their lat,lon center point

citiesLoc=pd.DataFrame.from_dict({'city': {0: 'Shanghai',

  1: 'Beijing',

  2: 'Wuhan',

  3: 'Chengdu',

  4: 'Tianjin',

  5: 'Shenyang',

  6: 'Xian',

  7: 'Chongqing',

  8: 'Guangzhou',

  9: 'Harbin'},

 'latitude': {0: 31.23,

  1: 39.93,

  2: 30.579999999999998,

  3: 30.670000000000002,

  4: 39.130000000000003,

  5: 41.799999999999997,

  6: 34.270000000000003,

  7: 29.57,

  8: 23.120000000000001,

  9: 45.75},

 'longitude': {0: 121.47,

  1: 116.40000000000001,

  2: 114.27,

  3: 104.06999999999999,

  4: 117.2,

  5: 123.45,

  6: 108.90000000000001,

  7: 106.58,

  8: 113.25,

  9: 126.65000000000001}})

citiesLoc.head()
#Remove all the events that have a lat,lon outside of China

mask=(events_device_gender_age.latitude>20) & (events_device_gender_age.longitude>70)

prep=events_device_gender_age[mask]

divider=prep[["latitude","longitude"]].drop_duplicates()



#Apply the assignCity function to all the different pairs of lat,lon in our dataset

divider["city"]=divider.apply(lambda row: assignCity(row["latitude"],row["longitude"]),axis=1)

divider.head()
#create some new variables that define each record as Older Male,Older Female, Younger Male, Younger Female

events_device_gender_age_cities=pd.merge(events_device_gender_age,divider,on=["latitude","longitude"])

events_device_gender_age_cities["age_binary"] = np.where(events_device_gender_age_cities.age>37, 'Older', 'Younger')

events_device_gender_age_cities["age_gender"] = events_device_gender_age_cities.gender.str.cat(events_device_gender_age_cities.age_binary)

events_device_gender_age_cities.age_gender=events_device_gender_age_cities.age_gender.replace(to_replace=["MOlder","FOlder","MYounger","FYounger"],

                                                   value=["Older Males","Older Females","Younger Males","Younger Female"])


#Pivot the primary dataset and get the number of phone users in each of the 4 groups above for the top 10 cities

events_gender_city=pd.pivot_table(events_device_gender_age_cities,index="city",columns="age_gender",values="device_id",aggfunc="count")

toPlot=events_gender_city.div(events_gender_city.sum(axis=1),axis="rows")

toPlot.plot(kind="barh",stacked=True)